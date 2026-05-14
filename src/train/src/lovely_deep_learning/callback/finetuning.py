from __future__ import annotations

from typing import Any, Sequence

import lightning.pytorch as pl
from lightning.pytorch.callbacks import BaseFinetuning
from lightning.pytorch.utilities.rank_zero import rank_zero_info
from typing_extensions import override

from lovely_deep_learning.model.DAGNet import DAGNet


class DAGNetMilestoneUnfreezeFinetuning(BaseFinetuning):
    """Milestone finetuning for ``DAGNet``: list **unfreeze_layers** (few), freeze the rest until one epoch.

    ``unfreeze_layers`` names ``DAGNet.layers`` keys that stay ``requires_grad=True`` from epoch 0
    (typically the last few blocks). All **other** layer keys are frozen first; at ``unfreeze_epoch``
    they are unfrozen together as **one** new param group. The new group's lr matches
    ``optimizer.param_groups[0]["lr"]`` at that moment.

    ``unfrozen_layer_names`` lists the DAG keys that were released at the milestone (i.e. the
    complement of ``unfreeze_layers``), sorted.
    """

    def __init__(
        self,
        unfreeze_layers: Sequence[str],
        unfreeze_epoch: int,
    ) -> None:
        super().__init__()
        self._unfreeze_layers: tuple[str, ...] = tuple(dict.fromkeys(unfreeze_layers))
        self._unfreeze_epoch = int(unfreeze_epoch)

        self._frozen_layers: tuple[str, ...] = ()
        self._did_unfreeze: bool = False
        self._unfrozen_names: list[str] = []

        if not self._unfreeze_layers:
            raise ValueError(
                "`unfreeze_layers` must name at least one `DAGNet.layers` key to keep training from epoch 0."
            )
        if self._unfreeze_epoch < 0:
            raise ValueError("`unfreeze_epoch` must be >= 0.")

    @property
    def unfrozen_layer_names(self) -> tuple[str, ...]:
        """DAG ``layers`` keys unfrozen at the milestone (empty until that epoch runs)."""
        return tuple(self._unfrozen_names)

    @override
    def state_dict(self) -> dict[str, Any]:
        out = super().state_dict()
        out["dagnet_did_unfreeze"] = self._did_unfreeze
        out["dagnet_unfrozen_names"] = list(self._unfrozen_names)
        out["dagnet_frozen_layers"] = list(self._frozen_layers)
        return out

    @override
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        sd = dict(state_dict)
        self._did_unfreeze = bool(sd.pop("dagnet_did_unfreeze", False))
        self._unfrozen_names = list(sd.pop("dagnet_unfrozen_names", []))
        self._frozen_layers = tuple(sd.pop("dagnet_frozen_layers", []))
        super().load_state_dict(sd)

    def _dag(self, pl_module: pl.LightningModule) -> DAGNet:
        model = getattr(pl_module, "model", None)
        if not isinstance(model, DAGNet):
            raise TypeError(
                f"{self.__class__.__name__} requires `pl_module.model` to be a DAGNet, "
                f"got {type(model).__name__!r}."
            )
        return model

    def _derive_frozen(self, dag: DAGNet) -> tuple[str, ...]:
        all_keys = set(dag.layers.keys())
        tab = set(self._unfreeze_layers)
        unknown = sorted(tab - all_keys)
        if unknown:
            known = ", ".join(sorted(all_keys))
            raise KeyError(
                f"Unknown `unfreeze_layers` DAG layer name(s) {unknown!r}. "
                f"Valid `DAGNet.layers` keys: {known}"
            )
        frozen = sorted(all_keys - tab)
        return tuple(frozen)

    @override
    def freeze_before_training(self, pl_module: pl.LightningModule) -> None:
        dag = self._dag(pl_module)
        self._frozen_layers = self._derive_frozen(dag)
        if not self._frozen_layers:
            return
        to_freeze = [dag.layers[n] for n in self._frozen_layers]
        self.freeze(to_freeze, train_bn=True)

    @override
    def finetune_function(
        self,
        pl_module: pl.LightningModule,
        epoch: int,
        optimizer: Any,
    ) -> None:
        if self._did_unfreeze or not self._frozen_layers:
            return
        if epoch != self._unfreeze_epoch:
            return

        dag = self._dag(pl_module)
        modules = [dag.layers[n] for n in self._frozen_layers]
        head_lr = float(optimizer.param_groups[0]["lr"])
        self.unfreeze_and_add_param_group(
            modules=modules,
            optimizer=optimizer,
            lr=head_lr,
            initial_denom_lr=10.0,
            train_bn=True,
        )
        self._unfrozen_names = list(self._frozen_layers)
        self._did_unfreeze = True
        new_lr = float(optimizer.param_groups[-1]["lr"])
        rank_zero_info(
            "%s: epoch %s — unfrozen DAG layers (single param group, lr=%s from optimizer): %s",
            self.__class__.__name__,
            epoch,
            new_lr,
            self._unfrozen_names,
        )
