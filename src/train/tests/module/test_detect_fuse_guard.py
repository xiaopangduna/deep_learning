"""fit 期间不得 fuse Conv+BN。"""

from types import SimpleNamespace

from lightning.pytorch.trainer.states import TrainerFn

from lovely_deep_learning.module.object_detect import standalone_eval_should_fuse


def test_fit_sanity_and_val_do_not_fuse():
    for fn in (TrainerFn.FITTING, "fit"):
        trainer = SimpleNamespace(
            state=SimpleNamespace(fn=fn),
            sanity_checking=True,
            training=False,
        )
        assert standalone_eval_should_fuse(trainer) is False
        trainer.sanity_checking = False
        assert standalone_eval_should_fuse(trainer) is False


def test_standalone_validate_and_test_do_fuse():
    for fn in (TrainerFn.VALIDATING, TrainerFn.TESTING, "validate", "test"):
        trainer = SimpleNamespace(
            state=SimpleNamespace(fn=fn),
            sanity_checking=False,
            training=False,
        )
        assert standalone_eval_should_fuse(trainer) is True


def test_missing_fitting_attr_must_not_mean_fuse_during_fit():
    """回归：Trainer 没有 fitting 时不能当成可以 fuse。"""
    trainer = SimpleNamespace(state=SimpleNamespace(fn="fit"))
    assert not hasattr(trainer, "fitting")
    assert standalone_eval_should_fuse(trainer) is False
