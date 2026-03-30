from __future__ import annotations

import csv
import os
from datetime import datetime
from pathlib import Path

import lightning.pytorch as pl


def _read_vm_rss_kb(pid: int) -> int:
    """Return VmRSS in kB for pid; 0 when unavailable."""
    try:
        with open(f"/proc/{pid}/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    # Format: VmRSS:   123456 kB
                    parts = line.split()
                    return int(parts[1])
    except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError):
        return 0
    return 0


def _read_available_mem_kb() -> int:
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1])
    except (FileNotFoundError, PermissionError, ValueError):
        return 0
    return 0


def _read_children_pids(pid: int) -> list[int]:
    """Read direct children pids from procfs."""
    try:
        with open(f"/proc/{pid}/task/{pid}/children", "r", encoding="utf-8") as f:
            raw = f.read().strip()
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return []
    if not raw:
        return []
    out: list[int] = []
    for token in raw.split():
        try:
            out.append(int(token))
        except ValueError:
            continue
    return out


class MemoryMonitorCallback(pl.Callback):
    """
    Periodically logs CPU memory usage (RSS) to CSV.

    It records the trainer process memory and direct child process memory
    (typically DataLoader workers), so long runs can be diagnosed post-mortem.
    """

    def __init__(self, save_dir: str = "logs/memory", filename: str = "memory_usage.csv", log_every_n_steps: int = 200):
        super().__init__()
        self.save_dir = Path(save_dir)
        self.filename = filename
        self.log_every_n_steps = max(1, int(log_every_n_steps))
        self.csv_path: Path | None = None
        self._last_logged_step = -1

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.save_dir / self.filename
        with self.csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "timestamp",
                    "epoch",
                    "global_step",
                    "batch_idx",
                    "main_pid",
                    "main_rss_kb",
                    "children_count",
                    "children_total_rss_kb",
                    "children_max_rss_kb",
                    "mem_available_kb",
                ]
            )
        self._write_row(trainer=trainer, batch_idx=-1)

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        step = trainer.global_step
        if step == self._last_logged_step:
            return
        if step % self.log_every_n_steps != 0:
            return
        self._write_row(trainer=trainer, batch_idx=batch_idx)

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._write_row(trainer=trainer, batch_idx=-1)

    def _write_row(self, trainer: pl.Trainer, batch_idx: int) -> None:
        if self.csv_path is None:
            return

        main_pid = os.getpid()
        main_rss_kb = _read_vm_rss_kb(main_pid)
        child_pids = _read_children_pids(main_pid)
        child_rss = [_read_vm_rss_kb(pid) for pid in child_pids]
        child_rss = [v for v in child_rss if v > 0]
        mem_available_kb = _read_available_mem_kb()

        row = [
            datetime.now().isoformat(timespec="seconds"),
            trainer.current_epoch,
            trainer.global_step,
            batch_idx,
            main_pid,
            main_rss_kb,
            len(child_rss),
            sum(child_rss),
            max(child_rss) if child_rss else 0,
            mem_available_kb,
        ]
        with self.csv_path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)
        self._last_logged_step = trainer.global_step
