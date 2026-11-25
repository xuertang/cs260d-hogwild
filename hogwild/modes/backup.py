"""Sync SGD with backup workers: m + b workers, first m gradients win.

Chen, Monga, Bengio, Jozefowicz (2016). Each round, m + b workers compute a
gradient at the same parameters; the first m to arrive are averaged and the
other b are dropped. Staleness stays zero and a round no longer waits for the
slowest worker — the H3 contender against HOGWILD!.

The round logic lives in modes/sync.py (`round_worker`); this mode only starts
b extra workers. What happens to the b losers of a round:

  - Their gradient is DROPPED, never applied late. It is tagged with the round
    it was computed in and rejected once that round has closed. Applying it
    would reintroduce staleness and make this a different algorithm.
  - A loser cannot be interrupted mid-computation. It finishes, sees that the
    round has moved on, discards, and starts again at the current round — so
    a slow worker may sit out several rounds, as in the paper.

Each row's log line reports how many gradients were dropped (wasted work).
Note for the write-up: at the same m, this mode uses b more cores than sync
and HOGWILD! do.
"""

from __future__ import annotations

from hogwild.instrument import Logger
from hogwild.modes.sync import run_rounds
from hogwild.schema import Config, ResultRow, Task


class BackupMode:
    def run(self, task: Task, cfg: Config, log: Logger) -> ResultRow:
        if cfg.b < 1:
            raise ValueError("backup mode needs b >= 1 (with b = 0 it is sync mode)")
        return run_rounds(task, cfg, log, n_workers=cfg.m + cfg.b)
