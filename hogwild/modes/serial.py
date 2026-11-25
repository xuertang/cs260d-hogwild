"""Serial SGD: the reference curve (README §4). eta is tuned here.

Implemented as the HOGWILD! worker loop with exactly one worker. With nobody
else writing, read → gradient → write is ordinary sequential SGD and every
update has tau = 0. Running it through the same worker process, counters and
evaluator as the parallel modes means T1 in speed-up = T1 / Tm carries the
same fixed overheads as Tm.
"""

from __future__ import annotations

from hogwild.instrument import Logger
from hogwild.modes.hogwild import hogwild_worker
from hogwild.runner import train
from hogwild.schema import Config, ResultRow, Task


class SerialMode:
    def run(self, task: Task, cfg: Config, log: Logger) -> ResultRow:
        if cfg.m != 1:
            raise ValueError(f"serial mode needs m = 1, got m = {cfg.m}")
        return train(task, cfg, log, hogwild_worker, n_workers=1)
