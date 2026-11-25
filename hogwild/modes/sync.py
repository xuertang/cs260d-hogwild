"""Synchronous SGD: m workers, barrier, averaged gradient.

Zero staleness by construction — every gradient that enters an update was
computed at the same parameters. This is the baseline HOGWILD! has to beat on
wall-clock.

The round engine below also implements backup workers (modes/backup.py): the
only difference is how many workers run (m + b) for the same quorum (m). With
b = 0 the quorum is everyone, which is exactly a barrier.

One round, from a worker's point of view:

    r = current round                       (read without the lock)
    compute a gradient at the shared parameters
    take the lock
        round moved on while I was computing?  → drop the gradient, start over
        add my gradient to the shared accumulator
        if I am the m-th contributor:
            params -= eta * accumulator / m    (one update per round)
            advance the round, wake everyone
        else:
            wait until the round advances

Why winners are never stale: the update and the round increment happen under
the lock, update first. A worker that reads round r therefore reads it after
update r - 1 has fully landed, and if update r lands while it is still
computing, its gradient is rejected at the round check.

Accounting (agreed with the HOGWILD! mode): cfg.batch_size is per worker, so
one round applies m * batch_size examples in ONE update, where HOGWILD! makes
m updates from the same examples. eta is used as configured — it is not
rescaled by m; E4 is the experiment that varies it.
"""

from __future__ import annotations

import torch

from hogwild.instrument import Logger, StragglerSim
from hogwild.runner import Shared, mp_context, train
from hogwild.schema import Config, ResultRow, Task


def round_worker(worker_id: int, sh: Shared) -> dict:
    task, model, cfg = sh.task, sh.model, sh.cfg
    params = list(model.parameters())
    acc: list[torch.Tensor] = sh.extra["acc"]      # shared gradient sum, one tensor per parameter
    state = sh.extra["state"].numpy()              # shared int64[2] = [round, contributions so far]
    cond = sh.extra["cond"]
    stop = sh.stop.numpy()
    quorum, batch_size = cfg.m, cfg.batch_size
    step = cfg.eta / quorum                        # average of the quorum's gradients
    straggler = StragglerSim(cfg.straggler, cfg.seed, worker_id)
    updates = dropped = 0

    def finished() -> dict:
        return {"tau": {0: updates} if updates else {}, "dropped": dropped}

    epoch = 0
    while True:
        for batch in task.train_loader(sh.loader_seed(worker_id, epoch), batch_size):
            if stop[0]:
                return finished()
            r = int(state[0])
            grads = task.grads(model, batch)
            sh.probe.maybe_dump(worker_id, r, grads)
            straggler.sleep()

            with cond:
                if stop[0]:
                    return finished()
                if state[0] != r:                  # backup workers only: lost the race for round r
                    dropped += 1
                    continue
                for a, g in zip(acc, grads):
                    a.add_(g)
                state[1] += 1
                if state[1] < quorum:
                    while state[0] == r and not stop[0]:
                        # The barrier. No timeout on purpose: macOS has no sem_timedwait,
                        # so a timed wait is emulated by polling and costs milliseconds.
                        cond.wait()
                    continue
                with torch.no_grad():
                    for p, a in zip(params, acc):
                        p.add_(a, alpha=-step)
                        a.zero_()
                state[1] = 0
                state[0] = r + 1
                updates += 1
                sh.version.bump(worker_id)
                sh.examples.bump(worker_id, quorum * batch_size)
                if sh.examples.read() >= sh.example_limit:
                    stop[0] = 1
                cond.notify_all()
        epoch += 1


def run_rounds(task: Task, cfg: Config, log: Logger, n_workers: int) -> ResultRow:
    ctx = mp_context()
    cond = ctx.Condition()
    extra = {
        "cond": cond,
        "state": torch.zeros(2, dtype=torch.int64).share_memory_(),
        "acc": [torch.zeros_like(p).share_memory_() for p in task.model().parameters()],
    }

    def wake() -> None:                            # run aborted: release anyone waiting at the barrier
        with cond:
            cond.notify_all()

    return train(task, cfg, log, round_worker, n_workers=n_workers, extra=extra, on_stop=wake)


class SyncMode:
    def run(self, task: Task, cfg: Config, log: Logger) -> ResultRow:
        return run_rounds(task, cfg, log, n_workers=cfg.m)
