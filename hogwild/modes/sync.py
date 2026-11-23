"""Synchronous SGD: m workers, barrier, averaged gradient (stream C).

Zero staleness by construction — every gradient in a step is computed at the
same parameters. This is the baseline HOGWILD! has to beat on wall-clock.

STUB: signatures and comments only. Nothing here is implemented yet.
"""

from __future__ import annotations

from hogwild.instrument import Logger
from hogwild.schema import Config, ResultRow, Task


def _worker(worker_id: int, *shared_state) -> dict:
    """Body of one worker process.

    One round:
        wait at the start-of-round barrier
        read shared params (all workers see the same version)
        compute gradient on own minibatch
        straggler.sleep()                 # E3: the slowest worker sets the round time
        write gradient into own slot of a shared (m x num_params) buffer
        wait at the end-of-round barrier

    The coordinator (main process, or worker 0 as the barrier action) then
    averages the m slots and applies ONE update: p -= eta * mean(g_i).

    Reuse from stream A: runner.pin_worker, runner.SharedParams,
    runner.WorkerPool, instrument.StragglerSim.
    """
    raise NotImplementedError


class SyncMode:
    def run(self, task: Task, cfg: Config, log: Logger) -> ResultRow:
        """Main-process side.

        - Barrier: multiprocessing.Barrier(m) is the direct translation. Its
          cost per round IS part of what we measure (README §9, "overhead
          ceiling"), so do not optimise it away — but do report it.
        - Batch accounting (agree with stream A): README says "same total batch
          as hogwild". If cfg.batch_size is per worker, one sync step consumes
          m * batch_size examples and makes one update, where hogwild makes m
          updates. Epoch counting must use examples processed, not steps.
        - Step size: averaging m gradients at the serial-tuned eta is a
          large-batch step. E4 sweeps eta in {0.5x, 2x} for this reason; do not
          silently rescale eta by m here.
        - ResultRow: tau_hist = {0: num_updates}; interference_mean = NaN
          (or compute the cosine between workers' same-round gradients — it is
          free here and gives analysis a zero-staleness reference. Check with D).
        """
        raise NotImplementedError
