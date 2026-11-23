"""HOGWILD!: m workers, no barrier, lock-free in-place adds (stream A).

Covers both rows of the mode table in README §4:
    hogwild            natural staleness, measured
    hogwild --tau K    staleness >= K injected (cfg.tau_inject > 0)

STUB: signatures and comments only. Nothing here is implemented yet.
"""

from __future__ import annotations

from hogwild.instrument import Logger
from hogwild.schema import Config, ResultRow, Task


def _worker(worker_id: int, *shared_state) -> dict:
    """Body of one worker process. Returns its stats dict to WorkerPool.join().

    One step, in order — the order is what defines tau, so keep it exact:

        pin_worker(worker_id)                       # once, before the loop
        for (x, y) in task.train_loader(seed, batch_size):
            v_read = version.read()
            shared.load_into(local_model)           # read (possibly torn) params
            loss = criterion(local_model(x), y)
            loss.backward()                         # gradient at the stale point
            probe.maybe_dump(worker_id, step, grads)
            straggler.sleep()                       # E3 only; no-op otherwise
            # cfg.tau_inject > 0: delay mechanism goes here (see the deadlock
            # warning on instrument.wait_for_staleness before writing this)
            v_write = version.read()
            tau = v_write - v_read
            hist.record(tau)
            with torch.no_grad():
                for p, g in zip(shared.tensors(), grads):
                    p.add_(g, alpha=-eta_eff)       # in place, no lock
            version.bump(worker_id)
            if stop_flag is set: break

    Open points:
      - Seeding: worker i uses a seed derived from (cfg.seed, worker_id) so
        workers do not all walk the data in the same order. Ties into the
        OPEN question on Task.train_loader (shared vs. sharded data).
      - Where the loss criterion comes from: the Task contract exposes model(),
        train_loader() and evaluate() but no training loss. Either the model's
        forward returns logits and every mode hard-codes cross-entropy (true
        for all four tasks, but T1 also needs its l2 term), or the contract
        needs a `loss(model, batch)` method → schema PR. Raise with stream B.
      - eta_eff for cfg.eta_scale == "1/(1+tau)": README E2 writes eta/(1+tau).
        Is tau the configured K (constant per run, simplest, and what the
        theory's step-size rule uses) or the tau measured for this very update
        (adaptive)? Proposal: the configured K; note it in the report.
      - Batch accounting: README says sync averages "the same total batch as
        hogwild". So if cfg.batch_size is the per-worker batch here, sync must
        average m such batches per step. Agree on this with stream C.
    """
    raise NotImplementedError


class HogwildMode:
    def run(self, task: Task, cfg: Config, log: Logger) -> ResultRow:
        """Main-process side.

        1. model = task.model(); shared = SharedParams(model)
        2. Build the shared instrumentation: VersionCounter(cfg.m),
           InterferenceProbe(num_params), stop flag, examples-processed counter.
        3. pool = WorkerPool(cfg.m, _worker, args=(...)); pool.start(); pool.release()
        4. Evaluator loop until target reached and/or cfg.max_epochs done;
           then pool.stop(); stats = pool.join().
        5. Assemble ResultRow: curves from the Logger, tau_hist from
           TauHistogram.merge over worker stats, interference_mean from the
           probe, wall_total_s from the release() timestamp.

        param_dist_to_seed0 cannot be computed inside one run (it needs the
        seed-0 run's final params). TODO: save final params next to the row,
        e.g. results/params/<run_id>.pt, and let analysis fill the field in.
        """
        raise NotImplementedError
