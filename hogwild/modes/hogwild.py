"""HOGWILD!: m workers, no barrier, lock-free in-place adds.

Covers both rows of the mode table in README §4:
    hogwild            natural staleness, measured
    hogwild --tau K    staleness of about K versions injected (cfg.tau_inject > 0)

The same worker loop with a single worker is plain serial SGD, which is how
modes/serial.py is implemented.
"""

from __future__ import annotations

import torch

from hogwild.instrument import DelayQueue, Logger, StragglerSim, TauHistogram
from hogwild.runner import Shared, train
from hogwild.schema import Config, ResultRow, Task


def effective_eta(cfg: Config) -> float:
    """eta, or eta / (1 + tau) with tau = the configured injection K.

    The configured K rather than the tau measured for each update: it is
    constant over the run (so the comparison with the fixed-eta run is clean)
    and it is the quantity a practitioner would actually know in advance.
    With tau_inject = 0 both rules give eta.
    """
    return cfg.eta / (1 + cfg.tau_inject) if cfg.eta_scale == "1/(1+tau)" else cfg.eta


def hogwild_worker(worker_id: int, sh: Shared) -> dict:
    """One worker process. The order of the steps below is what defines tau."""
    task, model, cfg = sh.task, sh.model, sh.cfg
    params = list(model.parameters())          # shared storage
    version, examples, probe = sh.version, sh.examples, sh.probe
    stop = sh.stop.numpy()
    limit, batch_size = sh.example_limit, cfg.batch_size
    eta = effective_eta(cfg)
    hist = TauHistogram()
    straggler = StragglerSim(cfg.straggler, cfg.seed, worker_id)
    delay = DelayQueue(cfg.tau_inject, sh.n_workers)

    epoch = 0
    while True:
        for batch in task.train_loader(sh.loader_seed(worker_id, epoch), batch_size):
            if stop[0] or examples.read() >= limit:
                return {"tau": dict(hist.counts)}

            v_read = version.read()
            grads = task.grads(model, batch)       # reads the shared params, no lock
            probe.maybe_dump(worker_id, v_read, grads)
            straggler.sleep()

            due = delay.push((v_read, grads))      # tau injection: apply an older gradient
            if due is None:
                continue                           # queue still filling
            v_read, grads = due

            hist.record(version.read() - v_read)   # tau = v_write - v_read
            with torch.no_grad():
                for p, g in zip(params, grads):
                    p.add_(g, alpha=-eta)          # in place on shared memory, no lock
            version.bump(worker_id)
            examples.bump(worker_id, batch_size)
        epoch += 1


class HogwildMode:
    def run(self, task: Task, cfg: Config, log: Logger) -> ResultRow:
        return train(task, cfg, log, hogwild_worker, n_workers=cfg.m)
