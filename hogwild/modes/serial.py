"""
Role B Task

Single-worker SGD baseline and standalone serial experiment CLI.

Defines the four models, from Sparse Logisitic Regression

Reads YAML file and connects to tasks.py to collect task and retrieve model from models.py
Results saved to results/serial_reference_*.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from itertools import islice
from time import perf_counter
from typing import TYPE_CHECKING

import torch

from hogwild.schema import Config, ResultRow, Task, append_row
from hogwild.tasks import DATA_ROOT, get_task, training_loss

if TYPE_CHECKING:
    from hogwild.instrument import Logger


class SerialMode:
    def __init__(self, smoke: bool = False) -> None:
        self.smoke = smoke

    def run(self, task: Task, cfg: Config, log: Logger | None) -> ResultRow:
        if cfg.mode != "serial" or cfg.m != 1:
            raise ValueError("Serial SGD requires mode='serial' and m=1")
        if cfg.max_epochs <= 0 or cfg.batch_size <= 0:
            raise ValueError("Serial SGD requires positive max_epochs and batch_size")
        if not math.isfinite(cfg.eta) or cfg.eta <= 0:
            raise ValueError("Serial SGD requires a finite positive eta")

        torch.set_num_threads(1)
        torch.manual_seed(cfg.seed)
        model = task.model()
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.eta, foreach=False)
        params = tuple(p.detach() for p in model.parameters())
        train_size = getattr(task, "train_size", None)
        target = task.target_loss if not self.smoke else math.nan
        curve_epoch: list[tuple[float, float]] = []
        curve_wall: list[tuple[float, float]] = []
        time_to_target = epoch_to_target = math.inf

        def record(epoch: float, wall_s: float, loss: float, accuracy: float) -> None:
            nonlocal time_to_target, epoch_to_target
            curve_epoch.append((epoch, loss))
            curve_wall.append((wall_s, loss))
            if loss <= target and math.isinf(time_to_target):
                time_to_target, epoch_to_target = wall_s, epoch
            if log is not None:
                log.eval_point(epoch, wall_s, loss, accuracy)

        # Warm the dataset and measure the initial model before starting the clock.
        loss, accuracy = task.evaluate(params)
        record(0.0, 0.0, loss, accuracy)
        start = perf_counter()
        next_eval = start + 2.0
        updates = 0
        epochs = 1 if self.smoke else cfg.max_epochs
        for epoch in range(epochs):
            batches = task.train_loader(cfg.seed + epoch, cfg.batch_size)
            if self.smoke:
                batches = islice(batches, 2)
            examples = 0
            pending = []
            for x, y in batches:
                optimizer.zero_grad(set_to_none=True)
                training_loss(task, model, x, y).backward()
                optimizer.step()
                updates += 1
                examples += y.shape[0]

                now = perf_counter()
                if now >= next_eval and examples != train_size:
                    loss, accuracy = task.evaluate(params)
                    if train_size is None:
                        pending.append((examples, now - start, loss, accuracy))
                    else:
                        record(epoch + examples / train_size, now - start, loss, accuracy)
                    next_eval = perf_counter() + 2.0

            if examples == 0:
                raise ValueError(f"Task {task.name!r} returned an empty training epoch")
            size = train_size or examples
            for count, wall_s, loss, accuracy in pending:
                record(epoch + count / size, wall_s, loss, accuracy)
            now = perf_counter()
            loss, accuracy = task.evaluate(params)
            record(epoch + examples / size, now - start, loss, accuracy)
            next_eval = perf_counter() + 2.0

        run_id = getattr(log, "run_id", None)
        if run_id is None:
            config_bytes = json.dumps(cfg.to_dict(), sort_keys=True).encode()
            digest = hashlib.sha256(config_bytes).hexdigest()[:10]
            run_id = f"{task.name}-serial-m1-s{cfg.seed}-{digest}"
            if self.smoke:
                run_id += "-smoke"
        return ResultRow(
            run_id=run_id,
            config=cfg,
            time_to_target_s=time_to_target,
            epochs_to_target=epoch_to_target,
            final_test_acc=accuracy,
            tau_hist={0: updates},
            interference_mean=math.nan,
            curve_epoch=curve_epoch,
            curve_wall=curve_wall,
            wall_total_s=perf_counter() - start,
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--eta", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--data-root", default=DATA_ROOT)
    parser.add_argument("--smoke", action="store_true", help="Run up to two updates")
    parser.add_argument("--out", help="Append the result to this JSONL file")
    args = parser.parse_args(argv)
    overrides = {name: value for name, value in (
        ("eta", args.eta), ("seed", args.seed)
    ) if value is not None}
    cfg = Config.from_yaml(args.config, **overrides)
    task = get_task(cfg.task, data_root=args.data_root)
    row = SerialMode(smoke=args.smoke).run(task, cfg, log=None)
    if args.out:
        append_row(args.out, row)
    print(row.to_json())


if __name__ == "__main__":
    main()
