"""Sync SGD with backup workers: m + b workers, first m gradients win (stream C).

Chen, Monga, Bengio, Jozefowicz (2016). Each round, m + b workers compute a
gradient at the same parameters; the coordinator averages the first m to
arrive and drops the other b. Staleness stays zero and the round no longer
waits for the slowest worker — this is the H3 contender against HOGWILD!.

STUB: signatures and comments only. Nothing here is implemented yet.
"""

from __future__ import annotations

from hogwild.instrument import Logger
from hogwild.schema import Config, ResultRow, Task


def _worker(worker_id: int, *shared_state) -> dict:
    """Body of one worker process.

    Same round structure as modes/sync.py, except the end of a round is "m
    gradients have arrived", not "everyone has arrived".

    The hard part is the b losers:
      - A loser's gradient belongs to the old round and must be DROPPED, not
        applied late (applying it would reintroduce staleness and make this a
        different algorithm). Tag every gradient with its round number and
        have the coordinator discard mismatches.
      - A loser that is mid-computation cannot be interrupted from Python. It
        finishes, sees the round has moved on, discards, and rejoins — so a
        straggler may sit out several rounds. That is faithful to the paper.
      - With straggler="none" the same b workers should not lose every round;
        log wins per worker to check.
    """
    raise NotImplementedError


class BackupMode:
    def run(self, task: Task, cfg: Config, log: Logger) -> ResultRow:
        """Main-process side.

        - Workers: cfg.m + cfg.b processes, i.e. b more cores than sync and
          hogwild use at the same m. E3 compares at m = 8 with b in {1, 2};
          say explicitly in the report that backup gets 9–10 cores.
        - Arrival order: a multiprocessing.Queue of (round, worker_id) tokens
          plus per-worker gradient slots in shared memory keeps the gradients
          themselves out of the pickle path.
        - Coordinator: take the first m tokens for the current round, average
          those slots, apply one update, advance the round counter (shared),
          release the workers.
        - cfg.b == 0 should behave exactly like SyncMode — a cheap
          cross-check between the two implementations.
        - ResultRow: tau_hist = {0: num_updates}; also worth logging the
          fraction of computed gradients that were dropped (wasted work).
          That is not in the schema; keep it in the Logger output unless
          analysis asks for it (→ schema PR).
        """
        raise NotImplementedError
