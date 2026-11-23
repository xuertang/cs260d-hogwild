"""HOGWILD! on dense, non-convex problems (CS 260D final project).

Package map (owners from README §8):

    schema.py       all      Config / ResultRow / Task / Mode contracts
    runner.py       A        process pool, shared-memory params, evaluator
    instrument.py   A        tau counter, delay injection, stragglers, probe
    modes/hogwild.py A       lock-free in-place updates
    tasks.py        B        task registry, loaders, target loss, eval   (not yet created)
    models.py       B        LogReg, Softmax, MLP, SmallCNN              (not yet created)
    modes/serial.py B        reference mode                              (not yet created)
    modes/sync.py   C        m workers + barrier
    modes/backup.py C        m + b workers, first m win
    sweep.py        C        sweep YAML → runs → results/*.jsonl
    stub_task.py    —        synthetic Task so A/C can smoke-test before B lands

Keep this file import-free: workers are separate processes and re-import the
package, so anything heavy here is paid once per worker.
"""
