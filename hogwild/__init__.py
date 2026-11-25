"""HOGWILD! on dense, non-convex problems (CS 260D final project).

    schema.py        Config / ResultRow / Task / Mode contracts
    tasks.py         task registry, data loaders, gradients, evaluation, download
    models.py        LogReg (sparse), Softmax, MLP, SmallCNN
    runner.py        worker processes, shared-memory model, evaluator, CLI
    instrument.py    tau counter, delay injection, straggler model, interference probe
    modes/           serial, sync, backup, hogwild
    tune.py          serial step-size sweep → configs/reference.yaml
    sweep.py         sweep YAML → runs → results/*.jsonl, with resume

Keep this file import-free: workers are separate processes and re-import the
package, so anything heavy here is paid once per worker.
"""
