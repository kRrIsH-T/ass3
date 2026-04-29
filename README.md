# From-scratch sequence model benchmark

An educational, fair comparison of manually implemented RNN, LSTM, selective SSM,
and Transformer encoders on AG News. All architectures share the exact split,
vocabulary, minibatch order, optimizer, scheduler, clipping, early stopping, and
evaluation pipeline.

## Quick start

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python main.py benchmark --smoke
```

The offline smoke run trains every architecture for one epoch and writes its
checkpoints, metrics, tables, and plots to `results/smoke`. For the automatically
downloaded AG News benchmark:

```powershell
.venv\Scripts\python main.py benchmark --config configs/benchmark.yaml
```

Use `--models rnn transformer` to run a subset. The default configuration uses a
balanced 20,000-row training subset for practical local runs; set
`max_train_samples: null` for all 120,000 training examples. See
[`benchmark/README.md`](benchmark/README.md) for the data and model interface.

For a short, real-data comparison before a full run, use
`python main.py benchmark --config configs/quick.yaml`. Unlike `--smoke`, this
downloads AG News and reports measured results on balanced train/validation/test
subsets.
