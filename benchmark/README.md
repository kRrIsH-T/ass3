# Benchmark infrastructure

`benchmark` keeps data and evaluation policy independent of the model implementations.
AG News is downloaded from the public CharCNN mirror into `outputs/data/ag_news/raw`.
If the network is unavailable, `--smoke` (or `force_fallback: true`) uses a balanced,
built-in 12-example corpus to validate the entire pipeline.

Run all configured models:

```bash
python -m benchmark --config configs/benchmark.yaml
```

Run a quick offline integration check or select architectures:

```bash
python -m benchmark --config configs/benchmark.yaml --smoke
python -m benchmark --config configs/benchmark.yaml --models rnn transformer
```

Each model factory is configured as `module:callable`. The runner injects constructor
arguments such as `vocab_size`, `num_classes`, `padding_idx`, and `max_length` when
the callable accepts them. A classifier may implement either `forward(tokens)`,
`forward(tokens, lengths=...)`, or `forward(tokens, padding_mask=...)`.

Every model gets a newly seeded loader with the same split and minibatch order.
Outputs include best checkpoints, per-run JSON metrics, CSV/Markdown comparison
tables, individual training curves, and an aggregate accuracy plot. Checkpoint
selection and early stopping are based only on validation loss; the test set is
evaluated once after restoring the best checkpoint.
