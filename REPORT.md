# Sequence modelling benchmark report

## Scope

This project compares four sequence encoders implemented from PyTorch
primitives: a vanilla RNN, LSTM, simplified Mamba-style selective state-space
model, and Transformer. All models use the same AG News preprocessing, split,
masked mean pooling, minibatch order, AdamW optimizer, scheduler, clipping,
early stopping rule, and evaluator.

The checked-in measurements are a reproducible **quick experiment**, not a
claim of converged performance. The default configuration is the stronger
20,000-example run; setting `max_train_samples: null` uses all 120,000 official
training examples.

## Protocol

- Dataset: AG News, four balanced topic labels (World, Sports, Business,
  Sci/Tech).
- Split construction: seed-42 stratified validation split from the official
  training data; official test data remains separate.
- Measured subset: 1,000 train, 400 validation, 400 test examples.
- Vocabulary: built only from the selected training rows, minimum frequency 2,
  4,014 tokens.
- Input: lowercase word/punctuation tokens, maximum length 48.
- Training: three epochs, batch size 64, AdamW at 1e-3, weight decay 0.01,
  gradient clipping at 1.0, validation-loss scheduling and checkpointing.
- Hardware: CPU; peak GPU memory is therefore not applicable and is stored as
  0 by the machine-readable schema.
- Metric: multiclass cross-entropy and accuracy. Perplexity is not applicable
  to this classification task.

## Measured quick-run results

| Model | Parameters | Best val loss | Val accuracy | Test loss | Test accuracy | Train time (s) | Inference examples/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| RNN | 272,740 | 1.2463 | 0.4350 | 1.2434 | **0.4450** | 3.48 | 3,015 |
| LSTM | 319,108 | 1.3445 | 0.3425 | 1.3429 | 0.3350 | 5.08 | 2,144 |
| Selective SSM | 284,853 | 1.2597 | **0.4475** | 1.2558 | 0.4350 | 13.90 | 681 |
| Transformer | 357,252 | 1.3404 | 0.3275 | 1.3284 | 0.3600 | 5.30 | **3,285** |

Raw metrics, histories, checkpoints, tables, and plots are under
`results/quick/`. The seconds-long offline integration check is under
`results/smoke/` and must not be interpreted as model quality evidence.

## Discussion

The RNN had the best test accuracy in this deliberately short run, while the
SSM had the best validation accuracy. The Transformer achieved the highest
measured inference throughput. The educational SSM was slowest because its
selective scan is a readable Python time loop; it does not include Mamba's
parallel scan, kernel fusion, or custom CUDA implementation. These observations
apply only to the tested CPU, subset, sequence length, and three-epoch budget.

The LSTM and Transformer were still improving under this small budget, so the
ranking is not evidence that their architectures are intrinsically weaker.
Useful next experiments are the default 20,000-example configuration, at least
three random seeds, and model-specific learning-rate tuning recorded as separate
run IDs. Capacity is similar but not identical; parameter counts are reported
so accuracy/efficiency trade-offs remain visible.

## Iteration log

| Run | Motivation | Change | Outcome |
|---|---|---|---|
| Smoke | Validate the complete system offline | 12 built-in examples, one epoch | All four models trained, restored checkpoints, evaluated, and plotted |
| Quick | Obtain genuine measured results cheaply | AG News 1,000/400/400, compact widths, three epochs | Produced the table above; exposed under-training and slow sequential SSM scan |
| Default (recommended next) | Test whether rankings persist | 20,000 train rows, length 128, up to 15 epochs | Configuration provided; not run on this CPU session |

## Limitations

The checked-in experiment uses one dataset, one seed, a small balanced subset,
short sequences, and a limited tuning budget. Timing is hardware-specific. The
models are educational implementations rather than optimized framework kernels,
especially the SSM. No bidirectional recurrent variants are included, while the
Transformer has non-causal global attention; this context-direction difference
should be considered when interpreting a full classification comparison.

## Reproduction

```powershell
python main.py benchmark --config configs/quick.yaml
python main.py benchmark --config configs/benchmark.yaml
```

All table and plot values are generated from saved JSON histories. Architecture
and tutorial references are documented in [RESEARCH.md](RESEARCH.md).
