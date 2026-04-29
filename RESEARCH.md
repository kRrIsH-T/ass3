# Research notes and implementation references

This benchmark uses tutorials for orientation, but treats original papers,
official repositories, and framework documentation as the source of truth. No
reference implementation should be copied wholesale; the project implements the
sequence mechanisms itself with PyTorch tensor operations.

## Dataset choice: AG News

AG News is a practical fit for this project: it is a moderately sized sequence
classification dataset with four balanced topic labels (World, Sports,
Business, and Sci/Tech). The standard release contains 120,000 training examples
and 7,600 test examples. It downloads automatically and is large enough to show
meaningful differences without requiring language-model-scale compute.

Primary references:

- Xiang Zhang, Junbo Zhao, and Yann LeCun, [*Character-level Convolutional
  Networks for Text Classification*](https://papers.nips.cc/paper_files/paper/2015/hash/250cf8b51c773f3f8dc8b4be867a9a02-Abstract.html),
  NeurIPS 2015. This paper documents the constructed AG News classification
  benchmark.
- [Hugging Face `fancyzhx/ag_news` dataset
  card](https://huggingface.co/datasets/fancyzhx/ag_news), the practical,
  auto-downloadable distribution used by many current projects.
- [TorchText AG_NEWS documentation](https://docs.pytorch.org/text/stable/datasets.html),
  useful for independently checking the standard split sizes. TorchText is in
  maintenance mode, so it is better used as a reference than as a new project
  dependency.

Dataset protocol:

1. Preserve the official test split untouched.
2. Make one seeded, stratified validation split from the official training set.
3. Fit the tokenizer vocabulary on the resulting training subset only.
4. Persist the split indices, label mapping, vocabulary, maximum length, and
   dataset revision/cache metadata.
5. Reuse exactly the same encoded examples and padding masks for every model.

Because this is multiclass classification, report cross-entropy loss and
accuracy (plus optional macro-F1/confusion matrices). Perplexity is a
language-modelling metric and should be marked **not applicable**, not computed
from classification cross-entropy as though it measured next-token prediction.

## Vanilla RNN

Authoritative references:

- Jeffrey L. Elman, [*Finding Structure in
  Time*](https://doi.org/10.1207/s15516709cog1402_1), *Cognitive Science* 14(2),
  1990. This is the canonical Elman recurrent-network reference.
- [PyTorch NLP From Scratch: character-level
  RNN](https://docs.pytorch.org/tutorials/intermediate/nlp_from_scratch_index.html),
  an official, small manual recurrence and training example.
- [`torch.nn.RNN` equations](https://docs.pytorch.org/docs/stable/generated/torch.nn.RNN.html),
  used only as an equation/shape oracle, not as the implementation.

Video references:

- Josh Starmer (StatQuest), [*Recurrent Neural Networks (RNNs), Clearly
  Explained*](https://www.youtube.com/watch?v=AsNTP8Kwu80). Strong intuition for
  unrolling, shared parameters, backpropagation through time, and gradient
  failure modes.
- Patrick Loeber (Python Engineer), [*PyTorch RNN Tutorial — Name
  Classification*](https://www.python-engineer.com/posts/pytorch-rnn/). The page
  embeds the YouTube walkthrough and links its code; compare its cell equation
  with the official PyTorch tutorial before adapting ideas.

Implementation takeaway: explicitly loop over time and layers, computing
`h_t = tanh(W_x x_t + W_h h_(t-1) + b)`. Apply inter-layer dropout only between
stacked layers, not independently to the recurrent state at every time step.
Ensure padding cannot update a sequence after its true length, or select/pool
states with a padding mask.

## LSTM

Authoritative references:

- Sepp Hochreiter and Jürgen Schmidhuber, [*Long Short-Term
  Memory*](https://doi.org/10.1162/neco.1997.9.8.1735), *Neural Computation*
  9(8), 1997.
- [`torch.nn.LSTM` equations](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html),
  a convenient independent check for gate definitions and tensor shapes.

Video reference:

- Josh Starmer (StatQuest), [*Long Short-Term Memory (LSTM), Clearly
  Explained*](https://www.youtube.com/watch?v=YCzL96nL7j0). A high-quality visual
  walkthrough of the forget, input/candidate, and output paths.

Implementation takeaway: calculate the four affine gate projections in one
matrix multiplication, split into input, forget, candidate, and output chunks,
then apply `sigmoid`, `sigmoid`, `tanh`, and `sigmoid`. Update the cell before
the hidden state. Document the chosen gate order, initialize forget-gate bias
deliberately, and unit-test a one-step cell against a hand calculation (and,
optionally, a weight-mapped `nn.LSTMCell` oracle used only in tests).

## Transformer encoder

Authoritative references:

- Ashish Vaswani et al., [*Attention Is All You
  Need*](https://papers.neurips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html),
  NeurIPS 2017.
- Alexander Rush, [*The Annotated
  Transformer*](https://nlp.seas.harvard.edu/annotated-transformer/), an
  equation-aligned educational implementation maintained as a paper companion.

Video references:

- Umar Jamil, [*Attention Is All You Need (Transformer): model explanation,
  math, inference and training*](https://www.youtube.com/watch?v=bCz4OMemCcA).
  It connects the paper equations to a from-scratch implementation.
- Sebastian Raschka, [*Build an LLM From Scratch* video
  series](https://www.youtube.com/playlist?list=PLTKMiZHVd_2IIEsoJrWACkIxLRdfMlw11),
  a longer implementation-oriented companion for token embeddings, manual
  attention, masking, and training. Only the relevant attention chapters are
  needed here.

Implementation takeaway: implement scaled dot-product attention, head
split/merge, padding masks, sinusoidal positions, feed-forward layers, residual
paths, and pre-layer normalization directly. For topic classification, use an
encoder-style (non-causal) padding mask unless the experimental protocol
explicitly makes every architecture causal. Confirm that padded key positions
receive no attention probability and that the classifier's pooling is masked.

## Selective state-space model (Mamba-style)

Authoritative references:

- Albert Gu and Tri Dao, [*Mamba: Linear-Time Sequence Modeling with Selective
  State Spaces*](https://arxiv.org/abs/2312.00752), 2023, especially Section 3
  and Algorithm 2.
- The authors' [official `state-spaces/mamba`
  repository](https://github.com/state-spaces/mamba), particularly
  `mamba_ssm/modules/mamba_simple.py` and the selective-scan interface.
- The authors' [official S4 repository](https://github.com/state-spaces/s4) for
  the structured-SSM lineage and simple S4D references.

Video references:

- Umar Jamil, [*Mamba and S4 Explained: Architecture, Parallel Scan, Kernel
  Fusion, Recurrent, Convolution, Math*](https://www.youtube.com/watch?v=8Q_tqwpTpVU).
  This is the most useful deep conceptual crosswalk from continuous SSMs to S4
  and Mamba.
- Yannic Kilcher, [*Mamba: Linear-Time Sequence Modeling with Selective State
  Spaces (Paper Explained)*](https://www.youtube.com/watch?v=9dSkvxS2EB0).
- AI Coffee Break with Letitia, [*MAMBA and State Space Models
  Explained*](https://www.youtube.com/watch?v=vrF3MtGwD0Y), a shorter visual
  explanation of selection.
- Algorithmic Simplicity, [*MAMBA from Scratch*](https://www.youtube.com/watch?v=N6Piou4oYx8).
  Treat this as an implementation companion only; check every equation against
  the paper and official repository.

Implementation takeaway: describe the project model precisely as a
**simplified Mamba-style selective SSM**, not a reproduction of optimized
Mamba. A defensible educational block has input projection and gating, a small
causal depthwise convolution, input-dependent `delta`, `B`, and `C`, a stable
diagonal state matrix such as `A = -exp(A_log)`, a recurrent selective scan,
output gating/projection, normalization, and a residual path. Prefer the exact
elementwise zero-order-hold discretization where practical; protect the
near-zero denominator numerically.

A Python/PyTorch time loop demonstrates the recurrence correctly, but it does
not implement the paper's parallel scan, kernel fusion, recomputation strategy,
or custom CUDA kernel. Consequently, observed wall-clock speed may be worse than
attention at AG News sequence lengths. Do not claim the runtime characteristics
of the official fused Mamba implementation for this educational model. The
official repository also cautions that recurrent SSM dynamics can be
precision-sensitive; AMP should keep master parameters in FP32 and numerical
stability should be checked explicitly.

## Fair-comparison design decisions

- Fix one data pipeline, seed set, batching policy, optimizer family, stopping
  rule, and evaluation implementation. Log any model-specific learning-rate or
  regularization change rather than hiding it.
- Publish both a common-capacity configuration and actual parameter counts.
  Equal hidden width does not imply equal capacity across these architectures.
- Decide and document context directionality. A non-causal Transformer sees both
  left and right context; a forward RNN/LSTM/SSM does not. Either make all models
  causal, or implement matched bidirectional recurrent/SSM baselines for the
  primary classification comparison.
- Use one masked pooling/readout rule wherever possible. Never classify from a
  padded final position.
- Time a warm, synchronized evaluation loop on the same device, precision,
  batch sizes, and sequence-length bins. Separate training throughput
  (examples/s or tokens/s) from single-batch inference latency.
- Record peak accelerator memory only when measured with an appropriate device
  API after resetting statistics. Report `N/A` on CPU rather than inventing a
  proxy.
- Run multiple seeds for final claims and report mean plus dispersion. A single
  exploratory run may be shown but must be labelled as such.
- Keep baseline runs immutable. Iterative tuning should create new run IDs with
  parent run, hypothesis, exact config diff, and result.

## Suggested notebook roles

Notebooks should call tested project modules rather than contain a second copy
of model or training logic:

1. `01_data_audit.ipynb`: split sizes, class balance, token-length distribution,
   vocabulary coverage, truncation rate, and leakage checks.
2. `02_model_sanity.ipynb`: tensor shapes, masks, one-batch overfit, gradient
   norms, parameter counts, and selective-state traces.
3. `03_benchmark_analysis.ipynb`: load machine-readable run files, build tables
   and curves, compare seeds, and generate report figures.

The repository README should give a short quick start, exact commands for data,
training, evaluation, and plotting, expected artifact locations, tested
software/hardware, deterministic limitations, and a clear statement that saved
benchmark numbers are measured rather than expected targets.
