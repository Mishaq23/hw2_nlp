# PyTorch Template for DL projects

<p align="center">
  <a href="#about">About</a> •
  <a href="#tutorials">Tutorials</a> •
  <a href="#examples">Examples</a> •
  <a href="#installation">Installation</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#useful-links">Useful Links</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>

<p align="center">
<a href="https://github.com/Blinorot/pytorch_project_template/generate">
  <img src="https://img.shields.io/badge/use%20this-template-green?logo=github">
</a>
<a href="https://github.com/Blinorot/pytorch_project_template/blob/main/LICENSE">
   <img src=https://img.shields.io/badge/license-MIT-blue.svg>
</a>
<a href="https://github.com/Blinorot/pytorch_project_template/blob/main/CITATION.cff">
   <img src="https://img.shields.io/badge/cite-this%20repo-purple">
</a>
</p>

## About

This repository contains a template for [PyTorch](https://pytorch.org/)-based Deep Learning projects.

The template utilizes different python-dev techniques to improve code readability. Configuration methods enhance reproducibility and experiments control.

The repository is released as a part of the [HSE DLA course](https://github.com/markovka17/dla), however, can easily be adopted for any DL-task.

This template is the official recommended template for the [EPFL CS-433 ML Course](https://www.epfl.ch/labs/mlo/machine-learning-cs-433/).

**New:** we added a [HF Main](https://github.com/Blinorot/pytorch_project_template/tree/hf_main) variant of the template with [HuggingFace](https://huggingface.co/) Integration for multi-GPU and multi-node training, automatic mixed precision, gradient accumulation, and seamless HuggingFace Ecosystem Compatibility.

> 📖 **If you use this template in your work, please cite this repository or include a reference. Attribution supports the project and encourages continued development.**

## Tutorials

This template utilizes experiment tracking techniques, such as [WandB](https://docs.wandb.ai/) and [Comet ML](https://www.comet.com/docs/v2/), and [Hydra](https://hydra.cc/docs/intro/) for the configuration. It also automatically reformats code and conducts several checks via [pre-commit](https://pre-commit.com/). If you are not familiar with these tools, we advise you to look at the tutorials below:

- [Python Dev Tips](https://github.com/ebezzam/python-dev-tips): information about [Git](https://git-scm.com/doc), [pre-commit](https://pre-commit.com/), [Hydra](https://hydra.cc/docs/intro/), and other stuff for better Python code development. The YouTube recording of the workshop is available [here](https://youtu.be/okxaTuBdDuY).

- [Seminar on R&D Coding 2025](https://youtu.be/PE1zaW5it_A): Seminar from the [LauzHack Deep Learning Bootcamp](https://github.com/LauzHack/deep-learning-bootcamp/) with discussion on logging, project-based coding, configuration, and reproducibility. The materials can be found [here](https://github.com/LauzHack/deep-learning-bootcamp/tree/summer25/day05).

- [Seminar on R&D Coding 2024](https://youtu.be/sEA-Js5ZHxU): Seminar from the [LauzHack Deep Learning Bootcamp](https://github.com/LauzHack/deep-learning-bootcamp/) with template discussion and reasoning. It also explains how to work with [WandB](https://docs.wandb.ai/). The seminar materials can be found [here](https://github.com/LauzHack/deep-learning-bootcamp/blob/main/day03/Seminar_WandB_and_Coding.ipynb).

- [HSE DLA Course Introduction Week](https://github.com/markovka17/dla/tree/2024/week01): combines the two seminars above into one with some updates, including an extra example for [Comet ML](https://www.comet.com/docs/v2/).

- [PyTorch Basics](https://github.com/markovka17/dla/tree/2024/week01/intro_to_pytorch): several notebooks with [PyTorch](https://pytorch.org/docs/stable/index.html) basics and corresponding seminar recordings from the [LauzHack Deep Learning Bootcamp](https://github.com/LauzHack/deep-learning-bootcamp/).

To start working with a template, just click on the `use this template` button.

<a href="https://github.com/Blinorot/pytorch_project_template/generate">
  <img src="https://img.shields.io/badge/use%20this-template-green?logo=github">
</a>

You can choose any of the branches as a starting point. [Set your choice as the default branch](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-branches-in-your-repository/changing-the-default-branch) in the repository settings. You can also [delete unnecessary branches](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-and-deleting-branches-within-your-repository).

## Examples

> [!IMPORTANT]
> The main branch leaves some of the code parts empty or fills them with dummy examples, showing just the base structure. The final users can add code required for their own tasks.

You can find examples of this template completed for different tasks in other branches:

- [HF Main](https://github.com/Blinorot/pytorch_project_template/tree/hf_main): the variant of the `main` branch with [HuggingFace](https://huggingface.co/) Integration. Supports multi-GPU and multi-node training, automatic mixed precision, gradient accumulation, and seamless HuggingFace Ecosystem Compatibility.

- [Image classification](https://github.com/Blinorot/pytorch_project_template/tree/example/image-classification): simple classification problem on [MNIST](https://yann.lecun.com/exdb/mnist/) and [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) datasets.

- [ASR](https://github.com/Blinorot/pytorch_project_template/tree/example/asr): template for the automatic speech recognition (ASR) task. Some of the parts (for example, `collate_fn` and beam search for `text_encoder`) are missing for studying purposes of [HSE DLA course](https://github.com/markovka17/dla).

## Installation

Installation may depend on your task. The general steps are the following:

0. (Optional) Create and activate new environment using [`conda`](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) or `venv` ([`+pyenv`](https://github.com/pyenv/pyenv)).

   a. `conda` version:

   ```bash
   # create env
   conda create -n project_env python=PYTHON_VERSION

   # activate env
   conda activate project_env
   ```

   b. `venv` (`+pyenv`) version:

   ```bash
   # create env
   ~/.pyenv/versions/PYTHON_VERSION/bin/python3 -m venv project_env

   # alternatively, using default python version
   python3 -m venv project_env

   # activate env
   source project_env/bin/activate
   ```

1. Install all required packages

   ```bash
   pip install -r requirements.txt
   ```

   On Kaggle/Colab, use the preinstalled CUDA-enabled PyTorch when it satisfies
   `torch>=2.2,<3`. Install requirements without `--upgrade`; the dependency
   ranges also allow the NumPy and PyArrow versions supplied by the environment.
   PyTorch 2.2.0 and NumPy 1.26.4 have no Python 3.13 wheels. Restart the notebook
   kernel if pip replaced packages that were already imported.

2. Install `pre-commit`:
   ```bash
   pre-commit install
   ```

## How To Use

### Code generation data

Start training from the command line; a notebook is optional for analysis:

```bash
python train.py -cn code_generation
```

Before training, this samples Parquet files from Hugging Face, creates train/validation
groups, trains BPE on train, and saves tokenized Arrow splits. Matching prepared
data is reused on later runs. Parameters are in
`src/configs/data_preparation/tinysms.yaml`. The default tokenizer uses up to
50,000 training examples; `data_preparation.tokenizer_train_rows=null` uses the
entire training partition. Preparation runs inside `train.py` through
`src/datasets/preparation.py`.

By default, preparation selects **1,200,000 source rows across all training
shards**, with quotas proportional to each file's row count. It samples row
indices without replacement using a fixed seed, then reads each remote file
in batches and writes only selected rows to a temporary Parquet file. The full
source corpus is never saved to the local HF cache. Source files still have to
be read; sampling saves disk space and tokenization work, not necessarily
download traffic. The temporary sample and preparation caches are removed after
the final Arrow splits and tokenizer are saved.

Configure the sample independently of model hyperparameters:

```bash
python train.py -cn code_generation data_preparation.sampling.max_rows=1200000 data_preparation.sampling.seed=42 data_preparation.sampling.batch_size=4096
```

`sampling.batch_size` controls source reading, not the training batch size.
The row limit applies **before** empty-text filtering, test-question exclusion,
and train/validation splitting; the final train count is slightly smaller.
`metadata.json` records source and sampled row counts for every shard, plus
the final split sizes. The existing loader shuffles the combined local sample
for training; it continues the same traversal between 500-step logging windows.
Read buffers are bounded by the current Parquet file/batches, not the corpus.
Set `data_preparation.sampling.max_rows=null` to use the previous full-loading
path, including its normal HF cache. An explicit `dataset.data_files` override
limits sampling to those files. Local Parquet files are supported with
`data_preparation.dataset.path=parquet` and a train/test `data_files` mapping.

On Kaggle, choose writable locations for the prepared data and HF cache:

```bash
export TINYSMS_DATA_DIR=/kaggle/working/tinysms
python train.py -cn code_generation data_preparation.dataset.cache_dir=/kaggle/working/hf_cache
```

The `tinysms_train` and `tinysms_test` dataset configs read these saved splits
through `CodeGenerationDataset` and use the same `TINYSMS_DATA_DIR`. Training
and inference select their respective dataset config automatically.
The collate function reads the tokenizer's PAD ID, pads inputs on the right,
creates boolean attention masks (`true` means a real token), and pads labels
with `-100`. Test batches contain encoder inputs and original sample IDs.
For training with the encoder-decoder Transformer, use the dedicated config:

```bash
python train.py -cn code_generation
```

This config trains on train and evaluates loss on val; official test has no
targets and is reserved for generation. AdamW and the cosine scheduler are
configured in `src/configs/optimizer/adamw.yaml` and
`src/configs/lr_scheduler/cosine.yaml`. The scheduler updates after each optimizer
step; by default its period is `n_epochs * steps_per_epoch`, with
`steps_per_epoch` taken from `trainer.epoch_len` or the train loader length.
Model hyperparameters are in
`src/configs/model/transformer.yaml`: dimensions, layer counts, heads, dropout,
activation, RoPE theta, LayerNorm epsilon, biases, embedding sharing, scaling,
and initialization. Vocabulary size and special-token IDs come from the saved
tokenizer. Loss parameters are in `src/configs/loss_function/code_generation.yaml`.

Change hyperparameters directly from the command line, for example:

```bash
python train.py -cn code_generation model.d_model=384 model.n_heads=6 model.d_ff=1536
python train.py -cn code_generation model.n_decoder_layers=6 loss_function.label_smoothing=0.1
```

The model uses Pre-LayerNorm, rotary self-attention in both stacks, and
cross-attention without RoPE. Its `encode` and `decode_step` methods support
generation with a reused encoder output and per-layer self/cross-attention K/V
caches. `decode_step` expects eval mode and unpadded new tokens. The loss is
mean token cross-entropy with `ignore_index=-100`; targets are already shifted
by the dataset.

Metrics are configured in `src/configs/metrics/code_generation.yaml`. Loss,
perplexity, and token accuracy ignore label padding and aggregate by the number
of target tokens, including EOS. Perplexity is exp(mean NLL) over all evaluated
tokens and uses unsmoothed cross-entropy even when training uses label smoothing.

Every 5 epochs (and at the final epoch), validation generates one
program per question on a **fixed random subset of up to 512 validation examples**.
`generation_validation.seed` fixes this subset independently of model RNG;
`max_new_tokens=256` matches inference. The default checkpoint monitor is
**`max val_pass_at_1`**. Missing metrics between generation checks are skipped
without disabling monitoring. Equal scores keep the earlier best checkpoint;
early stopping counts measured checks rather than skipped epochs.

The `exp07_swiglu` experiment uses SwiGLU (`d_model=512`, `d_ff=1792`, 6+6 layers,
8 heads, 56,217,600 parameters with vocabulary 8192). Its packed input projection
produces gate and value vectors; `SiLU(gate) * value` is projected back to
`d_model`. Dropout is unchanged. Existing GELU checkpoints retain their original
parameter names and shapes.

This experiment compares `generation_validation.num_beams=[1,3]` with a shared
length penalty of 1.0 and validation batch size 16. Both modes use the same fixed
questions. `val_pass_at_1` is the larger score, and `model_best.pth` stores the
winning checkpoint and decoding mode. `model_best_greedy.pth` and
`model_best_beam3.pth` additionally keep each mode's best inference weights.
The base `code_generation` config still evaluates only greedy by default.
Start this experiment with `python train.py -cn exp07_swiglu` after configuring
the prepared data path and Comet API key as described below.

Local pass@1 compares `solve()` results against the provided validation code
using `abs(prediction-reference) <= atol + rtol*abs(reference)` with both
tolerances set to `1e-6`. Reference results are cached. Invalid/non-numeric or
unsupported references are excluded from the denominator; generated failures
on valid references score zero. This measures agreement with the provided
programs, whose answers are not guaranteed correct, and is **not the official
bot score**. It creates no additional training targets.

Programs run in separate Python processes with wall-time/CPU limits and a
Linux memory limit, restricted builtins, and public exports from `math`,
`fractions`, `decimal`, and `statistics`. Other imports are unsupported.
These are best-effort execution restrictions, not a hardened OS sandbox.
Execution options are in `src/configs/metrics/code_generation.yaml`.

Comet logs `pass_at_1_val`, `reference_validity_val`, `execution_validity_val`,
and `pass_at_1_examples_val`, in addition to syntax validity and token metrics.
When comparing modes, it also logs `pass_at_1_greedy_val`, `pass_at_1_beam3_val`,
their separate generation/execution times, and `selected_num_beams_val`.
`generation_seconds` excludes CPU code checking; `execution_seconds` reports
that checking separately. The 180-second T4 limit still needs a separate
full-test inference measurement. Configure or disable this check with:

```bash
python train.py -cn code_generation generation_validation.every_n_epochs=10
python train.py -cn code_generation generation_validation.enabled=false 'trainer.monitor=min val_loss'
```

When reusing older launch commands, replace an explicit
`generation_validation.max_examples=32` with `512`; CLI overrides take
precedence over the new default. Syntax validity and token accuracy are
diagnostics and do not replace execution-based correctness.

To train a model, run the following command:

```bash
python train.py HYDRA_CONFIG_ARGUMENTS
```

The default training config is `code_generation`; Hydra arguments override its settings.

To log training curves to **Comet Experiment Management**, set `COMET_API_KEY`
and select the Comet writer. Use your Experiment Management workspace and project;
an Opik workspace is configured separately.

```bash
python train.py writer=cometml writer.workspace=YOUR_WORKSPACE \
  writer.project_name=hw2-generation writer.run_name=exp03_d512_ff2048_20k \
  model.d_model=512 model.d_ff=2048 model.n_heads=8 \
  trainer.n_epochs=40 trainer.epoch_len=500 lr_scheduler.T_max=20000
```

On Kaggle, add a `COMET_API_KEY` secret and read it before launching the script:

```python
import os
from kaggle_secrets import UserSecretsClient

os.environ["COMET_API_KEY"] = UserSecretsClient().get_secret("COMET_API_KEY")
```

The writer records the resolved configuration, train/val metrics, learning rate,
training speed, and validation generation metrics. Generation checks also log a
table of up to eight questions, reference programs, and generated programs.
Open the experiment URL printed at startup, then use **Panels** with the **Step**
axis to compare runs. Epoch summaries aggregate every training batch; curves
show logging windows. Public pass@1 is supplied separately by the competition bot.

Comet uses a 32-character run ID saved in the run configuration. With the same
run name and `trainer.resume_from`, metrics continue in the same experiment.
The script finishes pending Comet uploads when training exits. To check the
pipeline without cloud logging, use `writer.mode=offline` or `disabled`;
offline archives are saved under `saved/comet_offline`.

To run inference (evaluate the model or save predictions):

```bash
python inference.py HYDRA_CONFIG_ARGUMENTS
```

The default inference config is `code_generation_inference`. To select the trained run:

```bash
python inference.py inferencer.checkpoint_dir=saved/rope_preln
```

Each training run saves `tokenizer.json` beside its `config.yaml` and checkpoints.
Inference reads the trained architecture from that config and uses this
tokenizer. It uses the decoding mode stored in the checkpoint (greedy for old
checkpoints), with K/V caches, and saves
`data/saved/code_generation/test/predictions.jsonl`, with one `{sample_id, code}`
record per test example, sorted by the original sample ID. The last partial
batch is retained. Syntax validity and full-partition generation time are logged;
`metrics.json` records generation time and the beam count. Set
`inferencer.generation.num_beams=1` or `3` to override automatic selection;
`inferencer.generation.length_penalty` can also be overridden. The bot submission
uses only `{"code": "..."}` from each prediction, in the same order.

## Useful Links:

You may find the following links useful:

- [Report branch](https://github.com/Blinorot/pytorch_project_template/tree/report): Guidelines for writing a scientific report/paper (with an emphasis on DL projects).

- [CLAIRE Template](https://github.com/CLAIRE-Labo/python-ml-research-template): additional template by [EPFL CLAIRE Laboratory](https://www.epfl.ch/labs/claire/) that can be combined with ours to enhance experiments reproducibility via [Docker](https://www.docker.com/).

- [Mamba](https://github.com/mamba-org/mamba) and [Poetry](https://python-poetry.org/): alternatives to [Conda](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) and [pip](https://pip.pypa.io/en/stable/installation/) package managers given above.

- [Awesome README](https://github.com/matiassingers/awesome-readme): a list of awesome README files for inspiration. Check the basics [here](https://github.com/PurpleBooth/a-good-readme-template).

## Credits

This repository is based on a heavily modified fork of [pytorch-template](https://github.com/victoresque/pytorch-template) and [asr_project_template](https://github.com/WrathOfGrapes/asr_project_template) repositories.

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
