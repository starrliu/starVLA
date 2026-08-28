# CogACT Dataset Baselines

This example trains StarVLA baselines directly on CogACT/AGIBot episodic data.
It does not convert the source videos to LeRobot and does not depend on a
CogACT source checkout.

## Data contract

The native adapter follows CogACT's robot-space `6d/abs/abs` baseline:

```text
left translation (3) + left rotation 6D (6) + left gripper (1)
+ right translation (3) + right rotation 6D (6) + right gripper (1) = 20D
```

For a sample at frame `t`, pose targets come from states `t+1...t+H` and
gripper targets come from command rows `t...t+H-1`. The configured horizon is
30, matching the latest CogACT multi-dataset training recipes.
As in CogACT, a normally completed episode keeps its first three repeated tail
targets valid (`finished_padding_tolerance: 3`). The current state is returned
as one normalized 20D vector.

For a multi-dataset mixture, each source first loads its own CogACT `mean/std`.
The loader then computes frame-and-weight-weighted global mean/std via the law
of total variance, installs that merged normalizer into every in-memory source,
and saves the same statistics under the single key `merged`. This follows
CogACT's `feature/enable-norm-merge-v2` training path (the branch referenced by
its latest multi-dataset experiment scripts). Set
`merge_statistics: false` only for a deliberate per-dataset-normalization
ablation.

By default action directories are selected in CogACT order:

```text
actions_new_gripper > actions_cmd > actions_gaussian > actions
```

Therefore the current sample data uses `actions_cmd`.

## Configure datasets

Datasets are configured entirely in YAML under
`datasets.vla_data.dataset_list`. To add another source, append an entry:

```yaml
- name: new_collection
  folder: cogact_new_collection_split_500_240h
  metadata_file: episodic_dataset_en.npy
  weight: 1.0
```

`folder` is relative to `data_root_dir`. Sampling probability is proportional
to `weight * number_of_valid_frames`, matching CogACT's weighted sampler.
Only `good` and `medium` episodes are used by default; `fail` and `frame_drop`
are excluded.

A dataset entry may override any common setting, including `action_source`,
`quality`, `view_names`, `image_size`, `normalization`, `action_horizon`, and
`statistics_key`.

The baseline uses the canonical instruction metadata files verbatim; canonical
language augmentation is disabled. Image color/crop augmentation is also
disabled. View dropout follows CogACT's `balanced_5way` policy: each sample
uniformly keeps one of all three views, head only, both hand views, head+right,
or head+left. The Qwen processor accepts the resulting variable image count per
sample.

## Dataloader smoke test

The loader prefers the existing JPEG-frame LMDB. Install the small optional
reader dependency first:

```bash
pip install lmdb
```

Then inspect real samples without loading a model:

```bash
python starVLA/dataloader/cogact_datasets.py \
  --config_yaml examples/realRobots/CogACT/train_files/starvla_qwenoft_cogact.yaml
```

The module-level CLI is intentionally not part of training; unit tests cover
action alignment and quality filtering. A full one-batch smoke command will be
added once the target GPU environment and pretrained VLM path are confirmed.

## Pretrained backbones

The verified local configs use:

```text
/data/checkpoints/starvla_cogact/Qwen3-VL-4B-Instruct-Action
/data/checkpoints/starvla_cogact/Qwen3-VL-4B-Instruct
```

## Train QwenOFT

QwenOFT requires a Qwen checkpoint containing the StarVLA action token:

```bash
FRAMEWORK=qwenoft NUM_PROCESSES=8 \
  bash examples/realRobots/CogACT/train_files/run_cogact_train.sh
```

For a short run:

```bash
WANDB_MODE=disabled FRAMEWORK=qwenoft NUM_PROCESSES=1 USE_DEEPSPEED=0 \
MAX_STEPS=1 BATCH_SIZE=1 NUM_WORKERS=0 FREEZE_MODULES=qwen_vl_interface \
RUN_ID=cogact_qwenoft_smoke \
  bash examples/realRobots/CogACT/train_files/run_cogact_train.sh
```

## Train QwenPI_v3

```bash
FRAMEWORK=qwenpi_v3 NUM_PROCESSES=8 \
  bash examples/realRobots/CogACT/train_files/run_cogact_train.sh
```

A memory-bounded functional smoke test can retain all 36 layer-wise DiT
connections while temporarily reducing the DiT width:

```bash
WANDB_MODE=disabled FRAMEWORK=qwenpi_v3 NUM_PROCESSES=1 USE_DEEPSPEED=0 \
MAX_STEPS=1 BATCH_SIZE=1 NUM_WORKERS=0 FREEZE_MODULES=qwen_vl_interface \
DIT_HIDDEN_DIM=128 REPEATED_DIFFUSION_STEPS=1 RUN_ID=cogact_qwenpi_v3_smoke \
  bash examples/realRobots/CogACT/train_files/run_cogact_train.sh
```

Do not use the reduced `DIT_HIDDEN_DIM` override for the actual baseline run;
the checked-in training config uses 1024.

The initial baseline uses synchronous chunk inference. RTC evaluation and a
robot-side 20D action decoder should be implemented separately so synchronous
model quality can be measured before introducing chunk-splicing behavior.

## Important limitations

- The QwenOFT and QwenPI_v3 baseline paths consume the loader's timestep
  `action_mask`, so repeated episode-tail padding is excluded from loss. Other
  StarVLA frameworks may not yet consume this optional field.
- Multi-dataset runs save a single `merged` normalization block. Deployment
  must use that block to unnormalize predictions.
- This baseline uses compact 20D vectors rather than CogACT's sparse 470D
  `ActionFeature`; all active values and their order are preserved.
