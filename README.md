# Finetune Clip

## Installing
Run the following
```bash
cd open_clip/
make install
```
Then install pytorch and torchvision and run.
```bash
make install-training
```
To install the benchmarking stuff run the following in the project root.
```bash
make install
```

## Current Project Structure

Benchmarks: Separate scripts and files for testing all open_clip models

Misc-Testing: random visualizations

Scripts: Scripts to finetune CLIP

Benchmark-scripts: Scripts to run benchmarks on different CLIP models

Benchmark-results: Restults from benchmarking open_clip models

Open-Clip: src

## Running Finetuning Scripts
Very similar to open_clip, but with a separate script for finetuning instead of training. i.e.

```bash
export MASTER_PORT=12802

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr

torchrun --nproc_per_node 1 -m finetune.main \
    --dataset-type "csv" \
    --train-data "/path/to/dataset.csv" \
    --warmup 1000 \
    --batch-size 128 \
    --lr 1e-5 \
    --wd 0.1 \
    --epochs 1 \
    --workers 2 \
    --model "ViT-B-32" \
    --lora "10:1" \
    --report-to "wandb" \
    --log-every-n-steps 100
```

Additional Scripts can be found in `scripts/`, I will try to make slurm and sh pairs for every script so that it can easily be run on tacc.

## Running Benchmarking Scripts

```bash
python benchmark/main.py \
    --dataset "/path/to/dataset" \
    --dataset-ann "/path/to/annotation.csv" \
    --pretrained "path/to/checkpoint" \ # This can also be the name of an online source
    --finetune-path "logs/run-name" \ # Path to Log directory of a finetune run
    --name "Name_of_Benchmark_Run" \
    --k 1 5 \
    --batchsize 256
```

Running Benchmarking on local logs will infer how to construct the model. Otherwise, just specify model specifics when calling the benchmarking script.

## Adding New Finetuning Methods
Go to `finetune.configure_finetune` in order to add a new method to finetune clip models. Currently the following
methods are supported.

- LoRA (Done and tested)
- Linear Probing (Not Implemented yet)
- Layer Freezing (Done but tested)

## TODO List
- [x] Add LoRA finetuning
- [x] Test LoRA finetuning
- [x] Add Layerwise finetuning
- [x] Test Layerwise finetuning
- [ ] Add Huggingface support for LoRA
- [ ] Add Linear Probing Finetuning
- [ ] Test Linear Probing Fintuning
- [x] Add Benchmarking for Colbert
      - Mostly done just need to also add fine-tuned model support
- [x] Add new Loss function for fine-grain training
    - [x] Sparc
    - [x] Colbert
- [x] Test new Loss function for fine-grain training
    - [x] Sparc
    - [x] Colbert
- [x] Add more template scripts
