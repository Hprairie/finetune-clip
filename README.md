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

## Current Project Structure

Benchmarks: retrieval benchmarks
- Need to check but can just use clip-benchmark for all retrieval benchmarks

Misc-Testing : random visualizations

Scripts : Scripts to finetune CLIP

Open-Clip : src

## Running Finetuning Scripts
Very similar to open_clip, but with a seperate script for finetuning instead of training. i.e.

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

## Adding New Finetuning Methods
Go to `finetune.configure_finetune` in order to add a new method to finetune clip models. Currently the following
methods are supported.

- LoRA
- Linear Probing
- Layer Freezing

## Things that need to be done
- Add Huggingface finetuning support for LoRA
- Add a nice way to append linear-probe to a model
