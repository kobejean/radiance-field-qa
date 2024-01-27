#!/usr/bin/env bash
SWEEP_ID=$(wandb sweep --project radiance-field-qa wandb/config.yaml)
echo "SWEEP_ID=$SWEEP_ID"
wandb agent $SWEEP_ID