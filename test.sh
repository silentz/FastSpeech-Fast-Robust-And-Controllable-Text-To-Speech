#!/bin/bash

latest_file=$(ls -t1 ./checkpoints/model | head -n 1)

python trainer.py validate \
    --config config/common.yaml \
    --ckpt_path "./checkpoints/model/$latest_file"

