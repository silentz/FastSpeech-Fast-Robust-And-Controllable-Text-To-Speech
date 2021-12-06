#!/bin/bash

python trainer.py fit \
    --config config/overfit.yaml

latest_file=$(ls -t1 ./checkpoints/model | head -n 1)

python trainer.py fit \
    --config config/common.yaml \
    --ckpt_path "./checkpoints/model/$latest_file"
