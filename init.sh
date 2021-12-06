#!/bin/bash

mkdir -p checkpoints/
mkdir -p checkpoints/model/
mkdir -p checkpoints/vocoder/

git submodule update --init --recursive

wget http://silentz.ml/waveglow_256channels_universal_v5.pt \
    -O ./checkpoints/vocoder/waveglow_256channels_universal_v5.pt

python3 make_durations.py
