#!/bin/bash

python autoencode.py fit --model ltng.ae.VoiceAutoEncoder --config cfg/ae/vctk.yaml --model cfg/ae/decoder/golf.yaml --data.batch_size 128
