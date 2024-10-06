#!/bin/bash

sudo apt install gh

pip install -r requirements.txt
pushd .
cd ~
mkdir datasets
cd datasets
gdown 1wO-0mtk25vAfsKEiyb3M6J9Xl48KWC_z
popd

git config --global url."https://github.com/".insteadOf git@github.com:
