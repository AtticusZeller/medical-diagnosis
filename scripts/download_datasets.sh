#!/usr/bin/env bash
mkdir -p ./datasets/

curl -L -o ./datasets/chest-xray-pneumonia.zip\
  https://www.kaggle.com/api/v1/datasets/download/paultimothymooney/chest-xray-pneumonia

unzip ./datasets/chest-xray-pneumonia.zip -d ./datasets/
rm ./datasets/chest-xray-pneumonia.zip
