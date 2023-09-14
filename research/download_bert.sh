#!/bin/bash
mkdir model

python ../tools/convert_huggingface_bert_pytorch_to_npz.py bert-base-uncased model/bert-base-uncased.npz