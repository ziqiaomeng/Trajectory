#!/bin/bash
for dataset in chameleon cornell film squirrel texas wisconsin
#for dataset in chameleon
do
  for model in GCN GAT SGC SAGE ChebNet MLP SimGCN
  do
    python main.py \
    --dataset ${dataset} \
    --num_hidden 48 \
    --model ${model} \
    --iter 1 \
    --dataset_split splits/${dataset}_split_0.6_0.2_0.npz
  done
done
