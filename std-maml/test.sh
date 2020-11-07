#!/bin/bash

date=092020  # for log dir name
save=250  # save model interval
print=250  # print loss interval
infer=250  # make test interval
itrs=6001  # training iterations
in_lr=0.001  # inner learning rate in local  update
out_lr=0.001  # outer learning rate in global update
bs=128  # batch size

run_all=true
stage=1

genres="singing movie"

for outset in $genres; do
   
  if [ "$run_all" = "true"  -o $stage -eq 1 ];then
    python -u main_partial.py \
                         --outset $outset \
                         --date $date \
                         --exp mc-maml-genre 2>&1 \
                         --root ./data/train_mc/txts \
                         --save_interval $save \
                         --print_interval $print \
                         --infer_interval $infer \
                         --train_iterations $itrs \
                         --inner_lr $in_lr \
                         --outer_lr $out_lr \
                         --meta_batch_size $bs \
                         --num_classes 5074 \
                         2>&1 | tee -a partial_rmaml.log || exit 1
  fi

 if [ "$run_all" = "true"  -o $stage -eq 2 ];then
    python -u main.py \
                         --outset $outset \
                         --date $date \
                         --exp pair-maml-genre 2>&1 \
                         --root ./data/train/pairs_relabel \
                         --save_interval $save \
                         --print_interval $print \
                         --infer_interval $infer \
                         --train_iterations $itrs \
                         --inner_lr $in_lr \
                         --outer_lr $out_lr \
                         --meta_batch_size $bs \
                         --num_classes 1126 \
                         2>&1 | tee -a rmaml.log || exit 1
  fi

done  
