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
    # pair data
    if [ "$run_all" = "true"  -o $stage -eq 0 ];then
    python -u main.py \
                         --outset $outset \
                         --date $date \
                         --exp mc-mct-genre 2>&1 \
                         --root ./data/train/mct_data/genre \
                         --save_interval $save \
                         --print_interval $print \
                         --infer_interval $infer \
                         --train_iterations $itrs \
                         --inner_lr $in_lr \
                         --outer_lr $out_lr \
                         --meta_batch_size $bs \
                         --num_classes 5074 \
                         2>&1 | tee -a mct.log || exit 1
 
  fi

  # partial
  if [ "$run_all" = "true"  -o $stage -eq 1 ];then
    python -u main.py \
                         --outset $outset \
                         --date $date \
                         --exp mc-mct-genre 2>&1 \
                         --root ./data/train_mc/mct_data/genre \
                         --save_interval $save \
                         --print_interval $print \
                         --infer_interval $infer \
                         --train_iterations $itrs \
                         --inner_lr $in_lr \
                         --outer_lr $out_lr \
                         --meta_batch_size $bs \
                         --num_classes 5074 \
                         2>&1 | tee -a mct.log || exit 1
 
  fi


done  
