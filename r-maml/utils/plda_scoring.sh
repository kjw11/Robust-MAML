#!/bin/bash
# Copyright 2015   David Snyder
# Apache 2.0.
#
# This script trains PLDA models and does scoring.
nj=8
lda_dim=150
covar_factor=0.0
normalize_length=true
simple_length_norm=false # If true, replace the default length normalization
                         # performed in PLDA  by an alternative that
                         # normalizes the length of the iVectors to be equal
                         # to the square root of the iVector dimension.

#echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 6 ]; then
  echo "Usage: $0 <vec-type> <dev-dir> <enroll-dir> <test-dir> <trials-file> <scores-dir>"
fi

vec_type=$1
dev_dir=$2
enroll_data=$3
eval_dir=$4
trials=$5
scores_dir=$6

trl_name=`basename $trials`
trl_dir=`dirname $trials`

mkdir -p $scores_dir/log
run.pl JOB=1:$nj $scores_dir/log/lda_plda_scoring.JOB.log \
  ivector-plda-scoring --normalize-length=$normalize_length \
    --simple-length-normalization=$simple_length_norm \
    --num-utts=ark:$enroll_data/num_utts.ark \
    "ivector-copy-plda --smoothing=0.0 $dev_dir/plda - |" \
    "ark:ivector-mean ark:$enroll_data/spk2utt ark:$eval_dir/$vec_type ark:- | ivector-subtract-global-mean $dev_dir/mean.vec ark:- ark:- | transform-vec $dev_dir/lda.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean $dev_dir/mean.vec ark:$eval_dir/$vec_type ark:- | transform-vec $dev_dir/lda.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat '${trl_dir}/${nj}/${trl_name}.JOB' | cut -d\  --fields=1,2 |" $scores_dir/lda_plda_scores.JOB || exit 1;

for n in $(seq $nj); do
  cat $scores_dir/lda_plda_scores.$n
done > $scores_dir/lda_plda_scores

for n in $(seq $nj); do
  rm $scores_dir/lda_plda_scores.$n
done

eer=$(paste $trials $scores_dir/lda_plda_scores | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
echo "LDA_PLDA EER: $eer%"
