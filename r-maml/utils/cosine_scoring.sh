#!/bin/bash
# This script does cosine scoring.

# Begin configuration section.
nj=8
cmd=run.pl
use_global_mean=false
# End configuration section.

#echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 5 ]; then
  echo "Usage: $0 <vec-type> <enroll-dir> <vec-dir> <trials-file> <scores-dir>"
fi

vec_type=$1
enroll_dir=$2
vec_dir=$3
trials=$4
scores_dir=$5

trl_name=`basename $trials`
trl_dir=`dirname $trials`

mkdir -p $scores_dir/log

  $cmd JOB=1:$nj $scores_dir/log/cosine_scoring.JOB.log \
   cat $trl_dir/$nj/$trl_name.JOB \| awk '{print $1" "$2}' \| \
   ivector-compute-dot-products - \
    "ark:ivector-mean ark:$enroll_dir/spk2utt ark:$vec_dir/$vec_type ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-normalize-length ark:$vec_dir/$vec_type ark:- |" \
     $scores_dir/cosine_scores.JOB || exit 1;

for n in $(seq $nj); do
  cat $scores_dir/cosine_scores.$n
done > $scores_dir/cosine_scores

for n in $(seq $nj); do
  rm $scores_dir/cosine_scores.$n
done

eer=$(paste $trials $scores_dir/cosine_scores | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
echo "Cosine EER: $eer%"
