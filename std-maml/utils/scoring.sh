#!/bin/bash

# infer paths
model_path=/work5/cslt/kangjiawen/070720-maml-cn2/log/07182020/sssc
data_dir=/work5/cslt/kangjiawen/070720-maml-cn2/data
train_data=$data_dir/train_ssmc_100k
enroll_data=$data_dir/eval/enroll
test_data=$data_dir/eval/test
out_file=/work5/cslt/kangjiawen/070720-maml-cn2/output/temp #output file

# scoring paths
srcdir=$out_file/scores
trials=$test_data/trials.lst
trl_name=`basename $trials`
dev=$out_file/dev #PLDA output set

stage=1
nj=8
cmd=/work9/cslt/kangjiawen/temp/kaldi-cnceleb/egs/wsj/s5/utils/run.pl


if [ $stage -le 2 ]; then
# Compute Cosine scores
    rm -rf $srcdir/cosine_scores
    mkdir -p $srcdir/cosine_scores
    rm -rf $srcdir/log
    mkdir -p $srcdir/log

    scores_dir=$srcdir/cosine_scores

  $cmd $scores_dir/log/cosine_scoring.log \
   cat $trials \| awk '{print $1" "$2}' \| \
   ivector-compute-dot-products - \
    "ark:ivector-mean ark:$enroll_data/spk2utt scp:$enroll_data/xvector.scp ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-normalize-length scp:$test_data/xvector.scp ark:- |" \
     $scores_dir/cosine_scores || exit 1;

    eer=$(paste $trials  $scores_dir/cosine_scores | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
    echo "$name Cosine EER: $eer%" 
fi

if [ $stage -le 3 ]; then
  rm -rf $dev 
  mkdir $dev
  # Compute the mean.vec used for centering.
  $cmd $dev/log/compute_mean.log \
    ivector-mean scp:$train_data/xvector.scp \
    $dev/mean.vec || exit 1;

  # Use LDA to decrease the dimensionality prior to PLDA.
  lda_dim=150
  $cmd $dev/log/lda.log \
    ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
    "ark:ivector-subtract-global-mean scp:$train_data/xvector.scp ark:- |" \
    ark:$train_data/utt2spk $dev/lda.mat || exit 1;

  # Train the PLDA model.
  $cmd $dev/log/plda.log \
    ivector-compute-plda ark:$train_data/spk2utt \
    "ark:ivector-subtract-global-mean scp:$train_data/xvector.scp ark:- | transform-vec $dev/lda.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    $dev/plda || exit 1;
fi

if [ $stage -le 4 ]; then
    rm -rf $scores
    scores=$srcdir/plda_scores
    $cmd JOB=1:$nj $scores/log/lda_plda_scoring.JOB.log \
      ivector-plda-scoring --normalize-length=true \
        --num-utts=ark:$enroll_data/num_utts.ark \
        "ivector-copy-plda --smoothing=0.0 $dev/plda - |" \
        "ark:ivector-mean ark:$enroll_data/spk2utt scp:$enroll_data/xvector.scp ark:- | ivector-subtract-global-mean $dev/mean.vec ark:- ark:- | transform-vec $dev/lda.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
        "ark:ivector-subtract-global-mean $dev/mean.vec scp:$test_data/xvector.scp ark:- | transform-vec $dev/lda.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
        "cat '$test_data/trials/$nj/trials.JOB' | cut -d\  --fields=1,2 |" $scores/lda_plda_scores.JOB || exit 1;

    for n in $(seq $nj); do
      cat $scores/lda_plda_scores.$n
    done > $scores/lda_plda_scores
    for n in $(seq $nj); do
      rm $scores/lda_plda_scores.$n
    done

    eer=$(paste $trials $scores/lda_plda_scores | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
    echo "LDA_PLDA EER: $eer%"
fi


