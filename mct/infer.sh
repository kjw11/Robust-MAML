#!/bin/bash
# "match" means the enroll and test follow the same genre

  pre_fix=$1
  outset=$2
  model_path=/work103/kangjiawen/091620-maml-cn2/log/$pre_fix
  data_dir=/work103/kangjiawen/091620-maml-cn2/data
  eval_data=$data_dir/sub_genre_0804/xvector_eval.txt
  train_data=$data_dir/train_ssmc_100k/outset/${outset}/xvector.txt
  out_file=/work103/kangjiawen/091620-maml-cn2/output/$pre_fix #output file

  # scoring paths
  genre_dir=$data_dir/sub_genre_0804/eval
  dev_dir=$out_file/dev

  stage=0
  nj=8
  cmd=/work9/cslt/kangjiawen/temp/kaldi-cnceleb/egs/wsj/s5/utils/run.pl

  if [ $stage -le 1 ]; then
    # clear old file
    rm -rf $out_file

    # infer
    ./infer.py --model_path $model_path --ark_file $eval_data --out_file $out_file/eval || exit 1;
    ./infer.py --model_path $model_path --ark_file $train_data --out_file $out_file/train || exit 1;
  fi

  if [ $stage -le 2 ]; then
  # Compute Cosine scores
    for genre in $(ls $genre_dir); do
      echo -e "\nFor $genre"
      vec_type=output.ark
      enroll_dir=$genre_dir/$genre/enroll
      vec_dir=$out_file/eval
      trials_dir=$genre_dir/$genre/test/sub_trials
      scores_dir=$genre_dir/$genre/cosine_scores
      rm -rf $scores_dir

      trials=$trials_dir/${genre}_trials
      socres=$scores_dir/${genre}_scores
      ./utils/cosine_scoring.sh $vec_type \
                                  $enroll_dir \
                                  $vec_dir \
                                  $trials \
                                  $socres || exit 1;

    done
  fi

  if [ $stage -le 3 ]; then
    # make lda_plda
    rm -rf $dev_dir
    mkdir $dev_dir
    mkdir -p $dev_dir/log
    cp $data_dir/train_ssmc_100k/outset/${outset}/spk2utt $dev_dir/
    cp $data_dir/train_ssmc_100k/outset/${outset}/utt2spk $dev_dir/

    # Compute the mean.vec used for centering.
    $cmd $dev_dir/log/compute_mean.log \
      ivector-mean ark:$out_file/train/output.ark \
      $dev_dir/mean.vec || exit 1;

    # Use LDA to decrease the dimensionality prior to PLDA.
    lda_dim=150
    $cmd $dev_dir/log/lda.log \
      ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
      "ark:ivector-subtract-global-mean ark:$out_file/train/output.ark ark:- |" \
      ark:$dev_dir/utt2spk $dev_dir/lda.mat || exit 1;

    # Train the PLDA model.
    $cmd $dev_dir/log/plda.log \
      ivector-compute-plda ark:$dev_dir/spk2utt \
      "ark:ivector-subtract-global-mean ark:$out_file/train/output.ark ark:- | transform-vec $dev_dir/lda.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |"     \
      $dev_dir/plda || exit 1;
  fi


if [ $stage -le 4 ]; then
    # make lda_plda
    for genre in $(ls $genre_dir); do
      echo -e "\nFor $genre"
      vec_type=output.ark
      dev_dir=$dev_dir
      enroll_data=$genre_dir/$genre/enroll
      eval_dir=$out_file/eval
      trials_dir=$genre_dir/$genre/test/sub_trials
      scores_dir=$genre_dir/$genre/lda_plda_scores
      rm -rf $scores_dir

      trials=$trials_dir/${genre}_trials
      socres=$scores_dir/${genre}_scores
      ./utils/plda_scoring.sh $vec_type \
                              $dev_dir \
                              $enroll_data \
                              $eval_dir \
                              $trials \
                              $socres || exit 1;

    done
  fi
