#!/usr/bin/env bash

# fix segmentation fault reported in https://github.com/k2-fsa/icefall/issues/674
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

set -eou pipefail

nj=30
stage=1
stop_stage=3

# We assume dl_dir (download dir) contains the following
# directories and files. If not, you need to apply aishell2 through
# their official website.
# https://www.aishelltech.com/aishell_2
#
#  - $dl_dir/aishell2

dl_dir=$PWD/download

text_extractor="pypinyin_initials_finals"
audio_extractor="Encodec"  # or Fbank
audio_feats_dir=data/tokenized


. shared/parse_options.sh || exit 1

mkdir -p data

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

log "dl_dir: $dl_dir"

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  log "Stage 1: Prepare genshin manifest"
  if [ ! -f data/manifests/.genshin_manifests.done ]; then
    mkdir -p data/manifests
    python genshin.py --input-dir $dl_dir/genshin --output-dir data/manifests -j $nj
    touch data/manifests/.genshin_manifests.done
  fi
fi

name=genshin

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  log "Stage 2: Tokenize/Fbank GenShin"
  mkdir -p ${audio_feats_dir}
  if [ ! -e ${audio_feats_dir}/.${name}.tokenize.done ]; then
    CUDA_VISIBLE_DEVICES="2,3" python3 bin/tokenizer.py --dataset-parts "train dev test" \
    --prefix ${name} \
    --audio-extractor ${audio_extractor} \
    --text-extractor ${text_extractor} \
    --batch-duration 400 \
    --src-dir "data/manifests" \
    --output-dir "${audio_feats_dir}"
  fi
  touch ${audio_feats_dir}/.${name}.tokenize.done
fi


if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  log "Stage 3: Prepare $name train/dev/test"
  if [ ! -e ${audio_feats_dir}/.${name}.train.done ]; then
    # dev
    lhotse copy \
      ${audio_feats_dir}/${name}_cuts_dev.jsonl.gz \
      ${audio_feats_dir}/cuts_dev.jsonl.gz

    # train
    lhotse copy \
      ${audio_feats_dir}/${name}_cuts_train.jsonl.gz \
      ${audio_feats_dir}/cuts_train.jsonl.gz

    # test
    lhotse copy \
      ${audio_feats_dir}/${name}_cuts_test.jsonl.gz \
      ${audio_feats_dir}/cuts_test.jsonl.gz

    touch ${audio_feats_dir}/.${name}.train.done
  fi
fi

# 最后完成过后再加一个这个
python3 ./bin/display_manifest_statistics.py --manifest-dir ${audio_feats_dir}