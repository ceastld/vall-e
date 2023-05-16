#!/usr/bin/env bash
# . shared/parse_options.sh || exit 1


exp_dir=exp/valle

# text_prompts="KNOT one point one five miles per hour."
# audio_prompts=./prompts/8463_294825_000043_000000.wav
# text="To get up and running quickly just follow the steps below."

text_prompts=
audio_prompts=
text=
output_dir="infer"

. shared/parse_options.sh || exit 1 # 用于接收命令行参数

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

# log $text_prompts, $audio_prompts, $text, $output_dir


CUDA_VISIBLE_DEVICES=0 python bin/infer.py \
    --model-name valle --norm-first true --add-prenet false \
    --share-embedding true --norm-first true --add-prenet false \
    --text-prompts "${text_prompts}" \
    --audio-prompts ${audio_prompts} \
    --output-dir "${output_dir}" \
    --text "${text}" \
    --text-extractor "pypinyin_initials_finals" \
    --checkpoint ${exp_dir}/best-valid-loss.pt
