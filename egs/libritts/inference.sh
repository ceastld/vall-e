#!/usr/bin/env bash
# . shared/parse_options.sh || exit 1


exp_dir=exp/valle

# text_prompts="KNOT one point one five miles per hour."
# audio_prompts=./prompts/8463_294825_000043_000000.wav
# text="To get up and running quickly just follow the steps below."
text_prompts="This I read with great attention, while they sat silent."
audio_prompts=./prompts/8455_210777_000067_000000.wav
text="'I'm sure those are not the right words,' said poor Alice, and her eyes filled with tears again as she went on, 'I must be Mabel after all, and I shall have to go and live in that poky little house."



CUDA_VISIBLE_DEVICES=0 python bin/infer.py --output-dir infer/demos \
    --model-name valle --norm-first true --add-prenet false \
    --share-embedding true --norm-first true --add-prenet false \
    --text-prompts "${text_prompts}" \
    --audio-prompts ${audio_prompts} \
    --text "${text}" \
    --checkpoint=${exp_dir}/best-valid-loss.pt
