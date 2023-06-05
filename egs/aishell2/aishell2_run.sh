exp_dir=exp/valle2

gpus="0"
text=aishell2.txt

. shared/parse_options.sh || exit 1

CUDA_VISIBLE_DEVICES="${gpus}" python bin/infer.py --output-dir infer/demos \
    --model-name valle --norm-first true --add-prenet false \
    --share-embedding true --norm-first true --add-prenet false \
    --text-prompts "" \
    --audio-prompts "" \
    --text ${text} \
    --text-extractor "pypinyin_initials_finals" \
    --checkpoint ${exp_dir}/best-valid-loss.pt