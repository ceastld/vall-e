exp_dir=exp/valle

gpus="0"

. shared/parse_options.sh || exit 1

CUDA_VISIBLE_DEVICES="${gpus}" python bin/infer.py --output-dir infer/demos \
    --model-name valle --norm-first true --add-prenet false \
    --share-embedding true --norm-first true --add-prenet false \
    --text-prompts "" \
    --audio-prompts "" \
    --text aishell2.txt \
    --text-extractor "pypinyin_initials_finals" \
    --checkpoint ${exp_dir}/best-valid-loss.pt