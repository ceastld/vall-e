exp_dir=exp/valle

text=libritts.txt
gpus="3"
checkpoint=${exp_dir}/best-valid-loss.pt

. shared/parse_options.sh || exit 1

CUDA_VISIBLE_DEVICES="${gpus}" python bin/infer.py --output-dir infer/demos \
    --model-name valle --norm-first true --add-prenet false \
    --share-embedding true --norm-first true --add-prenet false \
    --text-prompts "" \
    --audio-prompts "" \
    --text "${text}" \
    --output-dir "infer/demos" \
    --checkpoint ${checkpoint}