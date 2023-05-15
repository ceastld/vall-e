exp_dir=exp/valle

CUDA_VISIBLE_DEVICES=0 python bin/infer.py --output-dir infer/demos \
    --model-name valle --norm-first true --add-prenet false \
    --share-embedding true --norm-first true --add-prenet false \
    --text-prompts "" \
    --audio-prompts "" \
    --text libritts.txt \
    --output-dir "infer/demos" \
    --checkpoint ${exp_dir}/best-valid-loss.pt