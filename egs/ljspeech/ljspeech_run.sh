exp_dir=exp/valle1

CUDA_VISIBLE_DEVICES=3 python bin/infer.py --output-dir infer/demos \
    --model-name valle --norm-first true --add-prenet false \
    --share-embedding true --norm-first true --add-prenet false \
    --text-prompts "" \
    --audio-prompts "" \
    --text ljspeech.txt \
    --output-dir "infer/demos" \
    --checkpoint ${exp_dir}/best-valid-loss.pt