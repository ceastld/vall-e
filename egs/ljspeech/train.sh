exp_dir=exp/valle1

CUDA_VISIBLE_DEVICES=0

## Train AR model
python bin/trainer.py --max-duration 80 --filter-min-duration 0.5 --filter-max-duration 14 --train-stage 1 \
      --num-buckets 6 --dtype "bfloat16" --save-every-n 10000 --valid-interval 20000 \
      --model-name valle --share-embedding true --norm-first true --add-prenet false \
      --decoder-dim 1024 --nhead 16 --num-decoder-layers 12 --prefix-mode 1 \
      --base-lr 0.05 --warmup-steps 200 --average-period 0 \
      --num-epochs 1000 --start-epoch 1 --start-batch 0 --accumulate-grad-steps 4 \
      --exp-dir ${exp_dir}

# nohup cmd > output.log 2>&1 &

# cp ${exp_dir}/best-valid-loss.pt ${exp_dir}/epoch-2.pt  # --start-epoch 3=2+1
# python bin/trainer.py --max-duration 40 --filter-min-duration 0.5 --filter-max-duration 14 --train-stage 2 \
#       --num-buckets 6 --dtype "float32" --save-every-n 10000 --valid-interval 20000 \
#       --model-name valle --share-embedding true --norm-first true --add-prenet false \
#       --decoder-dim 1024 --nhead 16 --num-decoder-layers 12 --prefix-mode 1 \
#       --base-lr 0.05 --warmup-steps 200 --average-period 0 \
#       --num-epochs 40 --start-epoch 3 --start-batch 0 --accumulate-grad-steps 4 \
#       --exp-dir ${exp_dir}

# python bin/infer.py --output-dir infer/demos \
#     --model-name valle --norm-first true --add-prenet false \
#     --share-embedding true --norm-first true --add-prenet false \
#     --text-prompts "In addition, the restriction would probably eliminate a need for the requirement which has been urged as necessary for the exercise of Federal power," \
#     --audio-prompts ./prompts/LJ049-0108_24K.wav \
#     --text "To get up and running quickly just follow the steps below." \
#     --checkpoint=${exp_dir}/best-valid-loss.pt