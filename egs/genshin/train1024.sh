exp_dir=exp/valle

# mkdir -p ${exp_dir}
# CUDA_VISIBLE_DEVICES="2" nohup python bin/trainer.py --max-duration 80 --filter-min-duration 0.5 --filter-max-duration 14 --train-stage 1 \
#     --num-buckets 6 --dtype "bfloat16" --save-every-n 5000 --valid-interval 2000 \
#     --model-name valle --share-embedding true --norm-first true --add-prenet false \
#     --decoder-dim 1024 --nhead 16 --num-decoder-layers 12 --prefix-mode 1 \
#     --base-lr 0.05 --warmup-steps 200 --average-period 0 \
#     --num-epochs 50 --start-epoch 3 --start-batch 0 --accumulate-grad-steps 4 \
#     --master-port 12356 \
#     --exp-dir ${exp_dir} > ${exp_dir}/output1.log 2>&1 &

## Train NAR model
# keep-last-k 保留最后多少个checkpoint


mkdir -p ${exp_dir}2
# cp ${exp_dir}/best-valid-loss.pt ${exp_dir}2/epoch-1.pt
CUDA_VISIBLE_DEVICES="3" nohup python bin/trainer.py --max-duration 40 --filter-min-duration 0.5 --filter-max-duration 14 --train-stage 2 \
    --num-buckets 6 --dtype "float32" --save-every-n 5000 --valid-interval 2000 \
    --model-name valle --share-embedding true --norm-first true --add-prenet false \
    --decoder-dim 1024 --nhead 16 --num-decoder-layers 12 --prefix-mode 1 \
    --base-lr 0.05 --warmup-steps 200 --average-period 0 \
    --num-epochs 1000 --start-epoch 2 --start-batch 0 --accumulate-grad-steps 4 \
    --master-port 12357 --world-size 1 \
    --exp-dir ${exp_dir}2 > ${exp_dir}2/output2.log 2>&1 &