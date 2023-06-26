exp_dir=exp/valle

mkdir -p $exp_dir
CUDA_VISIBLE_DEVICES="7" nohup python bin/trainer.py --max-duration 80 --filter-min-duration 0.5 --filter-max-duration 14 --train-stage 1 \
    --num-buckets 6 --dtype "bfloat16" --save-every-n 10000 --valid-interval 5000 \
    --model-name valle --share-embedding true --norm-first true --add-prenet false \
    --decoder-dim 1024 --nhead 16 --num-decoder-layers 12 --prefix-mode 1 \
    --base-lr 0.05 --warmup-steps 200 --average-period 0 \
    --world-size 1 \
    --num-epochs 100 --start-epoch 1 --start-batch 0 --accumulate-grad-steps 4 \
    --master-port 12355 \
    --exp-dir ${exp_dir} > ${exp_dir}/output1.log 2>&1 &

# genshin 数据训练出来过后，看一看用float32有没有什么提升，如果没有的话，就换成bfloat16 然后 max-duration 80
# mkdir -p ${exp_dir}1
# cp ${exp_dir}/best-valid-loss.pt ${exp_dir}1/epoch-1.pt

# CUDA_VISIBLE_DEVICES="4,5,6,7" nohup python bin/trainer.py --max-duration 80 --filter-min-duration 0.5 --filter-max-duration 14 --train-stage 2 \
#     --num-buckets 6 --dtype "bfloat16" --save-every-n 10000 --valid-interval 5000 \
#     --model-name valle --share-embedding true --norm-first true --add-prenet false \
#     --decoder-dim 1024 --nhead 16 --num-decoder-layers 12 --prefix-mode 1 \
#     --base-lr 0.05 --warmup-steps 200 --average-period 0 \
#     --num-epochs 200 --start-epoch 2 --start-batch 0 --accumulate-grad-steps 4 \
#     --master-port 12356 --world-size 4 \
#     --exp-dir ${exp_dir}1 > ${exp_dir}1/output2.log 2>&1 &