

prepare_data:
	bash prepare.sh --stage 0 --stop-stage 3

exp_dir=exp/valle2

tensorboard:
	tensorboard --logdir=${exp_dir} --port=1234

## Train AR model
# 多线程训练，保持频率稍微高一点
train_AR:
	CUDA_VISIBLE_DEVICES="2,3" nohup python bin/trainer.py --max-duration 80 --filter-min-duration 0.5 --filter-max-duration 14 --train-stage 1 \
		--num-buckets 6 --dtype "bfloat16" --save-every-n 5000 --valid-interval 10000 \
		--model-name valle --share-embedding true --norm-first true --add-prenet false \
		--decoder-dim 1024 --nhead 16 --num-decoder-layers 12 --prefix-mode 1 \
		--base-lr 0.05 --warmup-steps 200 --average-period 0 \
		--world-size 2 \
		--num-epochs 50 --start-epoch 1 --start-batch 0 --accumulate-grad-steps 4 \
		--master-port 12355 \
		--exp-dir ${exp_dir} > ${exp_dir}/output1.log 2>&1 &

# 单卡训练
train_AR1:
	CUDA_VISIBLE_DEVICES="2" nohup python bin/trainer.py --max-duration 80 --filter-min-duration 0.5 --filter-max-duration 14 --train-stage 1 \
		--num-buckets 6 --dtype "bfloat16" --save-every-n 5000 --valid-interval 10000 \
		--model-name valle --share-embedding true --norm-first true --add-prenet false \
		--decoder-dim 1024 --nhead 16 --num-decoder-layers 12 --prefix-mode 1 \
		--base-lr 0.05 --warmup-steps 200 --average-period 0 \
		--num-epochs 50 --start-epoch 1 --start-batch 0 --accumulate-grad-steps 4 \
		--master-port 12356 \
		--exp-dir ${exp_dir} > ${exp_dir}/output1.log 2>&1 &

## Train NAR model
train_NAR:
	CUDA_VISIBLE_DEVICES="4,5,6,7" nohup python bin/trainer.py --max-duration 40 --filter-min-duration 0.5 --filter-max-duration 14 --train-stage 2 \
		--num-buckets 6 --dtype "float32" --save-every-n 5000 --valid-interval 5000 \
		--keep-last-k 40 \
		--model-name valle --share-embedding true --norm-first true --add-prenet false \
		--decoder-dim 1024 --nhead 16 --num-decoder-layers 12 --prefix-mode 1 \
		--base-lr 0.05 --warmup-steps 200 --average-period 0 \
		--num-epochs 100 --start-epoch 7 --start-batch 0 --accumulate-grad-steps 4 \
		--world-size 4 \
		--exp-dir ${exp_dir} > ${exp_dir}/output2.log 2>&1 &
