

prepare_data:
	bash prepare.sh --stage 0 --stop-stage 3

exp_dir=exp/valle

tensorboard:
	tensorboard --logdir=${exp_dir} --port=1234

## Train AR model
train_AR:
	nohup python bin/trainer.py --max-duration 80 --filter-min-duration 0.5 --filter-max-duration 14 --train-stage 1 \
		--num-buckets 6 --dtype "bfloat16" --save-every-n 10000 --valid-interval 20000 \
		--model-name valle --share-embedding true --norm-first true --add-prenet false \
		--decoder-dim 1024 --nhead 16 --num-decoder-layers 12 --prefix-mode 1 \
		--base-lr 0.05 --warmup-steps 200 --average-period 0 \
		--num-epochs 20 --start-epoch 1 --start-batch 0 --accumulate-grad-steps 4 \
		--exp-dir ${exp_dir} > ${exp_dir}/output1.log 2>&1 &


## Train NAR model
train_NAR:
	cp ${exp_dir}/best-valid-loss.pt ${exp_dir}/epoch-2.pt  # --start-epoch 3=2+1
	CUDA_VISIBLE_DEVICES=2 nohup python bin/trainer.py --max-duration 40 --filter-min-duration 0.5 --filter-max-duration 14 --train-stage 2 \
		--num-buckets 6 --dtype "float32" --save-every-n 10000 --valid-interval 20000 \
		--model-name valle --share-embedding true --norm-first true --add-prenet false \
		--decoder-dim 1024 --nhead 16 --num-decoder-layers 12 --prefix-mode 1 \
		--base-lr 0.05 --warmup-steps 200 --average-period 0 \
		--num-epochs 50 --start-epoch 3 --start-batch 0 --accumulate-grad-steps 4 \
		--exp-dir ${exp_dir} > ${exp_dir}/output2.log 2>&1 &

# step3 inference
inference:
	CUDA_VISIBLE_DEVICES=1 python bin/infer.py --output-dir infer/demos \
		--model-name valle --norm-first true --add-prenet false \
		--share-embedding true --norm-first true --add-prenet false \
		--text-prompts "KNOT one point one five miles per hour." \
		--audio-prompts ./prompts/8463_294825_000043_000000.wav \
		--text "To get up and running quickly just follow the steps below." \
		--checkpoint=${exp_dir}/best-valid-loss.pt