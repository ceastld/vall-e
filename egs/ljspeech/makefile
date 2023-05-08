prepare_data:
	bash prepare.sh --stage -1 --stop-stage 3 \
		--audio_extractor "Encodec" \
		--audio_feats_dir data/tokenized


exp_dir=exp/valle

tensorboard:
	tensorboard --logdir=${exp_dir} --port=1234

export CUDA_VISIBLE_DEVICES=3,4

# Train AR model
train_AR:
	python bin/trainer.py --max-duration 80 --filter-min-duration 0.5 --filter-max-duration 14 --train-stage 1 \
		--num-buckets 6 --dtype "bfloat16" --save-every-n 10000 --valid-interval 20000 \
		--model-name valle --share-embedding true --norm-first true --add-prenet false \
		--decoder-dim 1024 --nhead 16 --num-decoder-layers 12 --prefix-mode 1 \
		--base-lr 0.05 --warmup-steps 200 --average-period 0 \
		--num-epochs 1000 --start-epoch 184 --start-batch 0 --accumulate-grad-steps 4 \
		--exp-dir ${exp_dir}

# Train NAR model
train_NAR:
	cp ${exp_dir}/checkpoint-valid-loss.pt ${exp_dir}/epoch-2.pt  # --start-epoch 3=2+1
	python bin/trainer.py --max-duration 40 --filter-min-duration 0.5 --filter-max-duration 14 --train-stage 2 \
		--num-buckets 6 --dtype "float32" --save-every-n 10000 --valid-interval 20000 \
		--model-name valle --share-embedding true --norm-first true --add-prenet false \
		--decoder-dim 1024 --nhead 16 --num-decoder-layers 12 --prefix-mode 1 \
		--base-lr 0.05 --warmup-steps 200 --average-period 0 \
		--num-epochs 1000 --start-epoch 3 --start-batch 0 --accumulate-grad-steps 4 \
		--exp-dir ${exp_dir}

inference:
	CUDA_VISIBLE_DEVICES=1 python bin/infer.py --output-dir infer/demos \
		--model-name valle --norm-first true --add-prenet false \
		--share-embedding true --norm-first true --add-prenet false \
		--text-prompts "During the period the Commission was giving thought to this situation," \
		--audio-prompts ./prompts/LJ049-0185_24K.wav \
		--text "The government's consequences of a targeted official's assassination give the US power to act for protection." \
		--checkpoint=${exp_dir}/best-valid-loss.pt
