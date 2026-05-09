export MUJOCO_GL=egl
export TOKENIZERS_PARALLELISM=true
export CUDA_VISIBLE_DEVICES=0

ckpt_path=$1

python train.py \
    --config-name=hylap \
    experiment_name=hylap-eval-action-chunking \
    data.batch_size=256 \
    version=libero_90_eval_ep20 \
    data=libero_90 \
    trainer.devices=1 \
    data.eval_config.enabled=true \
    model.policy_network.use_compile=false \
    model.policy_network.num_layers=1 \
    precision=32 \
    +trainer.limit_val_batches=1 \
    +stage=test \
    data.eval_config.num_parallel=20 \
    data.eval_config.num_episodes_per_task=20 \
    +ckpt_path=$ckpt_path