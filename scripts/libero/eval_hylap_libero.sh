export MUJOCO_GL=egl
export TOKENIZERS_PARALLELISM=true

ckpt_path=$1
epoch=$2

python train.py \
    --config-name=libero_90_eval \
    model=hylap \
    data.batch_size=256 \
    data=libero_90 \
    trainer.devices=1 \
    data.eval_config.enabled=true \
    model.policy_network.use_compile=false \
    model.policy_network.num_layers=1 \
    precision=32 \
    +ckpt_path=$ckpt_path \
    +epoch=$epoch