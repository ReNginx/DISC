export MUJOCO_GL=egl
export TOKENIZERS_PARALLELISM=true
export CUDA_VISIBLE_DEVICES=$1
export MUJOCO_EGL_DEVICE_ID=$1

ckpt_path=$2
epoch=$3

python train.py \
    --config-name=metaworld_eval \
    model=hylap \
    data.batch_size=256 \
    trainer.devices=1 \
    data.eval_config.enabled=true \
    model.policy_network.use_compile=false \
    +ckpt_path=$ckpt_path \
    +epoch=$epoch