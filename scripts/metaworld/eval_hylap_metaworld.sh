export MUJOCO_GL=egl
export TOKENIZERS_PARALLELISM=true
export CUDA_VISIBLE_DEVICES=$1
export MUJOCO_EGL_DEVICE_ID=$1

python train.py \
    --config-name=hylap \
    experiment_name=hylap \
    data.batch_size=256 \
    version=meta_world_eval_sync_bn \
    data=metaworld \
    trainer.devices=1 \
    data.eval_config.enabled=true \
    model.policy_network.use_compile=false \
    model.image_encoder.convert_bn2ln=false \
    +trainer.limit_val_batches=1 \
    +stage=test \
    +ckpt_path=ckpt/hylap_meta_world_epoch_199_syncbn.ckpt