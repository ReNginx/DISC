export MUJOCO_GL=egl
export TOKENIZERS_PARALLELISM=true
# export CUDA_VISIBLE_DEVICES=$1

export NVIDIA_DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES:-compute,utility}"
if [ -d /lib/x86_64-linux-gnu ]; then
    export LD_LIBRARY_PATH="/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

python train.py \
    --config-name=metaworld \
    model=hylap \
    trainer.devices=4 \
    data.batch_size=64 \
    data.eval_config.enabled=false \
    model.policy_network.use_compile=false \
    max_epochs=50 \
    +trainer.strategy=ddp_find_unused_parameters_true
