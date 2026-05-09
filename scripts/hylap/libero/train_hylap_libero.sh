export MUJOCO_GL=osmesa
export TOKENIZERS_PARALLELISM=true
export CUDA_VISIBLE_DEVICES=$1

python train.py \
    --config-name=hylap \
    experiment_name=hylap \
    data.batch_size=32 \
    version=libero_90 \
    data=libero_90 \
    trainer.devices=1 \
    data.eval_config.enabled=false \
    model.policy_network.use_compile=false \
    max_epochs=200