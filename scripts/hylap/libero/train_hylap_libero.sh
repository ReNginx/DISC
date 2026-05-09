export MUJOCO_GL=osmesa
export TOKENIZERS_PARALLELISM=true
export CUDA_VISIBLE_DEVICES=$1

python train.py \
    --config-name=hylap \
    experiment_name=hylap \
    data.batch_size=64 \
    version=libero_90 \
    data=libero_90 \
    trainer.devices=4 \
    data.eval_config.enabled=false \
    model.policy_network.use_compile=true \
    max_epochs=50