export TOKENIZERS_PARALLELISM=true

python train.py \
    --config-name=hylap \
    experiment_name=hylap \
    data.batch_size=64 \
    version=libero_90 \
    data=libero_90 \
    trainer.devices=4 \
    data.eval_config.enabled=false \
    model.policy_network.use_compile=true \
    +trainer.strategy=ddp_find_unused_parameters_true \
    max_epochs=50