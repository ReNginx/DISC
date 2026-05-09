export MUJOCO_GL=egl
export TOKENIZERS_PARALLELISM=true

python train.py \
    --config-name=libero_90 \
    model=hylap \
    experiment_name=hylap_scenename \
    data.include_scene_name=true \
    data.batch_size=64 \
    trainer.devices=4 \
    data.eval_config.enabled=false \
    model.policy_network.use_compile=true \
    +trainer.ddp_find_unused_parameters=true