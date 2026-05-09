export MUJOCO_GL=egl
export TOKENIZERS_PARALLELISM=true
export CUDA_VISIBLE_DEVICES=1

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
    +ckpt_path=/mnt/sda/renhanxiang/Project/lang2policy/HyLaP/logs/hylap_ntasks/tasks_80_job_05/checkpoints/epoch_39-val_loss_0.0075_success_rate_0.0000.ckpt