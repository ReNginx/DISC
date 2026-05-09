export MUJOCO_GL=egl
export TOKENIZERS_PARALLELISM=true
export CUDA_VISIBLE_DEVICES=$1
export MUJOCO_EGL_DEVICE_ID=$1

demo_num=$2
finetune_steps=$3
ckpt_path=$4   

python train.py \
    --config-name=metaworld_ttt \
    model=hylap \
    trainer.devices=1 \
    demo_num=$demo_num \
    finetune_steps=$finetune_steps \
    +ckpt_path=$ckpt_path \
    model.policy_network.use_compile=false \
    model.test_time_training_config.adaptation_method=policynet