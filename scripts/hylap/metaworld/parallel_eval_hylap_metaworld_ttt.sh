export MUJOCO_GL=egl
export TOKENIZERS_PARALLELISM=true
export CUDA_VISIBLE_DEVICES=$1
export MUJOCO_EGL_DEVICE_ID=$1

ckpt_path=$2
config_path=$3

parallel -ut -j 1\
    python train.py \
        --config-name=$config_path \
        model=hylap \
        trainer.devices=1 \
        demo_num={1} \
        finetune_steps={2} \
        +ckpt_path=$ckpt_path \
        model.policy_network.use_compile=false \
        finetune_method=policynet \
    ::: 1 3 5 7 \
    ::: 2000 5000