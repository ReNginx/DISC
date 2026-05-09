export MUJOCO_GL=egl
export TOKENIZERS_PARALLELISM=true
export CUDA_VISIBLE_DEVICES=$1

demo_num=$2
finetune_steps=$3
ckpt_path=$4   

python train.py \
    --config-name=libero_90_ttt \
    trainer.devices=1 \
    demo_num=$demo_num \
    finetune_steps=$finetune_steps \
    +ckpt_path=$ckpt_path \
    model.test_time_training_config.adaptation_method=policynet