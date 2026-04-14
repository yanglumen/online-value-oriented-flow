#!/bin/bash
#SBATCH --job-name=pretrain
#SBATCH --cpus-per-task=144
#SBATCH --ntasks=1
#SBATCH --partition=gpujl
#SBATCH --mem=400G
#SBATCH --gres=gpu:4

#  --nodelist=node30

export WANDB_API_KEY='93531d18e07b0568f59a5f78092b69e9cc7dfca6'
export WANDB_BASE_URL=https://api.bandw.top
source /home/hujifeng/anaconda3/bin/activate diffusion_env

#  todo Description: train mode is QFloT
export CUDA_VISIBLE_DEVICES=0
echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
python3 flow_transformer_main.py --debug_mode False --wandb_log True --rl_mode Q_value_flow_tf --iql_tau 0.5 --Q_guided_coef 1.0 --load_critic_model iql_rl_flow --batch_size 256 --sequence_length 10 --energy_scale 10 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --dataset halfcheetah-medium-v2 --current_exp_label QFloT > ./output/slurm_result/QFloT-halfcheetah-medium-v2.txt 2>&1 &
sleep 2

export CUDA_VISIBLE_DEVICES=0
echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
python3 flow_transformer_main.py --debug_mode False --wandb_log True --rl_mode Q_value_flow_tf --iql_tau 0.5 --Q_guided_coef 1.0 --load_critic_model iql_rl_flow --batch_size 256 --sequence_length 10 --energy_scale 10 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --dataset halfcheetah-medium-expert-v2 --current_exp_label QFloT > ./output/slurm_result/QFloT-halfcheetah-medium-expert-v2.txt 2>&1 &
sleep 2

export CUDA_VISIBLE_DEVICES=0
echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
python3 flow_transformer_main.py --debug_mode False --wandb_log True --rl_mode Q_value_flow_tf --iql_tau 0.5 --Q_guided_coef 1.0 --load_critic_model iql_rl_flow --batch_size 256 --sequence_length 10 --energy_scale 10 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --dataset halfcheetah-medium-replay-v2 --current_exp_label QFloT > ./output/slurm_result/QFloT-halfcheetah-medium-replay-v2.txt 2>&1 &
sleep 2

export CUDA_VISIBLE_DEVICES=1
echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
python3 flow_transformer_main.py --debug_mode False --wandb_log True --rl_mode Q_value_flow_tf --iql_tau 0.5 --Q_guided_coef 1.0 --load_critic_model iql_rl_flow --batch_size 256 --sequence_length 10 --energy_scale 10 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --dataset halfcheetah-full-replay-v2 --current_exp_label QFloT > ./output/slurm_result/QFloT-halfcheetah-full-replay-v2.txt 2>&1 &
sleep 2

export CUDA_VISIBLE_DEVICES=1
echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
python3 flow_transformer_main.py --debug_mode False --wandb_log True --rl_mode Q_value_flow_tf --iql_tau 0.5 --Q_guided_coef 1.0 --load_critic_model iql_rl_flow --batch_size 256 --sequence_length 10 --energy_scale 10 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --dataset hopper-medium-v2 --current_exp_label QFloT > ./output/slurm_result/QFloT-hopper-medium-v2.txt 2>&1 &
sleep 2

export CUDA_VISIBLE_DEVICES=1
echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
python3 flow_transformer_main.py --debug_mode False --wandb_log True --rl_mode Q_value_flow_tf --iql_tau 0.5 --Q_guided_coef 1.0 --load_critic_model iql_rl_flow --batch_size 256 --sequence_length 10 --energy_scale 10 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --dataset hopper-medium-expert-v2 --current_exp_label QFloT > ./output/slurm_result/QFloT-hopper-medium-expert-v2.txt 2>&1 &
sleep 2

export CUDA_VISIBLE_DEVICES=2
echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
python3 flow_transformer_main.py --debug_mode False --wandb_log True --rl_mode Q_value_flow_tf --iql_tau 0.5 --Q_guided_coef 1.0 --load_critic_model iql_rl_flow --batch_size 256 --sequence_length 10 --energy_scale 10 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --dataset hopper-medium-replay-v2 --current_exp_label QFloT > ./output/slurm_result/QFloT-hopper-medium-replay-v2.txt 2>&1 &
sleep 2

export CUDA_VISIBLE_DEVICES=2
echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
python3 flow_transformer_main.py --debug_mode False --wandb_log True --rl_mode Q_value_flow_tf --iql_tau 0.5 --Q_guided_coef 1.0 --load_critic_model iql_rl_flow --batch_size 256 --sequence_length 10 --energy_scale 10 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --dataset hopper-full-replay-v2 --current_exp_label QFloT > ./output/slurm_result/QFloT-hopper-full-replay-v2.txt 2>&1 &
sleep 2

export CUDA_VISIBLE_DEVICES=2
echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
python3 flow_transformer_main.py --debug_mode False --wandb_log True --rl_mode Q_value_flow_tf --iql_tau 0.5 --Q_guided_coef 1.0 --load_critic_model iql_rl_flow --batch_size 256 --sequence_length 10 --energy_scale 10 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --dataset walker2d-medium-v2 --current_exp_label QFloT > ./output/slurm_result/QFloT-walker2d-medium-v2.txt 2>&1 &
sleep 2

export CUDA_VISIBLE_DEVICES=3
echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
python3 flow_transformer_main.py --debug_mode False --wandb_log True --rl_mode Q_value_flow_tf --iql_tau 0.5 --Q_guided_coef 1.0 --load_critic_model iql_rl_flow --batch_size 256 --sequence_length 10 --energy_scale 10 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --dataset walker2d-medium-expert-v2 --current_exp_label QFloT > ./output/slurm_result/QFloT-walker2d-medium-expert-v2.txt 2>&1 &
sleep 2

export CUDA_VISIBLE_DEVICES=3
echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
python3 flow_transformer_main.py --debug_mode False --wandb_log True --rl_mode Q_value_flow_tf --iql_tau 0.5 --Q_guided_coef 1.0 --load_critic_model iql_rl_flow --batch_size 256 --sequence_length 10 --energy_scale 10 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --dataset walker2d-medium-replay-v2 --current_exp_label QFloT > ./output/slurm_result/QFloT-walker2d-medium-replay-v2.txt 2>&1 &
sleep 2

export CUDA_VISIBLE_DEVICES=3
echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
python3 flow_transformer_main.py --debug_mode False --wandb_log True --rl_mode Q_value_flow_tf --iql_tau 0.5 --Q_guided_coef 1.0 --load_critic_model iql_rl_flow --batch_size 256 --sequence_length 10 --energy_scale 10 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --dataset walker2d-full-replay-v2 --current_exp_label QFloT > ./output/slurm_result/QFloT-walker2d-full-replay-v2.txt 2>&1 &
sleep 2

#  todo Description: train mode is flow_constrained_rl2
#export CUDA_VISIBLE_DEVICES=0
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --debug_mode False --wandb_log True --iql_tau 0.5 --critic_type ciql --load_critic_model ciql_rl_flow --batch_size 1024 --dataset halfcheetah-medium-v2 --current_exp_label ciql_rl_flow_flow_constrained_rl2 > ./output/slurm_result/ciql_rl_flow_flow_constrained_rl2-halfcheetah-medium-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=0
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --debug_mode False --wandb_log True --iql_tau 0.5 --critic_type ciql --load_critic_model ciql_rl_flow --batch_size 1024 --dataset halfcheetah-medium-expert-v2 --current_exp_label ciql_rl_flow_flow_constrained_rl2 > ./output/slurm_result/ciql_rl_flow_flow_constrained_rl2-halfcheetah-medium-expert-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=0
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --debug_mode False --wandb_log True --iql_tau 0.5 --critic_type ciql --load_critic_model ciql_rl_flow --batch_size 1024 --dataset halfcheetah-medium-replay-v2 --current_exp_label ciql_rl_flow_flow_constrained_rl2 > ./output/slurm_result/ciql_rl_flow_flow_constrained_rl2-halfcheetah-medium-replay-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES = 1
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --debug_mode False --wandb_log True --iql_tau 0.5 --critic_type ciql --load_critic_model ciql_rl_flow --batch_size 1024 --dataset halfcheetah-full-replay-v2 --current_exp_label ciql_rl_flow_flow_constrained_rl2 > ./output/slurm_result/ciql_rl_flow_flow_constrained_rl2-halfcheetah-full-replay-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES = 1
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --debug_mode False --wandb_log True --iql_tau 0.5 --critic_type ciql --load_critic_model ciql_rl_flow --batch_size 1024 --dataset hopper-medium-v2 --current_exp_label ciql_rl_flow_flow_constrained_rl2 > ./output/slurm_result/ciql_rl_flow_flow_constrained_rl2-hopper-medium-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES = 1
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --debug_mode False --wandb_log True --iql_tau 0.5 --critic_type ciql --load_critic_model ciql_rl_flow --batch_size 1024 --dataset hopper-medium-expert-v2 --current_exp_label ciql_rl_flow_flow_constrained_rl2 > ./output/slurm_result/ciql_rl_flow_flow_constrained_rl2-hopper-medium-expert-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES = 2
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --debug_mode False --wandb_log True --iql_tau 0.5 --critic_type ciql --load_critic_model ciql_rl_flow --batch_size 1024 --dataset hopper-medium-replay-v2 --current_exp_label ciql_rl_flow_flow_constrained_rl2 > ./output/slurm_result/ciql_rl_flow_flow_constrained_rl2-hopper-medium-replay-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES = 2
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --debug_mode False --wandb_log True --iql_tau 0.5 --critic_type ciql --load_critic_model ciql_rl_flow --batch_size 1024 --dataset hopper-full-replay-v2 --current_exp_label ciql_rl_flow_flow_constrained_rl2 > ./output/slurm_result/ciql_rl_flow_flow_constrained_rl2-hopper-full-replay-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES = 2
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --debug_mode False --wandb_log True --iql_tau 0.5 --critic_type ciql --load_critic_model ciql_rl_flow --batch_size 1024 --dataset walker2d-medium-v2 --current_exp_label ciql_rl_flow_flow_constrained_rl2 > ./output/slurm_result/ciql_rl_flow_flow_constrained_rl2-walker2d-medium-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES = 3
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --debug_mode False --wandb_log True --iql_tau 0.5 --critic_type ciql --load_critic_model ciql_rl_flow --batch_size 1024 --dataset walker2d-medium-expert-v2 --current_exp_label ciql_rl_flow_flow_constrained_rl2 > ./output/slurm_result/ciql_rl_flow_flow_constrained_rl2-walker2d-medium-expert-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES = 3
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --debug_mode False --wandb_log True --iql_tau 0.5 --critic_type ciql --load_critic_model ciql_rl_flow --batch_size 1024 --dataset walker2d-medium-replay-v2 --current_exp_label ciql_rl_flow_flow_constrained_rl2 > ./output/slurm_result/ciql_rl_flow_flow_constrained_rl2-walker2d-medium-replay-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES = 3
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --debug_mode False --wandb_log True --iql_tau 0.5 --critic_type ciql --load_critic_model ciql_rl_flow --batch_size 1024 --dataset walker2d-full-replay-v2 --current_exp_label ciql_rl_flow_flow_constrained_rl2 > ./output/slurm_result/ciql_rl_flow_flow_constrained_rl2-walker2d-full-replay-v2.txt 2>&1 &
#sleep 2

##  todo Description: train mode is grpo
#export CUDA_VISIBLE_DEVICES=0
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode grpo --iql_tau 0.5 --divergence_coef 1.0 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --energy_scale 5 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --dataset halfcheetah-medium-v2 --current_exp_label iql_grpo_flow_es5 > ./output/slurm_result/iql_grpo_flow_es5-halfcheetah-medium-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=0
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode grpo --iql_tau 0.5 --divergence_coef 1.0 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --energy_scale 5 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --dataset halfcheetah-medium-expert-v2 --current_exp_label iql_grpo_flow_es5 > ./output/slurm_result/iql_grpo_flow_es5-halfcheetah-medium-expert-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=0
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode grpo --iql_tau 0.5 --divergence_coef 1.0 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --energy_scale 5 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --dataset halfcheetah-medium-replay-v2 --current_exp_label iql_grpo_flow_es5 > ./output/slurm_result/iql_grpo_flow_es5-halfcheetah-medium-replay-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=1
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode grpo --iql_tau 0.5 --divergence_coef 1.0 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --energy_scale 5 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --dataset halfcheetah-full-replay-v2 --current_exp_label iql_grpo_flow_es5 > ./output/slurm_result/iql_grpo_flow_es5-halfcheetah-full-replay-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=1
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode grpo --iql_tau 0.5 --divergence_coef 1.0 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --energy_scale 5 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --dataset hopper-medium-v2 --current_exp_label iql_grpo_flow_es5 > ./output/slurm_result/iql_grpo_flow_es5-hopper-medium-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=1
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode grpo --iql_tau 0.5 --divergence_coef 1.0 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --energy_scale 5 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --dataset hopper-medium-expert-v2 --current_exp_label iql_grpo_flow_es5 > ./output/slurm_result/iql_grpo_flow_es5-hopper-medium-expert-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=2
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode grpo --iql_tau 0.5 --divergence_coef 1.0 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --energy_scale 5 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --dataset hopper-medium-replay-v2 --current_exp_label iql_grpo_flow_es5 > ./output/slurm_result/iql_grpo_flow_es5-hopper-medium-replay-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=2
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode grpo --iql_tau 0.5 --divergence_coef 1.0 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --energy_scale 5 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --dataset hopper-full-replay-v2 --current_exp_label iql_grpo_flow_es5 > ./output/slurm_result/iql_grpo_flow_es5-hopper-full-replay-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=2
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode grpo --iql_tau 0.5 --divergence_coef 1.0 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --energy_scale 5 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --dataset walker2d-medium-v2 --current_exp_label iql_grpo_flow_es5 > ./output/slurm_result/iql_grpo_flow_es5-walker2d-medium-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=3
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode grpo --iql_tau 0.5 --divergence_coef 1.0 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --energy_scale 5 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --dataset walker2d-medium-expert-v2 --current_exp_label iql_grpo_flow_es5 > ./output/slurm_result/iql_grpo_flow_es5-walker2d-medium-expert-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=3
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode grpo --iql_tau 0.5 --divergence_coef 1.0 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --energy_scale 5 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --dataset walker2d-medium-replay-v2 --current_exp_label iql_grpo_flow_es5 > ./output/slurm_result/iql_grpo_flow_es5-walker2d-medium-replay-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=3
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode grpo --iql_tau 0.5 --divergence_coef 1.0 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --energy_scale 5 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --dataset walker2d-full-replay-v2 --current_exp_label iql_grpo_flow_es5 > ./output/slurm_result/iql_grpo_flow_es5-walker2d-full-replay-v2.txt 2>&1 &
#sleep 2

##  todo Description: train mode is adv_rl_flow
#export CUDA_VISIBLE_DEVICES=0
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --iql_tau 0.5 --divergence_coef 1.0 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --energy_scale 10 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --dataset halfcheetah-medium-v2 --current_exp_label iql_adv_rl_flow_energy_scale_10 > ./output/slurm_result/iql_adv_rl_flow_energy_scale_10-halfcheetah-medium-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=0
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --iql_tau 0.5 --divergence_coef 1.0 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --energy_scale 10 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --dataset halfcheetah-medium-expert-v2 --current_exp_label iql_adv_rl_flow_energy_scale_10 > ./output/slurm_result/iql_adv_rl_flow_energy_scale_10-halfcheetah-medium-expert-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=0
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --iql_tau 0.5 --divergence_coef 1.0 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --energy_scale 10 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --dataset halfcheetah-medium-replay-v2 --current_exp_label iql_adv_rl_flow_energy_scale_10 > ./output/slurm_result/iql_adv_rl_flow_energy_scale_10-halfcheetah-medium-replay-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=1
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --iql_tau 0.5 --divergence_coef 1.0 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --energy_scale 10 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --dataset halfcheetah-full-replay-v2 --current_exp_label iql_adv_rl_flow_energy_scale_10 > ./output/slurm_result/iql_adv_rl_flow_energy_scale_10-halfcheetah-full-replay-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=1
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --iql_tau 0.5 --divergence_coef 1.0 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --energy_scale 10 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --dataset hopper-medium-v2 --current_exp_label iql_adv_rl_flow_energy_scale_10 > ./output/slurm_result/iql_adv_rl_flow_energy_scale_10-hopper-medium-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=1
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --iql_tau 0.5 --divergence_coef 1.0 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --energy_scale 10 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --dataset hopper-medium-expert-v2 --current_exp_label iql_adv_rl_flow_energy_scale_10 > ./output/slurm_result/iql_adv_rl_flow_energy_scale_10-hopper-medium-expert-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=2
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --iql_tau 0.5 --divergence_coef 1.0 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --energy_scale 10 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --dataset hopper-medium-replay-v2 --current_exp_label iql_adv_rl_flow_energy_scale_10 > ./output/slurm_result/iql_adv_rl_flow_energy_scale_10-hopper-medium-replay-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=2
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --iql_tau 0.5 --divergence_coef 1.0 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --energy_scale 10 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --dataset hopper-full-replay-v2 --current_exp_label iql_adv_rl_flow_energy_scale_10 > ./output/slurm_result/iql_adv_rl_flow_energy_scale_10-hopper-full-replay-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=2
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --iql_tau 0.5 --divergence_coef 1.0 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --energy_scale 10 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --dataset walker2d-medium-v2 --current_exp_label iql_adv_rl_flow_energy_scale_10 > ./output/slurm_result/iql_adv_rl_flow_energy_scale_10-walker2d-medium-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=3
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --iql_tau 0.5 --divergence_coef 1.0 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --energy_scale 10 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --dataset walker2d-medium-expert-v2 --current_exp_label iql_adv_rl_flow_energy_scale_10 > ./output/slurm_result/iql_adv_rl_flow_energy_scale_10-walker2d-medium-expert-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=3
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --iql_tau 0.5 --divergence_coef 1.0 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --energy_scale 10 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --dataset walker2d-medium-replay-v2 --current_exp_label iql_adv_rl_flow_energy_scale_10 > ./output/slurm_result/iql_adv_rl_flow_energy_scale_10-walker2d-medium-replay-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=3
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --iql_tau 0.5 --divergence_coef 1.0 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --energy_scale 10 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --dataset walker2d-full-replay-v2 --current_exp_label iql_adv_rl_flow_energy_scale_10 > ./output/slurm_result/iql_adv_rl_flow_energy_scale_10-walker2d-full-replay-v2.txt 2>&1 &
#sleep 2



##  todo Description: train mode is flow_constrained_rl3
#export CUDA_VISIBLE_DEVICES=0
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl3 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --update_energy_end_epoch 40 --update_behavior_end_epoch 200 --update_flow_start_epoch 40 --dataset halfcheetah-medium-v2 --current_exp_label iql_rl_flow_constrained_rl3 > ./output/slurm_result/iql_rl_flow_constrained_rl3-halfcheetah-medium-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=0
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl3 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --energy_scale 10 --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --update_energy_end_epoch 40 --update_behavior_end_epoch 200 --update_flow_start_epoch 40 --dataset halfcheetah-medium-expert-v2 --current_exp_label iql_rl_flow_constrained_rl3 > ./output/slurm_result/iql_rl_flow_constrained_rl3-halfcheetah-medium-expert-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=0
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl3 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --update_energy_end_epoch 40 --update_behavior_end_epoch 200 --update_flow_start_epoch 40 --dataset halfcheetah-medium-replay-v2 --current_exp_label iql_rl_flow_constrained_rl3 > ./output/slurm_result/iql_rl_flow_constrained_rl3-halfcheetah-medium-replay-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=1
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl3 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --update_energy_end_epoch 40 --update_behavior_end_epoch 200 --update_flow_start_epoch 40 --dataset halfcheetah-full-replay-v2 --current_exp_label iql_rl_flow_constrained_rl3 > ./output/slurm_result/iql_rl_flow_constrained_rl3-halfcheetah-full-replay-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=1
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl3 --iql_tau 0.5 --divergence_coef 0.3 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --update_energy_end_epoch 40 --update_behavior_end_epoch 200 --update_flow_start_epoch 40 --dataset hopper-medium-v2 --current_exp_label iql_rl_flow_constrained_rl3 > ./output/slurm_result/iql_rl_flow_constrained_rl3-hopper-medium-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=1
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl3 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --update_energy_end_epoch 40 --update_behavior_end_epoch 200 --update_flow_start_epoch 40 --dataset hopper-medium-expert-v2 --current_exp_label iql_rl_flow_constrained_rl3 > ./output/slurm_result/iql_rl_flow_constrained_rl3-hopper-medium-expert-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=2
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl3 --iql_tau 0.5 --divergence_coef 0.3 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --update_energy_end_epoch 40 --update_behavior_end_epoch 200 --update_flow_start_epoch 40 --dataset hopper-medium-replay-v2 --current_exp_label iql_rl_flow_constrained_rl3 > ./output/slurm_result/iql_rl_flow_constrained_rl3-hopper-medium-replay-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=2
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl3 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --update_energy_end_epoch 40 --update_behavior_end_epoch 200 --update_flow_start_epoch 40 --dataset hopper-full-replay-v2 --current_exp_label iql_rl_flow_constrained_rl3 > ./output/slurm_result/iql_rl_flow_constrained_rl3-hopper-full-replay-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=2
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl3 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --update_energy_end_epoch 40 --update_behavior_end_epoch 200 --update_flow_start_epoch 40 --dataset walker2d-medium-v2 --current_exp_label iql_rl_flow_constrained_rl3 > ./output/slurm_result/iql_rl_flow_constrained_rl3-walker2d-medium-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=3
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl3 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --update_energy_end_epoch 40 --update_behavior_end_epoch 200 --update_flow_start_epoch 40 --dataset walker2d-medium-expert-v2 --current_exp_label iql_rl_flow_constrained_rl3 > ./output/slurm_result/iql_rl_flow_constrained_rl3-walker2d-medium-expert-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=3
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl3 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --update_energy_end_epoch 40 --update_behavior_end_epoch 200 --update_flow_start_epoch 40 --dataset walker2d-medium-replay-v2 --current_exp_label iql_rl_flow_constrained_rl3 > ./output/slurm_result/iql_rl_flow_constrained_rl3-walker2d-medium-replay-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=3
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl3 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --update_energy_end_epoch 40 --update_behavior_end_epoch 200 --update_flow_start_epoch 40 --dataset walker2d-full-replay-v2 --current_exp_label iql_rl_flow_constrained_rl3 > ./output/slurm_result/iql_rl_flow_constrained_rl3-walker2d-full-replay-v2.txt 2>&1 &
#sleep 2



##  todo Description: train mode is flow_constrained_rl4
#export CUDA_VISIBLE_DEVICES=0
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl4 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --update_energy_end_epoch 40 --update_behavior_end_epoch 200 --update_flow_start_epoch 40 --dataset halfcheetah-medium-v2 --current_exp_label iql_rl_flow_constrained_rl4_scratch_behavior > ./output/slurm_result/iql_rl_flow_constrained_rl4_scratch_behavior-halfcheetah-medium-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=0
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl4 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --energy_scale 10 --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --update_energy_end_epoch 40 --update_behavior_end_epoch 200 --update_flow_start_epoch 40 --dataset halfcheetah-medium-expert-v2 --current_exp_label iql_rl_flow_constrained_rl4_scratch_behavior > ./output/slurm_result/iql_rl_flow_constrained_rl4_scratch_behavior-halfcheetah-medium-expert-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=0
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl4 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --update_energy_end_epoch 40 --update_behavior_end_epoch 200 --update_flow_start_epoch 40 --dataset halfcheetah-medium-replay-v2 --current_exp_label iql_rl_flow_constrained_rl4_scratch_behavior > ./output/slurm_result/iql_rl_flow_constrained_rl4_scratch_behavior-halfcheetah-medium-replay-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=1
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl4 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --update_energy_end_epoch 40 --update_behavior_end_epoch 200 --update_flow_start_epoch 40 --dataset halfcheetah-full-replay-v2 --current_exp_label iql_rl_flow_constrained_rl4_scratch_behavior > ./output/slurm_result/iql_rl_flow_constrained_rl4_scratch_behavior-halfcheetah-full-replay-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=1
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl4 --iql_tau 0.5 --divergence_coef 0.3 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --update_energy_end_epoch 40 --update_behavior_end_epoch 200 --update_flow_start_epoch 40 --dataset hopper-medium-v2 --current_exp_label iql_rl_flow_constrained_rl4_scratch_behavior > ./output/slurm_result/iql_rl_flow_constrained_rl4_scratch_behavior-hopper-medium-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=1
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl4 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --update_energy_end_epoch 40 --update_behavior_end_epoch 200 --update_flow_start_epoch 40 --dataset hopper-medium-expert-v2 --current_exp_label iql_rl_flow_constrained_rl4_scratch_behavior > ./output/slurm_result/iql_rl_flow_constrained_rl4_scratch_behavior-hopper-medium-expert-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=2
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl4 --iql_tau 0.5 --divergence_coef 0.3 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --update_energy_end_epoch 40 --update_behavior_end_epoch 200 --update_flow_start_epoch 40 --dataset hopper-medium-replay-v2 --current_exp_label iql_rl_flow_constrained_rl4_scratch_behavior > ./output/slurm_result/iql_rl_flow_constrained_rl4_scratch_behavior-hopper-medium-replay-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=2
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl4 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --update_energy_end_epoch 40 --update_behavior_end_epoch 200 --update_flow_start_epoch 40 --dataset hopper-full-replay-v2 --current_exp_label iql_rl_flow_constrained_rl4_scratch_behavior > ./output/slurm_result/iql_rl_flow_constrained_rl4_scratch_behavior-hopper-full-replay-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=2
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl4 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --update_energy_end_epoch 40 --update_behavior_end_epoch 200 --update_flow_start_epoch 40 --dataset walker2d-medium-v2 --current_exp_label iql_rl_flow_constrained_rl4_scratch_behavior > ./output/slurm_result/iql_rl_flow_constrained_rl4_scratch_behavior-walker2d-medium-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=3
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl4 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --update_energy_end_epoch 40 --update_behavior_end_epoch 200 --update_flow_start_epoch 40 --dataset walker2d-medium-expert-v2 --current_exp_label iql_rl_flow_constrained_rl4_scratch_behavior > ./output/slurm_result/iql_rl_flow_constrained_rl4_scratch_behavior-walker2d-medium-expert-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=3
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl4 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --update_energy_end_epoch 40 --update_behavior_end_epoch 200 --update_flow_start_epoch 40 --dataset walker2d-medium-replay-v2 --current_exp_label iql_rl_flow_constrained_rl4_scratch_behavior > ./output/slurm_result/iql_rl_flow_constrained_rl4_scratch_behavior-walker2d-medium-replay-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=3
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl4 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --update_energy_end_epoch 40 --update_behavior_end_epoch 200 --update_flow_start_epoch 40 --dataset walker2d-full-replay-v2 --current_exp_label iql_rl_flow_constrained_rl4_scratch_behavior > ./output/slurm_result/iql_rl_flow_constrained_rl4_scratch_behavior-walker2d-full-replay-v2.txt 2>&1 &
#sleep 2

##  todo Description: train mode is flow_constrained_rl5
#export CUDA_VISIBLE_DEVICES=0
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl5 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --energy_scale 20 --batch_size 1024 --flow_constrained_rl5_multiple_actions 20 --dataset halfcheetah-medium-v2 --current_exp_label iql_rl_flow_constrained_rl5 > ./output/slurm_result/iql_rl_flow_constrained_rl5-halfcheetah-medium-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=0
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl5 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --energy_scale 20 --batch_size 1024 --flow_constrained_rl5_multiple_actions 20 --dataset halfcheetah-medium-expert-v2 --current_exp_label iql_rl_flow_constrained_rl5 > ./output/slurm_result/iql_rl_flow_constrained_rl5-halfcheetah-medium-expert-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=0
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl5 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --energy_scale 20 --batch_size 1024 --flow_constrained_rl5_multiple_actions 20 --dataset halfcheetah-medium-replay-v2 --current_exp_label iql_rl_flow_constrained_rl5 > ./output/slurm_result/iql_rl_flow_constrained_rl5-halfcheetah-medium-replay-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=1
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl5 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --energy_scale 20 --batch_size 1024 --flow_constrained_rl5_multiple_actions 20 --dataset halfcheetah-full-replay-v2 --current_exp_label iql_rl_flow_constrained_rl5 > ./output/slurm_result/iql_rl_flow_constrained_rl5-halfcheetah-full-replay-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=1
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl5 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --energy_scale 20 --batch_size 1024 --flow_constrained_rl5_multiple_actions 20 --dataset hopper-medium-v2 --current_exp_label iql_rl_flow_constrained_rl5 > ./output/slurm_result/iql_rl_flow_constrained_rl5-hopper-medium-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=1
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl5 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --energy_scale 20 --batch_size 1024 --flow_constrained_rl5_multiple_actions 20 --dataset hopper-medium-expert-v2 --current_exp_label iql_rl_flow_constrained_rl5 > ./output/slurm_result/iql_rl_flow_constrained_rl5-hopper-medium-expert-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=2
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl5 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --energy_scale 20 --batch_size 1024 --flow_constrained_rl5_multiple_actions 20 --dataset hopper-medium-replay-v2 --current_exp_label iql_rl_flow_constrained_rl5 > ./output/slurm_result/iql_rl_flow_constrained_rl5-hopper-medium-replay-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=2
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl5 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --energy_scale 20 --batch_size 1024 --flow_constrained_rl5_multiple_actions 20 --dataset hopper-full-replay-v2 --current_exp_label iql_rl_flow_constrained_rl5 > ./output/slurm_result/iql_rl_flow_constrained_rl5-hopper-full-replay-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=2
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl5 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --energy_scale 20 --batch_size 1024 --flow_constrained_rl5_multiple_actions 20 --dataset walker2d-medium-v2 --current_exp_label iql_rl_flow_constrained_rl5 > ./output/slurm_result/iql_rl_flow_constrained_rl5-walker2d-medium-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=3
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl5 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --energy_scale 20 --batch_size 1024 --flow_constrained_rl5_multiple_actions 20 --dataset walker2d-medium-expert-v2 --current_exp_label iql_rl_flow_constrained_rl5 > ./output/slurm_result/iql_rl_flow_constrained_rl5-walker2d-medium-expert-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=3
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl5 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --energy_scale 20 --batch_size 1024 --flow_constrained_rl5_multiple_actions 20 --dataset walker2d-medium-replay-v2 --current_exp_label iql_rl_flow_constrained_rl5 > ./output/slurm_result/iql_rl_flow_constrained_rl5-walker2d-medium-replay-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=3
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl5 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --energy_scale 20 --batch_size 1024 --flow_constrained_rl5_multiple_actions 20 --dataset walker2d-full-replay-v2 --current_exp_label iql_rl_flow_constrained_rl5 > ./output/slurm_result/iql_rl_flow_constrained_rl5-walker2d-full-replay-v2.txt 2>&1 &
#sleep 2

##  todo Description: train mode is use_rl_q
#export CUDA_VISIBLE_DEVICES=0
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode use_rl_q --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --dataset halfcheetah-medium-v2 --current_exp_label iql_rl_flow_use_rl_q > ./output/slurm_result/iql_rl_flow_use_rl_q-halfcheetah-medium-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=0
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode use_rl_q --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --dataset halfcheetah-medium-expert-v2 --current_exp_label iql_rl_flow_use_rl_q > ./output/slurm_result/iql_rl_flow_use_rl_q-halfcheetah-medium-expert-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=0
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode use_rl_q --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --dataset halfcheetah-medium-replay-v2 --current_exp_label iql_rl_flow_use_rl_q > ./output/slurm_result/iql_rl_flow_use_rl_q-halfcheetah-medium-replay-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES = 1
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode use_rl_q --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --dataset halfcheetah-full-replay-v2 --current_exp_label iql_rl_flow_use_rl_q > ./output/slurm_result/iql_rl_flow_use_rl_q-halfcheetah-full-replay-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES = 1
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode use_rl_q --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --dataset hopper-medium-v2 --current_exp_label iql_rl_flow_use_rl_q > ./output/slurm_result/iql_rl_flow_use_rl_q-hopper-medium-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES = 1
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode use_rl_q --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --dataset hopper-medium-expert-v2 --current_exp_label iql_rl_flow_use_rl_q > ./output/slurm_result/iql_rl_flow_use_rl_q-hopper-medium-expert-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES = 2
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode use_rl_q --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --dataset hopper-medium-replay-v2 --current_exp_label iql_rl_flow_use_rl_q > ./output/slurm_result/iql_rl_flow_use_rl_q-hopper-medium-replay-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES = 2
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode use_rl_q --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --dataset hopper-full-replay-v2 --current_exp_label iql_rl_flow_use_rl_q > ./output/slurm_result/iql_rl_flow_use_rl_q-hopper-full-replay-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES = 2
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode use_rl_q --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --dataset walker2d-medium-v2 --current_exp_label iql_rl_flow_use_rl_q > ./output/slurm_result/iql_rl_flow_use_rl_q-walker2d-medium-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES = 3
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode use_rl_q --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --dataset walker2d-medium-expert-v2 --current_exp_label iql_rl_flow_use_rl_q > ./output/slurm_result/iql_rl_flow_use_rl_q-walker2d-medium-expert-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES = 3
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode use_rl_q --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --dataset walker2d-medium-replay-v2 --current_exp_label iql_rl_flow_use_rl_q > ./output/slurm_result/iql_rl_flow_use_rl_q-walker2d-medium-replay-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES = 3
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode use_rl_q --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --dataset walker2d-full-replay-v2 --current_exp_label iql_rl_flow_use_rl_q > ./output/slurm_result/iql_rl_flow_use_rl_q-walker2d-full-replay-v2.txt 2>&1 &
#sleep 2

##  todo Description: train mode is adv_rl with IQL
#export CUDA_VISIBLE_DEVICES=0
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --flow_step 20 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 128 --adv_rl_multiple_actions 50 --dataset halfcheetah-medium-v2 --current_exp_label iql_rl_flow_adv_rl > ./output/slurm_result/iql_rl_flow_adv_rl-halfcheetah-medium-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=0
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --flow_step 20 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 128 --adv_rl_multiple_actions 50 --dataset halfcheetah-medium-expert-v2 --current_exp_label iql_rl_flow_adv_rl > ./output/slurm_result/iql_rl_flow_adv_rl-halfcheetah-medium-expert-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=0
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --flow_step 20 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 128 --adv_rl_multiple_actions 50 --dataset halfcheetah-medium-replay-v2 --current_exp_label iql_rl_flow_adv_rl > ./output/slurm_result/iql_rl_flow_adv_rl-halfcheetah-medium-replay-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES = 1
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --flow_step 20 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 128 --adv_rl_multiple_actions 50 --dataset halfcheetah-full-replay-v2 --current_exp_label iql_rl_flow_adv_rl > ./output/slurm_result/iql_rl_flow_adv_rl-halfcheetah-full-replay-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES = 1
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --flow_step 20 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 128 --adv_rl_multiple_actions 50 --dataset hopper-medium-v2 --current_exp_label iql_rl_flow_adv_rl > ./output/slurm_result/iql_rl_flow_adv_rl-hopper-medium-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES = 1
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --flow_step 20 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 128 --adv_rl_multiple_actions 50 --dataset hopper-medium-expert-v2 --current_exp_label iql_rl_flow_adv_rl > ./output/slurm_result/iql_rl_flow_adv_rl-hopper-medium-expert-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES = 2
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --flow_step 20 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 128 --adv_rl_multiple_actions 50 --dataset hopper-medium-replay-v2 --current_exp_label iql_rl_flow_adv_rl > ./output/slurm_result/iql_rl_flow_adv_rl-hopper-medium-replay-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES = 2
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --flow_step 20 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 128 --adv_rl_multiple_actions 50 --dataset hopper-full-replay-v2 --current_exp_label iql_rl_flow_adv_rl > ./output/slurm_result/iql_rl_flow_adv_rl-hopper-full-replay-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES = 2
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --flow_step 20 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 128 --adv_rl_multiple_actions 50 --dataset walker2d-medium-v2 --current_exp_label iql_rl_flow_adv_rl > ./output/slurm_result/iql_rl_flow_adv_rl-walker2d-medium-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES = 3
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --flow_step 20 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 128 --adv_rl_multiple_actions 50 --dataset walker2d-medium-expert-v2 --current_exp_label iql_rl_flow_adv_rl > ./output/slurm_result/iql_rl_flow_adv_rl-walker2d-medium-expert-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES = 3
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --flow_step 20 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 128 --adv_rl_multiple_actions 50 --dataset walker2d-medium-replay-v2 --current_exp_label iql_rl_flow_adv_rl > ./output/slurm_result/iql_rl_flow_adv_rl-walker2d-medium-replay-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES = 3
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --flow_step 20 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 128 --adv_rl_multiple_actions 50 --dataset walker2d-full-replay-v2 --current_exp_label iql_rl_flow_adv_rl > ./output/slurm_result/iql_rl_flow_adv_rl-walker2d-full-replay-v2.txt 2>&1 &
#sleep 2

##  todo Description: train mode is adv_rl with CIQL
#export CUDA_VISIBLE_DEVICES=0
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --flow_step 20 --iql_tau 0.5 --critic_type ciql --load_critic_model ciql_rl_flow --batch_size 128 --adv_rl_multiple_actions 50 --dataset halfcheetah-medium-v2 --current_exp_label ciql_rl_flow_adv_rl > ./output/slurm_result/ciql_rl_flow_adv_rl-halfcheetah-medium-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=0
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --flow_step 20 --iql_tau 0.5 --critic_type ciql --load_critic_model ciql_rl_flow --batch_size 128 --adv_rl_multiple_actions 50 --dataset halfcheetah-medium-expert-v2 --current_exp_label ciql_rl_flow_adv_rl > ./output/slurm_result/ciql_rl_flow_adv_rl-halfcheetah-medium-expert-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=0
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --flow_step 20 --iql_tau 0.5 --critic_type ciql --load_critic_model ciql_rl_flow --batch_size 128 --adv_rl_multiple_actions 50 --dataset halfcheetah-medium-replay-v2 --current_exp_label ciql_rl_flow_adv_rl > ./output/slurm_result/ciql_rl_flow_adv_rl-halfcheetah-medium-replay-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES = 1
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --flow_step 20 --iql_tau 0.5 --critic_type ciql --load_critic_model ciql_rl_flow --batch_size 128 --adv_rl_multiple_actions 50 --dataset halfcheetah-full-replay-v2 --current_exp_label ciql_rl_flow_adv_rl > ./output/slurm_result/ciql_rl_flow_adv_rl-halfcheetah-full-replay-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES = 1
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --flow_step 20 --iql_tau 0.5 --critic_type ciql --load_critic_model ciql_rl_flow --batch_size 128 --adv_rl_multiple_actions 50 --dataset hopper-medium-v2 --current_exp_label ciql_rl_flow_adv_rl > ./output/slurm_result/ciql_rl_flow_adv_rl-hopper-medium-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES = 1
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --flow_step 20 --iql_tau 0.5 --critic_type ciql --load_critic_model ciql_rl_flow --batch_size 128 --adv_rl_multiple_actions 50 --dataset hopper-medium-expert-v2 --current_exp_label ciql_rl_flow_adv_rl > ./output/slurm_result/ciql_rl_flow_adv_rl-hopper-medium-expert-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES = 2
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --flow_step 20 --iql_tau 0.5 --critic_type ciql --load_critic_model ciql_rl_flow --batch_size 128 --adv_rl_multiple_actions 50 --dataset hopper-medium-replay-v2 --current_exp_label ciql_rl_flow_adv_rl > ./output/slurm_result/ciql_rl_flow_adv_rl-hopper-medium-replay-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES = 2
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --flow_step 20 --iql_tau 0.5 --critic_type ciql --load_critic_model ciql_rl_flow --batch_size 128 --adv_rl_multiple_actions 50 --dataset hopper-full-replay-v2 --current_exp_label ciql_rl_flow_adv_rl > ./output/slurm_result/ciql_rl_flow_adv_rl-hopper-full-replay-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES = 2
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --flow_step 20 --iql_tau 0.5 --critic_type ciql --load_critic_model ciql_rl_flow --batch_size 128 --adv_rl_multiple_actions 50 --dataset walker2d-medium-v2 --current_exp_label ciql_rl_flow_adv_rl > ./output/slurm_result/ciql_rl_flow_adv_rl-walker2d-medium-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES = 3
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --flow_step 20 --iql_tau 0.5 --critic_type ciql --load_critic_model ciql_rl_flow --batch_size 128 --adv_rl_multiple_actions 50 --dataset walker2d-medium-expert-v2 --current_exp_label ciql_rl_flow_adv_rl > ./output/slurm_result/ciql_rl_flow_adv_rl-walker2d-medium-expert-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES = 3
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --flow_step 20 --iql_tau 0.5 --critic_type ciql --load_critic_model ciql_rl_flow --batch_size 128 --adv_rl_multiple_actions 50 --dataset walker2d-medium-replay-v2 --current_exp_label ciql_rl_flow_adv_rl > ./output/slurm_result/ciql_rl_flow_adv_rl-walker2d-medium-replay-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES = 3
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --flow_step 20 --iql_tau 0.5 --critic_type ciql --load_critic_model ciql_rl_flow --batch_size 128 --adv_rl_multiple_actions 50 --dataset walker2d-full-replay-v2 --current_exp_label ciql_rl_flow_adv_rl > ./output/slurm_result/ciql_rl_flow_adv_rl-walker2d-full-replay-v2.txt 2>&1 &
#sleep 2

wait