#!/bin/bash
#SBATCH --job-name=pretrain
#SBATCH --cpus-per-task=144
#SBATCH --ntasks=1
#SBATCH --partition=gpujl
#SBATCH --mem=400G
#SBATCH --gres=gpu:4

#  --nodelist=node30

#export WANDB_API_KEY='e4f12e79df09008c8e830c15787c73c317d66fd3'
export WANDB_API_KEY='93531d18e07b0568f59a5f78092b69e9cc7dfca6'
export WANDB_BASE_URL=https://api.bandw.top
source /home/hujifeng/anaconda3/bin/activate diffusion_env


# todo grpo
export CUDA_VISIBLE_DEVICES=0
echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
python3 multistep_rl_flow_main.py --mode grpo_flow --debug_mode False --wandb_log True --rl_mode grpo --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --divergence_coef 1 --energy_scale 5 --lr 0.0001 --dataset maze2d-umaze-v1 --current_exp_label iql_rl_flow_div1_es5_grpo_adv_var_norm_reward_legacy > ./output/slurm_result/iql_rl_flow_div1_es5_grpo_adv_var_norm_reward_legacy-maze2d-umaze-v1.txt 2>&1 &
sleep 2

export CUDA_VISIBLE_DEVICES=0
echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
python3 multistep_rl_flow_main.py --mode grpo_flow --debug_mode False --wandb_log True --rl_mode grpo --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --divergence_coef 1 --energy_scale 5 --lr 0.0001 --dataset maze2d-medium-v1 --current_exp_label iql_rl_flow_div1_es5_grpo_adv_var_norm_reward_legacy > ./output/slurm_result/iql_rl_flow_div1_es5_grpo_adv_var_norm_reward_legacy-maze2d-medium-v1.txt 2>&1 &
sleep 2

export CUDA_VISIBLE_DEVICES=0
echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
python3 multistep_rl_flow_main.py --mode grpo_flow --debug_mode False --wandb_log True --rl_mode grpo --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --divergence_coef 1 --energy_scale 5 --lr 0.0001 --dataset maze2d-large-v1 --current_exp_label iql_rl_flow_div1_es5_grpo_adv_var_norm_reward_legacy > ./output/slurm_result/iql_rl_flow_div1_es5_grpo_adv_var_norm_reward_legacy-maze2d-large-v1.txt 2>&1 &
sleep 2

export CUDA_VISIBLE_DEVICES=1
echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
python3 multistep_rl_flow_main.py --mode grpo_flow --debug_mode False --wandb_log True --rl_mode grpo --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --divergence_coef 1 --energy_scale 5 --lr 0.0001 --dataset maze2d-umaze-dense-v1 --current_exp_label iql_rl_flow_div1_es5_grpo_adv_var_norm_reward_legacy > ./output/slurm_result/iql_rl_flow_div1_es5_grpo_adv_var_norm_reward_legacy-maze2d-umaze-dense-v1.txt 2>&1 &
sleep 2

export CUDA_VISIBLE_DEVICES=1
echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
python3 multistep_rl_flow_main.py --mode grpo_flow --debug_mode False --wandb_log True --rl_mode grpo --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --divergence_coef 1 --energy_scale 5 --lr 0.0001 --dataset maze2d-medium-dense-v1 --current_exp_label iql_rl_flow_div1_es5_grpo_adv_var_norm_reward_legacy > ./output/slurm_result/iql_rl_flow_div1_es5_grpo_adv_var_norm_reward_legacy-maze2d-medium-dense-v1.txt 2>&1 &
sleep 2

export CUDA_VISIBLE_DEVICES=1
echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
python3 multistep_rl_flow_main.py --mode grpo_flow --debug_mode False --wandb_log True --rl_mode grpo --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --divergence_coef 1 --energy_scale 5 --lr 0.0001 --dataset maze2d-large-dense-v1 --current_exp_label iql_rl_flow_div1_es5_grpo_adv_var_norm_reward_legacy > ./output/slurm_result/iql_rl_flow_div1_es5_grpo_adv_var_norm_reward_legacy-maze2d-large-dense-v1.txt 2>&1 &
sleep 2

export CUDA_VISIBLE_DEVICES=2
echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
python3 multistep_rl_flow_main.py --mode grpo_flow --debug_mode False --wandb_log True --rl_mode grpo --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --divergence_coef 0.5 --energy_scale 5 --lr 0.0001 --dataset maze2d-umaze-v1 --current_exp_label iql_rl_flow_div05_es5_grpo_adv_var_norm_reward_legacy > ./output/slurm_result/iql_rl_flow_div05_es5_grpo_adv_var_norm_reward_legacy-maze2d-umaze-v1.txt 2>&1 &
sleep 2

export CUDA_VISIBLE_DEVICES=2
echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
python3 multistep_rl_flow_main.py --mode grpo_flow --debug_mode False --wandb_log True --rl_mode grpo --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --divergence_coef 0.5 --energy_scale 5 --lr 0.0001 --dataset maze2d-medium-v1 --current_exp_label iql_rl_flow_div05_es5_grpo_adv_var_norm_reward_legacy > ./output/slurm_result/iql_rl_flow_div05_es5_grpo_adv_var_norm_reward_legacy-maze2d-medium-v1.txt 2>&1 &
sleep 2

export CUDA_VISIBLE_DEVICES=2
echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
python3 multistep_rl_flow_main.py --mode grpo_flow --debug_mode False --wandb_log True --rl_mode grpo --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --divergence_coef 0.5 --energy_scale 5 --lr 0.0001 --dataset maze2d-large-v1 --current_exp_label iql_rl_flow_div05_es5_grpo_adv_var_norm_reward_legacy > ./output/slurm_result/iql_rl_flow_div05_es5_grpo_adv_var_norm_reward_legacy-maze2d-large-v1.txt 2>&1 &
sleep 2

export CUDA_VISIBLE_DEVICES=3
echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
python3 multistep_rl_flow_main.py --mode grpo_flow --debug_mode False --wandb_log True --rl_mode grpo --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --divergence_coef 0.5 --energy_scale 5 --lr 0.0001 --dataset maze2d-umaze-dense-v1 --current_exp_label iql_rl_flow_div05_es5_grpo_adv_var_norm_reward_legacy > ./output/slurm_result/iql_rl_flow_div05_es5_grpo_adv_var_norm_reward_legacy-maze2d-umaze-dense-v1.txt 2>&1 &
sleep 2

export CUDA_VISIBLE_DEVICES=3
echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
python3 multistep_rl_flow_main.py --mode grpo_flow --debug_mode False --wandb_log True --rl_mode grpo --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --divergence_coef 0.5 --energy_scale 5 --lr 0.0001 --dataset maze2d-medium-dense-v1 --current_exp_label iql_rl_flow_div05_es5_grpo_adv_var_norm_reward_legacy > ./output/slurm_result/iql_rl_flow_div05_es5_grpo_adv_var_norm_reward_legacy-maze2d-medium-dense-v1.txt 2>&1 &
sleep 2

export CUDA_VISIBLE_DEVICES=3
echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
python3 multistep_rl_flow_main.py --mode grpo_flow --debug_mode False --wandb_log True --rl_mode grpo --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --divergence_coef 0.5 --energy_scale 5 --lr 0.0001 --dataset maze2d-large-dense-v1 --current_exp_label iql_rl_flow_div05_es5_grpo_adv_var_norm_reward_legacy > ./output/slurm_result/iql_rl_flow_div05_es5_grpo_adv_var_norm_reward_legacy-maze2d-large-dense-v1.txt 2>&1 &
sleep 2

##  todo Description: task: maze2d train mode is flow_constrained_rl4
#export CUDA_VISIBLE_DEVICES=0
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --divergence_coef 5 --energy_scale 10 --dataset maze2d-umaze-v1 --current_exp_label iql_rl_flow_div5_adv_rl > ./output/slurm_result/iql_rl_flow_div5_adv_rl-maze2d-umaze-v1.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=0
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --divergence_coef 5 --energy_scale 10 --dataset maze2d-medium-v1 --current_exp_label iql_rl_flow_div5_adv_rl > ./output/slurm_result/iql_rl_flow_div5_adv_rl-maze2d-medium-v1.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=0
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --divergence_coef 5 --energy_scale 10 --dataset maze2d-large-v1 --current_exp_label iql_rl_flow_div5_adv_rl > ./output/slurm_result/iql_rl_flow_div5_adv_rl-maze2d-large-v1.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=1
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --divergence_coef 5 --energy_scale 10 --dataset maze2d-umaze-dense-v1 --current_exp_label iql_rl_flow_div5_adv_rl > ./output/slurm_result/iql_rl_flow_div5_adv_rl-maze2d-umaze-dense-v1.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=1
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --divergence_coef 5 --energy_scale 10 --dataset maze2d-medium-dense-v1 --current_exp_label iql_rl_flow_div5_adv_rl > ./output/slurm_result/iql_rl_flow_div5_adv_rl-maze2d-medium-dense-v1.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=1
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --divergence_coef 5 --energy_scale 10 --dataset maze2d-large-dense-v1 --current_exp_label iql_rl_flow_div5_adv_rl > ./output/slurm_result/iql_rl_flow_div5_adv_rl-maze2d-large-dense-v1.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=2
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0  --divergence_coef 2 --energy_scale 10 --dataset maze2d-umaze-v1 --current_exp_label iql_rl_flow_div2_adv_rl > ./output/slurm_result/iql_rl_flow_div2_adv_rl-maze2d-umaze-v1.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=2
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0  --divergence_coef 2 --energy_scale 10 --dataset maze2d-medium-v1 --current_exp_label iql_rl_flow_div2_adv_rl > ./output/slurm_result/iql_rl_flow_div2_adv_rl-maze2d-medium-v1.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=2
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0  --divergence_coef 2 --energy_scale 10 --dataset maze2d-large-v1 --current_exp_label iql_rl_flow_div2_adv_rl > ./output/slurm_result/iql_rl_flow_div2_adv_rl-maze2d-large-v1.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=3
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0  --divergence_coef 2 --energy_scale 10 --dataset maze2d-umaze-dense-v1 --current_exp_label iql_rl_flow_div2_adv_rl > ./output/slurm_result/iql_rl_flow_div2_adv_rl-maze2d-umaze-dense-v1.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=3
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0  --divergence_coef 2 --energy_scale 10 --dataset maze2d-medium-dense-v1 --current_exp_label iql_rl_flow_div2_adv_rl > ./output/slurm_result/iql_rl_flow_div2_adv_rl-maze2d-medium-dense-v1.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=3
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0  --divergence_coef 2 --energy_scale 10 --dataset maze2d-large-dense-v1 --current_exp_label iql_rl_flow_div2_adv_rl > ./output/slurm_result/iql_rl_flow_div2_adv_rl-maze2d-large-dense-v1.txt 2>&1 &
#sleep 2


###  todo Description: task: maze2d train mode is flow_constrained_rl4
#export CUDA_VISIBLE_DEVICES=0
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl4 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 10 --update_energy_end_epoch 40 --update_behavior_end_epoch 200 --update_flow_start_epoch 40 --divergence_coef 0.01 --dataset maze2d-umaze-v1 --current_exp_label iql_rl_flow_div001_constrained_rl4 > ./output/slurm_result/iql_rl_flow_div001_constrained_rl4-maze2d-umaze-v1.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=0
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl4 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 10 --update_energy_end_epoch 40 --update_behavior_end_epoch 200 --update_flow_start_epoch 40 --divergence_coef 0.01 --dataset maze2d-medium-v1 --current_exp_label iql_rl_flow_div001_constrained_rl4 > ./output/slurm_result/iql_rl_flow_div001_constrained_rl4-maze2d-medium-v1.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=0
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl4 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 10 --update_energy_end_epoch 40 --update_behavior_end_epoch 200 --update_flow_start_epoch 40 --divergence_coef 0.01 --dataset maze2d-large-v1 --current_exp_label iql_rl_flow_div001_constrained_rl4 > ./output/slurm_result/iql_rl_flow_div001_constrained_rl4-maze2d-large-v1.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=1
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl4 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 10 --update_energy_end_epoch 40 --update_behavior_end_epoch 200 --update_flow_start_epoch 40 --divergence_coef 0.01 --dataset maze2d-umaze-dense-v1 --current_exp_label iql_rl_flow_div001_constrained_rl4 > ./output/slurm_result/iql_rl_flow_div001_constrained_rl4-maze2d-umaze-dense-v1.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=1
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl4 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 10 --update_energy_end_epoch 40 --update_behavior_end_epoch 200 --update_flow_start_epoch 40 --divergence_coef 0.01 --dataset maze2d-medium-dense-v1 --current_exp_label iql_rl_flow_div001_constrained_rl4 > ./output/slurm_result/iql_rl_flow_div001_constrained_rl4-maze2d-medium-dense-v1.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=1
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl4 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 10 --update_energy_end_epoch 40 --update_behavior_end_epoch 200 --update_flow_start_epoch 40 --divergence_coef 0.01 --dataset maze2d-large-dense-v1 --current_exp_label iql_rl_flow_div001_constrained_rl4 > ./output/slurm_result/iql_rl_flow_div001_constrained_rl4-maze2d-large-dense-v1.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=2
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl4 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 10 --update_energy_end_epoch 40 --update_behavior_end_epoch 200 --update_flow_start_epoch 40 --divergence_coef 1.0 --dataset maze2d-umaze-v1 --current_exp_label iql_rl_flow_div1_constrained_rl4 > ./output/slurm_result/iql_rl_flow_div1_constrained_rl4-maze2d-umaze-v1.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=2
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl4 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 10 --update_energy_end_epoch 40 --update_behavior_end_epoch 200 --update_flow_start_epoch 40 --divergence_coef 1.0 --dataset maze2d-medium-v1 --current_exp_label iql_rl_flow_div1_constrained_rl4 > ./output/slurm_result/iql_rl_flow_div1_constrained_rl4-maze2d-medium-v1.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=2
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl4 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 10 --update_energy_end_epoch 40 --update_behavior_end_epoch 200 --update_flow_start_epoch 40 --divergence_coef 1.0 --dataset maze2d-large-v1 --current_exp_label iql_rl_flow_div1_constrained_rl4 > ./output/slurm_result/iql_rl_flow_div1_constrained_rl4-maze2d-large-v1.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=3
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl4 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 10 --update_energy_end_epoch 40 --update_behavior_end_epoch 200 --update_flow_start_epoch 40 --divergence_coef 1.0 --dataset maze2d-umaze-dense-v1 --current_exp_label iql_rl_flow_div1_constrained_rl4 > ./output/slurm_result/iql_rl_flow_div1_constrained_rl4-maze2d-umaze-dense-v1.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=3
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl4 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 10 --update_energy_end_epoch 40 --update_behavior_end_epoch 200 --update_flow_start_epoch 40 --divergence_coef 1.0 --dataset maze2d-medium-dense-v1 --current_exp_label iql_rl_flow_div1_constrained_rl4 > ./output/slurm_result/iql_rl_flow_div1_constrained_rl4-maze2d-medium-dense-v1.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=3
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl4 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 10 --update_energy_end_epoch 40 --update_behavior_end_epoch 200 --update_flow_start_epoch 40 --divergence_coef 1.0 --dataset maze2d-large-dense-v1 --current_exp_label iql_rl_flow_div1_constrained_rl4 > ./output/slurm_result/iql_rl_flow_div1_constrained_rl4-maze2d-large-dense-v1.txt 2>&1 &
#sleep 2

##  todo Description: train mode is flow_constrained_rl4
#export CUDA_VISIBLE_DEVICES=0
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl4 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 40 --dataset antmaze-umaze-v0 --current_exp_label iql_rl_flow_constrained_rl4 > ./output/slurm_result/iql_rl_flow_constrained_rl4-antmaze-umaze-v0.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=0
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl4 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 40 --dataset antmaze-umaze-diverse-v0 --current_exp_label iql_rl_flow_constrained_rl4 > ./output/slurm_result/iql_rl_flow_constrained_rl4-antmaze-umaze-diverse-v0.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=0
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl4 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 40 --dataset antmaze-medium-play-v0 --current_exp_label iql_rl_flow_constrained_rl4 > ./output/slurm_result/iql_rl_flow_constrained_rl4-antmaze-medium-play-v0.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=0
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl4 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 40 --dataset antmaze-medium-diverse-v0 --current_exp_label iql_rl_flow_constrained_rl4 > ./output/slurm_result/iql_rl_flow_constrained_rl4-antmaze-medium-diverse-v0.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=0
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl4 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 40 --dataset antmaze-large-play-v0 --current_exp_label iql_rl_flow_constrained_rl4 > ./output/slurm_result/iql_rl_flow_constrained_rl4-antmaze-large-play-v0.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=1
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl4 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 40 --dataset antmaze-large-diverse-v0 --current_exp_label iql_rl_flow_constrained_rl4 > ./output/slurm_result/iql_rl_flow_constrained_rl4-antmaze-large-diverse-v0.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=1
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl4 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 40 --dataset antmaze-umaze-v1 --current_exp_label iql_rl_flow_constrained_rl4 > ./output/slurm_result/iql_rl_flow_constrained_rl4-antmaze-umaze-v1.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=1
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl4 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 40 --dataset antmaze-umaze-diverse-v1 --current_exp_label iql_rl_flow_constrained_rl4 > ./output/slurm_result/iql_rl_flow_constrained_rl4-antmaze-umaze-diverse-v1.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=1
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl4 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 40 --dataset antmaze-medium-play-v1 --current_exp_label iql_rl_flow_constrained_rl4 > ./output/slurm_result/iql_rl_flow_constrained_rl4-antmaze-medium-play-v1.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=2
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl4 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 40 --dataset antmaze-medium-diverse-v1 --current_exp_label iql_rl_flow_constrained_rl4 > ./output/slurm_result/iql_rl_flow_constrained_rl4-antmaze-medium-diverse-v1.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=2
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl4 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 40 --dataset antmaze-large-play-v1 --current_exp_label iql_rl_flow_constrained_rl4 > ./output/slurm_result/iql_rl_flow_constrained_rl4-antmaze-large-play-v1.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=2
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl4 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 40 --dataset antmaze-large-diverse-v1 --current_exp_label iql_rl_flow_constrained_rl4 > ./output/slurm_result/iql_rl_flow_constrained_rl4-antmaze-large-diverse-v1.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=2
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl4 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 40 --dataset maze2d-umaze-v1 --current_exp_label iql_rl_flow_constrained_rl4 > ./output/slurm_result/iql_rl_flow_constrained_rl4-maze2d-umaze-v1.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=2
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl4 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 40 --dataset maze2d-medium-v1 --current_exp_label iql_rl_flow_constrained_rl4 > ./output/slurm_result/iql_rl_flow_constrained_rl4-maze2d-medium-v1.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=3
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl4 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 40 --dataset maze2d-large-v1 --current_exp_label iql_rl_flow_constrained_rl4 > ./output/slurm_result/iql_rl_flow_constrained_rl4-maze2d-large-v1.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=3
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl4 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 40 --dataset maze2d-umaze-dense-v1 --current_exp_label iql_rl_flow_constrained_rl4 > ./output/slurm_result/iql_rl_flow_constrained_rl4-maze2d-umaze-dense-v1.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=3
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl4 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 40 --dataset maze2d-medium-dense-v1 --current_exp_label iql_rl_flow_constrained_rl4 > ./output/slurm_result/iql_rl_flow_constrained_rl4-maze2d-medium-dense-v1.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=3
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl4 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 40 --dataset maze2d-large-dense-v1 --current_exp_label iql_rl_flow_constrained_rl4 > ./output/slurm_result/iql_rl_flow_constrained_rl4-maze2d-large-dense-v1.txt 2>&1 &
#sleep 2


wait