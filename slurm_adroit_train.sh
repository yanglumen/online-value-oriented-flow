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


# todo grpo experiments
export CUDA_VISIBLE_DEVICES=0
echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
python3 multistep_rl_flow_main.py --mode grpo_flow --debug_mode False --wandb_log True --rl_mode grpo --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --divergence_coef 1.0 --energy_scale 5 --eval_episodes 5 --dataset hammer-expert-v1 --current_exp_label iql_rl_flow_div1_es5_adv_rl > ./output/slurm_result/iql_rl_flow_div1_es5_adv_rl-hammer-expert-v1.txt 2>&1 &
sleep 2

export CUDA_VISIBLE_DEVICES=0
echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
python3 multistep_rl_flow_main.py --mode grpo_flow --debug_mode False --wandb_log True --rl_mode grpo --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --divergence_coef 1.0 --energy_scale 5 --eval_episodes 5 --dataset hammer-human-v1 --current_exp_label iql_rl_flow_div1_es5_adv_rl > ./output/slurm_result/iql_rl_flow_div1_es5_adv_rl-hammer-human-v1.txt 2>&1 &
sleep 2

export CUDA_VISIBLE_DEVICES=0
echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
python3 multistep_rl_flow_main.py --mode grpo_flow --debug_mode False --wandb_log True --rl_mode grpo --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --divergence_coef 1.0 --energy_scale 5 --eval_episodes 5 --dataset hammer-cloned-v1 --current_exp_label iql_rl_flow_div1_es5_adv_rl > ./output/slurm_result/iql_rl_flow_div1_es5_adv_rl-hammer-cloned-v1.txt 2>&1 &
sleep 2

export CUDA_VISIBLE_DEVICES=1
echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
python3 multistep_rl_flow_main.py --mode grpo_flow --debug_mode False --wandb_log True --rl_mode grpo --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --divergence_coef 1.0 --energy_scale 5 --eval_episodes 5 --dataset pen-expert-v1 --current_exp_label iql_rl_flow_div1_es5_adv_rl > ./output/slurm_result/iql_rl_flow_div1_es5_adv_rl-pen-expert-v1.txt 2>&1 &
sleep 2

export CUDA_VISIBLE_DEVICES=1
echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
python3 multistep_rl_flow_main.py --mode grpo_flow --debug_mode False --wandb_log True --rl_mode grpo --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --divergence_coef 1.0 --energy_scale 5 --eval_episodes 5 --dataset pen-human-v1 --current_exp_label iql_rl_flow_div1_es5_adv_rl > ./output/slurm_result/iql_rl_flow_div1_es5_adv_rl-pen-human-v1.txt 2>&1 &
sleep 2

export CUDA_VISIBLE_DEVICES=1
echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
python3 multistep_rl_flow_main.py --mode grpo_flow --debug_mode False --wandb_log True --rl_mode grpo --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --divergence_coef 1.0 --energy_scale 5 --eval_episodes 5 --dataset pen-cloned-v1 --current_exp_label iql_rl_flow_div1_es5_adv_rl > ./output/slurm_result/iql_rl_flow_div1_es5_adv_rl-pen-cloned-v1.txt 2>&1 &
sleep 2

export CUDA_VISIBLE_DEVICES=2
echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
python3 multistep_rl_flow_main.py --mode grpo_flow --debug_mode False --wandb_log True --rl_mode grpo --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --divergence_coef 1.0 --energy_scale 5 --eval_episodes 5 --dataset relocate-expert-v1 --current_exp_label iql_rl_flow_div1_es5_adv_rl > ./output/slurm_result/iql_rl_flow_div1_es5_adv_rl-relocate-expert-v1.txt 2>&1 &
sleep 2

export CUDA_VISIBLE_DEVICES=2
echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
python3 multistep_rl_flow_main.py --mode grpo_flow --debug_mode False --wandb_log True --rl_mode grpo --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --divergence_coef 1.0 --energy_scale 5 --eval_episodes 5 --dataset relocate-human-v1 --current_exp_label iql_rl_flow_div1_es5_adv_rl > ./output/slurm_result/iql_rl_flow_div1_es5_adv_rl-relocate-human-v1.txt 2>&1 &
sleep 2

export CUDA_VISIBLE_DEVICES=2
echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
python3 multistep_rl_flow_main.py --mode grpo_flow --debug_mode False --wandb_log True --rl_mode grpo --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --divergence_coef 1.0 --energy_scale 5 --eval_episodes 5 --dataset relocate-cloned-v1 --current_exp_label iql_rl_flow_div1_es5_adv_rl > ./output/slurm_result/iql_rl_flow_div1_es5_adv_rl-relocate-cloned-v1.txt 2>&1 &
sleep 2

export CUDA_VISIBLE_DEVICES=3
echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
python3 multistep_rl_flow_main.py --mode grpo_flow --debug_mode False --wandb_log True --rl_mode grpo --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --divergence_coef 1.0 --energy_scale 5 --eval_episodes 5 --dataset door-expert-v1 --current_exp_label iql_rl_flow_div1_es5_adv_rl > ./output/slurm_result/iql_rl_flow_div1_es5_adv_rl-door-expert-v1.txt 2>&1 &
sleep 2

export CUDA_VISIBLE_DEVICES=3
echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
python3 multistep_rl_flow_main.py --mode grpo_flow --debug_mode False --wandb_log True --rl_mode grpo --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --divergence_coef 1.0 --energy_scale 5 --eval_episodes 5 --dataset door-human-v1 --current_exp_label iql_rl_flow_div1_es5_adv_rl > ./output/slurm_result/iql_rl_flow_div1_es5_adv_rl-door-human-v1.txt 2>&1 &
sleep 2

export CUDA_VISIBLE_DEVICES=3
echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
python3 multistep_rl_flow_main.py --mode grpo_flow --debug_mode False --wandb_log True --rl_mode grpo --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --divergence_coef 1.0 --energy_scale 5 --eval_episodes 5 --dataset door-cloned-v1 --current_exp_label iql_rl_flow_div1_es5_adv_rl > ./output/slurm_result/iql_rl_flow_div1_es5_adv_rl-door-cloned-v1.txt 2>&1 &
sleep 2


# todo adv_rl variants experiments
#export CUDA_VISIBLE_DEVICES=0
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --divergence_coef 1.0 --energy_scale 15 --eval_episodes 5 --dataset pen-human-v1 --current_exp_label iql_rl_flow_div1_es15_adv_rl > ./output/slurm_result/iql_rl_flow_div1_es15_adv_rl-pen-human-v1.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=0
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --divergence_coef 1.0 --energy_scale 15 --eval_episodes 5 --dataset pen-cloned-v1 --current_exp_label iql_rl_flow_div1_es15_adv_rl > ./output/slurm_result/iql_rl_flow_div1_es15_adv_rl-pen-cloned-v1.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=1
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --iql_tau 0.7 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --divergence_coef 1.0 --energy_scale 10 --eval_episodes 5 --dataset pen-human-v1 --current_exp_label iql_rl_flow_div1_es10_tau07_adv_rl > ./output/slurm_result/iql_rl_flow_div1_es10_tau07_adv_rl-pen-human-v1.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=1
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --iql_tau 0.7 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --divergence_coef 1.0 --energy_scale 10 --eval_episodes 5 --dataset pen-cloned-v1 --current_exp_label iql_rl_flow_div1_es10_tau07_adv_rl > ./output/slurm_result/iql_rl_flow_div1_es10_tau07_adv_rl-pen-cloned-v1.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=2
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --divergence_coef 1.0 --energy_scale 10 --eval_episodes 5 --flow_step 15 --dataset pen-human-v1 --current_exp_label iql_rl_flow_div1_es10_ft15_adv_rl > ./output/slurm_result/iql_rl_flow_div1_es10_ft15_adv_rl-pen-human-v1.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=2
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --divergence_coef 1.0 --energy_scale 10 --eval_episodes 5 --flow_step 15 --dataset pen-cloned-v1 --current_exp_label iql_rl_flow_div1_es10_ft15_adv_rl > ./output/slurm_result/iql_rl_flow_div1_es10_ft15_adv_rl-pen-cloned-v1.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=3
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --iql_tau 0.7 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --divergence_coef 1.0 --energy_scale 10 --eval_episodes 5 --flow_step 15 --dataset pen-human-v1 --current_exp_label iql_rl_flow_div1_es10_tau07_ft15_adv_rl > ./output/slurm_result/iql_rl_flow_div1_es10_tau07_ft15_adv_rl-pen-human-v1.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=3
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --iql_tau 0.7 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --divergence_coef 1.0 --energy_scale 10 --eval_episodes 5 --flow_step 15 --dataset pen-cloned-v1 --current_exp_label iql_rl_flow_div1_es10_tau07_ft15_adv_rl > ./output/slurm_result/iql_rl_flow_div1_es10_tau07_ft15_adv_rl-pen-cloned-v1.txt 2>&1 &
#sleep 2


##  todo Description: train mode is adv_rl
#export CUDA_VISIBLE_DEVICES=0
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --divergence_coef 3.0 --energy_scale 5 --eval_episodes 5 --dataset hammer-expert-v1 --current_exp_label iql_rl_flow_div3_es5_adv_rl > ./output/slurm_result/iql_rl_flow_div3_es5_adv_rl-hammer-expert-v1.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=0
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --divergence_coef 3.0 --energy_scale 5 --eval_episodes 5 --dataset hammer-human-v1 --current_exp_label iql_rl_flow_div3_es5_adv_rl > ./output/slurm_result/iql_rl_flow_div3_es5_adv_rl-hammer-human-v1.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=0
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --divergence_coef 3.0 --energy_scale 5 --eval_episodes 5 --dataset hammer-cloned-v1 --current_exp_label iql_rl_flow_div3_es5_adv_rl > ./output/slurm_result/iql_rl_flow_div3_es5_adv_rl-hammer-cloned-v1.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=1
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --divergence_coef 3.0 --energy_scale 5 --eval_episodes 5 --dataset pen-expert-v1 --current_exp_label iql_rl_flow_div3_es5_adv_rl > ./output/slurm_result/iql_rl_flow_div3_es5_adv_rl-pen-expert-v1.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=1
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --divergence_coef 3.0 --energy_scale 5 --eval_episodes 5 --dataset pen-human-v1 --current_exp_label iql_rl_flow_div3_es5_adv_rl > ./output/slurm_result/iql_rl_flow_div3_es5_adv_rl-pen-human-v1.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=1
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --divergence_coef 3.0 --energy_scale 5 --eval_episodes 5 --dataset pen-cloned-v1 --current_exp_label iql_rl_flow_div3_es5_adv_rl > ./output/slurm_result/iql_rl_flow_div3_es5_adv_rl-pen-cloned-v1.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=2
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --divergence_coef 3.0 --energy_scale 5 --eval_episodes 5 --dataset relocate-expert-v1 --current_exp_label iql_rl_flow_div3_es5_adv_rl > ./output/slurm_result/iql_rl_flow_div3_es5_adv_rl-relocate-expert-v1.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=2
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --divergence_coef 3.0 --energy_scale 5 --eval_episodes 5 --dataset relocate-human-v1 --current_exp_label iql_rl_flow_div3_es5_adv_rl > ./output/slurm_result/iql_rl_flow_div3_es5_adv_rl-relocate-human-v1.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=2
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --divergence_coef 3.0 --energy_scale 5 --eval_episodes 5 --dataset relocate-cloned-v1 --current_exp_label iql_rl_flow_div3_es5_adv_rl > ./output/slurm_result/iql_rl_flow_div3_es5_adv_rl-relocate-cloned-v1.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=3
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --divergence_coef 3.0 --energy_scale 5 --eval_episodes 5 --dataset door-expert-v1 --current_exp_label iql_rl_flow_div3_es5_adv_rl > ./output/slurm_result/iql_rl_flow_div3_es5_adv_rl-door-expert-v1.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=3
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --divergence_coef 3.0 --energy_scale 5 --eval_episodes 5 --dataset door-human-v1 --current_exp_label iql_rl_flow_div3_es5_adv_rl > ./output/slurm_result/iql_rl_flow_div3_es5_adv_rl-door-human-v1.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=3
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --divergence_coef 3.0 --energy_scale 5 --eval_episodes 5 --dataset door-cloned-v1 --current_exp_label iql_rl_flow_div3_es5_adv_rl > ./output/slurm_result/iql_rl_flow_div3_es5_adv_rl-door-cloned-v1.txt 2>&1 &
#sleep 2


##  todo Description: train mode is flow_constrained_rl4
#export CUDA_VISIBLE_DEVICES=0
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl4 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --update_energy_end_epoch 40 --update_behavior_end_epoch 200 --update_flow_start_epoch 40 --divergence_coef 1.0 --dataset hammer-expert-v1 --current_exp_label iql_rl_flow_div1_constrained_rl4 > ./output/slurm_result/iql_rl_flow_div1_constrained_rl4-hammer-expert-v1.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=0
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl4 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --update_energy_end_epoch 40 --update_behavior_end_epoch 200 --update_flow_start_epoch 40 --divergence_coef 1.0 --dataset hammer-human-v1 --current_exp_label iql_rl_flow_div1_constrained_rl4 > ./output/slurm_result/iql_rl_flow_div1_constrained_rl4-hammer-human-v1.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=0
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl4 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --update_energy_end_epoch 40 --update_behavior_end_epoch 200 --update_flow_start_epoch 40 --divergence_coef 1.0 --dataset hammer-cloned-v1 --current_exp_label iql_rl_flow_div1_constrained_rl4 > ./output/slurm_result/iql_rl_flow_div1_constrained_rl4-hammer-cloned-v1.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=1
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl4 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --update_energy_end_epoch 40 --update_behavior_end_epoch 200 --update_flow_start_epoch 40 --divergence_coef 1.0 --dataset pen-expert-v1 --current_exp_label iql_rl_flow_div1_constrained_rl4 > ./output/slurm_result/iql_rl_flow_div1_constrained_rl4-pen-expert-v1.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=1
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl4 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --update_energy_end_epoch 40 --update_behavior_end_epoch 200 --update_flow_start_epoch 40 --divergence_coef 1.0 --dataset pen-human-v1 --current_exp_label iql_rl_flow_div1_constrained_rl4 > ./output/slurm_result/iql_rl_flow_div1_constrained_rl4-pen-human-v1.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=1
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl4 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --update_energy_end_epoch 40 --update_behavior_end_epoch 200 --update_flow_start_epoch 40 --divergence_coef 1.0 --dataset pen-cloned-v1 --current_exp_label iql_rl_flow_div1_constrained_rl4 > ./output/slurm_result/iql_rl_flow_div1_constrained_rl4-pen-cloned-v1.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=2
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl4 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --update_energy_end_epoch 40 --update_behavior_end_epoch 200 --update_flow_start_epoch 40 --divergence_coef 1.0 --dataset relocate-expert-v1 --current_exp_label iql_rl_flow_div1_constrained_rl4 > ./output/slurm_result/iql_rl_flow_div1_constrained_rl4-relocate-expert-v1.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=2
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl4 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --update_energy_end_epoch 40 --update_behavior_end_epoch 200 --update_flow_start_epoch 40 --divergence_coef 1.0 --dataset relocate-human-v1 --current_exp_label iql_rl_flow_div1_constrained_rl4 > ./output/slurm_result/iql_rl_flow_div1_constrained_rl4-relocate-human-v1.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=2
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl4 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --update_energy_end_epoch 40 --update_behavior_end_epoch 200 --update_flow_start_epoch 40 --divergence_coef 1.0 --dataset relocate-cloned-v1 --current_exp_label iql_rl_flow_div1_constrained_rl4 > ./output/slurm_result/iql_rl_flow_div1_constrained_rl4-relocate-cloned-v1.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=3
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl4 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --update_energy_end_epoch 40 --update_behavior_end_epoch 200 --update_flow_start_epoch 40 --divergence_coef 1.0 --dataset door-expert-v1 --current_exp_label iql_rl_flow_div1_constrained_rl4 > ./output/slurm_result/iql_rl_flow_div1_constrained_rl4-door-expert-v1.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=3
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl4 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --update_energy_end_epoch 40 --update_behavior_end_epoch 200 --update_flow_start_epoch 40 --divergence_coef 1.0 --dataset door-human-v1 --current_exp_label iql_rl_flow_div1_constrained_rl4 > ./output/slurm_result/iql_rl_flow_div1_constrained_rl4-door-human-v1.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=3
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl4 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --update_energy_end_epoch 40 --update_behavior_end_epoch 200 --update_flow_start_epoch 40 --divergence_coef 1.0 --dataset door-cloned-v1 --current_exp_label iql_rl_flow_div1_constrained_rl4 > ./output/slurm_result/iql_rl_flow_div1_constrained_rl4-door-cloned-v1.txt 2>&1 &
#sleep 2


wait