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


##  todo Description: task: antmaze train mode is grpo

export CUDA_VISIBLE_DEVICES=0
echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode grpo --reward_tune cql_antmaze --iql_tau 0.5 --divergence_coef 1.0 --critic_type iql --load_critic_model none --batch_size 1024 --energy_scale 10 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --dataset antmaze-large-play-v2 --current_exp_label iql_grpo_flow_es10 > ./output/slurm_result/iql_grpo_flow_es10-antmaze-large-play-v2.txt 2>&1 &
sleep 2

export CUDA_VISIBLE_DEVICES=0
echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode grpo --reward_tune cql_antmaze --iql_tau 0.5 --divergence_coef 1.0 --critic_type iql --load_critic_model none --batch_size 1024 --energy_scale 10 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --dataset antmaze-large-play-v1 --current_exp_label iql_grpo_flow_es10 > ./output/slurm_result/iql_grpo_flow_es10-antmaze-large-play-v1.txt 2>&1 &
sleep 2

export CUDA_VISIBLE_DEVICES=0
echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode grpo --reward_tune cql_antmaze --iql_tau 0.5 --divergence_coef 1.0 --critic_type iql --load_critic_model none --batch_size 1024 --energy_scale 10 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --dataset antmaze-large-play-v0 --current_exp_label iql_grpo_flow_es10 > ./output/slurm_result/iql_grpo_flow_es10-antmaze-large-play-v0.txt 2>&1 &
sleep 2

export CUDA_VISIBLE_DEVICES=1
echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode grpo --reward_tune cql_antmaze --iql_tau 0.5 --divergence_coef 1.0 --critic_type iql --load_critic_model none --batch_size 1024 --energy_scale 10 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --dataset antmaze-large-diverse-v2 --current_exp_label iql_grpo_flow_es10 > ./output/slurm_result/iql_grpo_flow_es10-antmaze-large-diverse-v2.txt 2>&1 &
sleep 2

export CUDA_VISIBLE_DEVICES=1
echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode grpo --reward_tune cql_antmaze --iql_tau 0.5 --divergence_coef 1.0 --critic_type iql --load_critic_model none --batch_size 1024 --energy_scale 10 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --dataset antmaze-large-diverse-v1 --current_exp_label iql_grpo_flow_es10 > ./output/slurm_result/iql_grpo_flow_es10-antmaze-large-diverse-v1.txt 2>&1 &
sleep 2

export CUDA_VISIBLE_DEVICES=1
echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode grpo --reward_tune cql_antmaze --iql_tau 0.5 --divergence_coef 1.0 --critic_type iql --load_critic_model none --batch_size 1024 --energy_scale 10 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --dataset antmaze-large-diverse-v0 --current_exp_label iql_grpo_flow_es10 > ./output/slurm_result/iql_grpo_flow_es10-antmaze-large-diverse-v0.txt 2>&1 &
sleep 2

export CUDA_VISIBLE_DEVICES=2
echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode grpo --reward_tune cql_antmaze --iql_tau 0.5 --divergence_coef 1.0 --critic_type iql --load_critic_model none --batch_size 1024 --energy_scale 10 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --dataset antmaze-medium-play-v2 --current_exp_label iql_grpo_flow_es10 > ./output/slurm_result/iql_grpo_flow_es10-antmaze-medium-play-v2.txt 2>&1 &
sleep 2

export CUDA_VISIBLE_DEVICES=2
echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode grpo --reward_tune cql_antmaze --iql_tau 0.5 --divergence_coef 1.0 --critic_type iql --load_critic_model none --batch_size 1024 --energy_scale 10 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --dataset antmaze-medium-play-v1 --current_exp_label iql_grpo_flow_es10 > ./output/slurm_result/iql_grpo_flow_es10-antmaze-medium-play-v1.txt 2>&1 &
sleep 2

export CUDA_VISIBLE_DEVICES=2
echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode grpo --reward_tune cql_antmaze --iql_tau 0.5 --divergence_coef 1.0 --critic_type iql --load_critic_model none --batch_size 1024 --energy_scale 10 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --dataset antmaze-medium-play-v0 --current_exp_label iql_grpo_flow_es10 > ./output/slurm_result/iql_grpo_flow_es10-antmaze-medium-play-v0.txt 2>&1 &
sleep 2

export CUDA_VISIBLE_DEVICES=3
echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode grpo --reward_tune cql_antmaze --iql_tau 0.5 --divergence_coef 1.0 --critic_type iql --load_critic_model none --batch_size 1024 --energy_scale 10 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --dataset antmaze-medium-diverse-v2 --current_exp_label iql_grpo_flow_es10 > ./output/slurm_result/iql_grpo_flow_es10-antmaze-medium-diverse-v2.txt 2>&1 &
sleep 2

export CUDA_VISIBLE_DEVICES=3
echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode grpo --reward_tune cql_antmaze --iql_tau 0.5 --divergence_coef 1.0 --critic_type iql --load_critic_model none --batch_size 1024 --energy_scale 10 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --dataset antmaze-medium-diverse-v1 --current_exp_label iql_grpo_flow_es10 > ./output/slurm_result/iql_grpo_flow_es10-antmaze-medium-diverse-v1.txt 2>&1 &
sleep 2

export CUDA_VISIBLE_DEVICES=3
echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode grpo --reward_tune cql_antmaze --iql_tau 0.5 --divergence_coef 1.0 --critic_type iql --load_critic_model none --batch_size 1024 --energy_scale 10 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --dataset antmaze-medium-diverse-v0 --current_exp_label iql_grpo_flow_es10 > ./output/slurm_result/iql_grpo_flow_es10-antmaze-medium-diverse-v0.txt 2>&1 &
sleep 2



###  todo Description: task: antmaze train mode is DF-dir
#
#export CUDA_VISIBLE_DEVICES=0
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --reward_tune cql_antmaze --iql_tau 0.8 --critic_type iql --load_critic_model none --batch_size 1024 --divergence_coef 1.0 --energy_scale 10 --dataset antmaze-large-play-v2 --current_exp_label iql_rl_flow_div1_adv_rl_cql_antmaze_tau08_naive_return_tune > ./output/slurm_result/iql_rl_flow_div1_adv_rl_cql_antmaze_tau08_naive_return_tune-antmaze-large-play-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=0
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --reward_tune cql_antmaze --iql_tau 0.8 --critic_type iql --load_critic_model none --batch_size 1024 --divergence_coef 1.0 --energy_scale 10 --dataset antmaze-large-play-v1 --current_exp_label iql_rl_flow_div1_adv_rl_cql_antmaze_tau08_naive_return_tune > ./output/slurm_result/iql_rl_flow_div1_adv_rl_cql_antmaze_tau08_naive_return_tune-antmaze-large-play-v1.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=0
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --reward_tune cql_antmaze --iql_tau 0.8 --critic_type iql --load_critic_model none --batch_size 1024 --divergence_coef 1.0 --energy_scale 10 --dataset antmaze-large-play-v0 --current_exp_label iql_rl_flow_div1_adv_rl_cql_antmaze_tau08_naive_return_tune > ./output/slurm_result/iql_rl_flow_div1_adv_rl_cql_antmaze_tau08_naive_return_tune-antmaze-large-play-v0.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=1
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --reward_tune cql_antmaze --iql_tau 0.8 --critic_type iql --load_critic_model none --batch_size 1024 --divergence_coef 1.0 --energy_scale 10 --dataset antmaze-large-diverse-v2 --current_exp_label iql_rl_flow_div1_adv_rl_cql_antmaze_tau08_naive_return_tune > ./output/slurm_result/iql_rl_flow_div1_adv_rl_cql_antmaze_tau08_naive_return_tune-antmaze-large-diverse-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=1
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --reward_tune cql_antmaze --iql_tau 0.8 --critic_type iql --load_critic_model none --batch_size 1024 --divergence_coef 1.0 --energy_scale 10 --dataset antmaze-large-diverse-v1 --current_exp_label iql_rl_flow_div1_adv_rl_cql_antmaze_tau08_naive_return_tune > ./output/slurm_result/iql_rl_flow_div1_adv_rl_cql_antmaze_tau08_naive_return_tune-antmaze-large-diverse-v1.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=1
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --reward_tune cql_antmaze --iql_tau 0.8 --critic_type iql --load_critic_model none --batch_size 1024 --divergence_coef 1.0 --energy_scale 10 --dataset antmaze-large-diverse-v0 --current_exp_label iql_rl_flow_div1_adv_rl_cql_antmaze_tau08_naive_return_tune > ./output/slurm_result/iql_rl_flow_div1_adv_rl_cql_antmaze_tau08_naive_return_tune-antmaze-large-diverse-v0.txt 2>&1 &
#sleep 2




#export CUDA_VISIBLE_DEVICES=0
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --reward_tune cql_antmaze --iql_tau 0.8 --critic_type iql --load_critic_model none --batch_size 1024 --divergence_coef 1.0 --energy_scale 10 --dataset antmaze-large-play-v2 --current_exp_label iql_rl_flow_div1_adv_rl_cql_antmaze_tau08 > ./output/slurm_result/iql_rl_flow_div1_adv_rl_cql_antmaze_tau08-antmaze-large-play-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=0
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --reward_tune cql_antmaze --iql_tau 0.7 --critic_type iql --load_critic_model none --batch_size 1024 --divergence_coef 1.0 --energy_scale 10 --dataset antmaze-large-play-v2 --current_exp_label iql_rl_flow_div1_adv_rl_cql_antmaze_tau07 > ./output/slurm_result/iql_rl_flow_div1_adv_rl_cql_antmaze_tau07-antmaze-large-play-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=0
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --reward_tune cql_antmaze --iql_tau 0.85 --critic_type iql --load_critic_model none --batch_size 1024 --divergence_coef 1.0 --energy_scale 10 --dataset antmaze-large-play-v2 --current_exp_label iql_rl_flow_div1_adv_rl_cql_antmaze_tau085 > ./output/slurm_result/iql_rl_flow_div1_adv_rl_cql_antmaze_tau085-antmaze-large-play-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=1
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --reward_tune cql_antmaze --iql_tau 0.8 --critic_type iql --load_critic_model none --batch_size 1024 --divergence_coef 1.0 --energy_scale 10 --dataset antmaze-large-diverse-v2 --current_exp_label iql_rl_flow_div1_adv_rl_cql_antmaze_tau08 > ./output/slurm_result/iql_rl_flow_div1_adv_rl_cql_antmaze_tau08-antmaze-large-diverse-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=1
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --reward_tune cql_antmaze --iql_tau 0.7 --critic_type iql --load_critic_model none --batch_size 1024 --divergence_coef 1.0 --energy_scale 10 --dataset antmaze-large-diverse-v2 --current_exp_label iql_rl_flow_div1_adv_rl_cql_antmaze_tau07 > ./output/slurm_result/iql_rl_flow_div1_adv_rl_cql_antmaze_tau07-antmaze-large-diverse-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=1
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --reward_tune cql_antmaze --iql_tau 0.85 --critic_type iql --load_critic_model none --batch_size 1024 --divergence_coef 1.0 --energy_scale 10 --dataset antmaze-large-diverse-v2 --current_exp_label iql_rl_flow_div1_adv_rl_cql_antmaze_tau085 > ./output/slurm_result/iql_rl_flow_div1_adv_rl_cql_antmaze_tau085-antmaze-large-diverse-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=2
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --reward_tune cql_antmaze --iql_tau 0.8 --critic_type iql --load_critic_model none --batch_size 1024 --divergence_coef 1.0 --energy_scale 10 --dataset antmaze-large-play-v1 --current_exp_label iql_rl_flow_div1_adv_rl_cql_antmaze_tau08 > ./output/slurm_result/iql_rl_flow_div1_adv_rl_cql_antmaze_tau08-antmaze-large-play-v1.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=2
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --reward_tune cql_antmaze --iql_tau 0.7 --critic_type iql --load_critic_model none --batch_size 1024 --divergence_coef 1.0 --energy_scale 10 --dataset antmaze-large-play-v1 --current_exp_label iql_rl_flow_div1_adv_rl_cql_antmaze_tau07 > ./output/slurm_result/iql_rl_flow_div1_adv_rl_cql_antmaze_tau07-antmaze-large-play-v1.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=2
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --reward_tune cql_antmaze --iql_tau 0.85 --critic_type iql --load_critic_model none --batch_size 1024 --divergence_coef 1.0 --energy_scale 10 --dataset antmaze-large-play-v1 --current_exp_label iql_rl_flow_div1_adv_rl_cql_antmaze_tau085 > ./output/slurm_result/iql_rl_flow_div1_adv_rl_cql_antmaze_tau085-antmaze-large-play-v1.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=3
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --reward_tune cql_antmaze --iql_tau 0.8 --critic_type iql --load_critic_model none --batch_size 1024 --divergence_coef 1.0 --energy_scale 10 --dataset antmaze-large-diverse-v1 --current_exp_label iql_rl_flow_div1_adv_rl_cql_antmaze_tau08 > ./output/slurm_result/iql_rl_flow_div1_adv_rl_cql_antmaze_tau08-antmaze-large-diverse-v1.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=3
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --reward_tune cql_antmaze --iql_tau 0.7 --critic_type iql --load_critic_model none --batch_size 1024 --divergence_coef 1.0 --energy_scale 10 --dataset antmaze-large-diverse-v1 --current_exp_label iql_rl_flow_div1_adv_rl_cql_antmaze_tau07 > ./output/slurm_result/iql_rl_flow_div1_adv_rl_cql_antmaze_tau07-antmaze-large-diverse-v1.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=3
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --reward_tune cql_antmaze --iql_tau 0.85 --critic_type iql --load_critic_model none --batch_size 1024 --divergence_coef 1.0 --energy_scale 10 --dataset antmaze-large-diverse-v1 --current_exp_label iql_rl_flow_div1_adv_rl_cql_antmaze_tau085 > ./output/slurm_result/iql_rl_flow_div1_adv_rl_cql_antmaze_tau085-antmaze-large-diverse-v1.txt 2>&1 &
#sleep 2




#export CUDA_VISIBLE_DEVICES=0
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --reward_tune antmaze --iql_tau 0.8 --critic_type iql --load_critic_model none --batch_size 1024 --divergence_coef 1.0 --energy_scale 10 --dataset antmaze-large-play-v0 --current_exp_label iql_rl_flow_div1_adv_rl_antmaze_tau08 > ./output/slurm_result/iql_rl_flow_div1_adv_rl_antmaze_tau08-antmaze-large-play-v0.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=0
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES 
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --reward_tune antmaze --iql_tau 0.7 --critic_type iql --load_critic_model none --batch_size 1024 --divergence_coef 1.0 --energy_scale 10 --dataset antmaze-large-play-v0 --current_exp_label iql_rl_flow_div1_adv_rl_antmaze_tau07 > ./output/slurm_result/iql_rl_flow_div1_adv_rl_antmaze_tau07-antmaze-large-play-v0.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=0
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --reward_tune antmaze --iql_tau 0.85 --critic_type iql --load_critic_model none --batch_size 1024 --divergence_coef 1.0 --energy_scale 10 --dataset antmaze-large-play-v0 --current_exp_label iql_rl_flow_div1_adv_rl_antmaze_tau085 > ./output/slurm_result/iql_rl_flow_div1_adv_rl_antmaze_tau085-antmaze-large-play-v0.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=1
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --reward_tune antmaze --iql_tau 0.8 --critic_type iql --load_critic_model none --batch_size 1024 --divergence_coef 1.0 --energy_scale 10 --dataset antmaze-large-diverse-v0 --current_exp_label iql_rl_flow_div1_adv_rl_antmaze_tau08 > ./output/slurm_result/iql_rl_flow_div1_adv_rl_antmaze_tau08-antmaze-large-diverse-v0.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=1
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --reward_tune antmaze --iql_tau 0.7 --critic_type iql --load_critic_model none --batch_size 1024 --divergence_coef 1.0 --energy_scale 10 --dataset antmaze-large-diverse-v0 --current_exp_label iql_rl_flow_div1_adv_rl_antmaze_tau07 > ./output/slurm_result/iql_rl_flow_div1_adv_rl_antmaze_tau07-antmaze-large-diverse-v0.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=1
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --reward_tune antmaze --iql_tau 0.85 --critic_type iql --load_critic_model none --batch_size 1024 --divergence_coef 1.0 --energy_scale 10 --dataset antmaze-large-diverse-v0 --current_exp_label iql_rl_flow_div1_adv_rl_antmaze_tau085 > ./output/slurm_result/iql_rl_flow_div1_adv_rl_antmaze_tau085-antmaze-large-diverse-v0.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=2
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --reward_tune cql_antmaze --iql_tau 0.8 --critic_type iql --load_critic_model none --batch_size 1024 --divergence_coef 1.0 --energy_scale 10 --dataset antmaze-large-play-v0 --current_exp_label iql_rl_flow_div1_adv_rl_cql_antmaze_tau08 > ./output/slurm_result/iql_rl_flow_div1_adv_rl_cql_antmaze_tau08-antmaze-large-play-v0.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=2
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --reward_tune cql_antmaze --iql_tau 0.7 --critic_type iql --load_critic_model none --batch_size 1024 --divergence_coef 1.0 --energy_scale 10 --dataset antmaze-large-play-v0 --current_exp_label iql_rl_flow_div1_adv_rl_cql_antmaze_tau07 > ./output/slurm_result/iql_rl_flow_div1_adv_rl_cql_antmaze_tau07-antmaze-large-play-v0.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=2
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --reward_tune cql_antmaze --iql_tau 0.85 --critic_type iql --load_critic_model none --batch_size 1024 --divergence_coef 1.0 --energy_scale 10 --dataset antmaze-large-play-v0 --current_exp_label iql_rl_flow_div1_adv_rl_cql_antmaze_tau085 > ./output/slurm_result/iql_rl_flow_div1_adv_rl_cql_antmaze_tau085-antmaze-large-play-v0.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=3
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --reward_tune cql_antmaze --iql_tau 0.8 --critic_type iql --load_critic_model none --batch_size 1024 --divergence_coef 1.0 --energy_scale 10 --dataset antmaze-large-diverse-v0 --current_exp_label iql_rl_flow_div1_adv_rl_cql_antmaze_tau08 > ./output/slurm_result/iql_rl_flow_div1_adv_rl_cql_antmaze_tau08-antmaze-large-diverse-v0.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=3
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --reward_tune cql_antmaze --iql_tau 0.7 --critic_type iql --load_critic_model none --batch_size 1024 --divergence_coef 1.0 --energy_scale 10 --dataset antmaze-large-diverse-v0 --current_exp_label iql_rl_flow_div1_adv_rl_cql_antmaze_tau07 > ./output/slurm_result/iql_rl_flow_div1_adv_rl_cql_antmaze_tau07-antmaze-large-diverse-v0.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=3
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --reward_tune cql_antmaze --iql_tau 0.85 --critic_type iql --load_critic_model none --batch_size 1024 --divergence_coef 1.0 --energy_scale 10 --dataset antmaze-large-diverse-v0 --current_exp_label iql_rl_flow_div1_adv_rl_cql_antmaze_tau085 > ./output/slurm_result/iql_rl_flow_div1_adv_rl_cql_antmaze_tau085-antmaze-large-diverse-v0.txt 2>&1 &
#sleep 2




#export CUDA_VISIBLE_DEVICES=0
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --reward_tune antmaze --iql_tau 0.8 --critic_type iql --load_critic_model none --batch_size 1024 --divergence_coef 1.0 --energy_scale 10 --dataset antmaze-large-play-v2 --current_exp_label iql_rl_flow_div1_adv_rl_antmaze_tau08 > ./output/slurm_result/iql_rl_flow_div1_adv_rl_antmaze_tau08-antmaze-large-play-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=0
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --reward_tune antmaze --iql_tau 0.7 --critic_type iql --load_critic_model none --batch_size 1024 --divergence_coef 1.0 --energy_scale 10 --dataset antmaze-large-play-v2 --current_exp_label iql_rl_flow_div1_adv_rl_antmaze_tau07 > ./output/slurm_result/iql_rl_flow_div1_adv_rl_antmaze_tau07-antmaze-large-play-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=0
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --reward_tune antmaze --iql_tau 0.85 --critic_type iql --load_critic_model none --batch_size 1024 --divergence_coef 1.0 --energy_scale 10 --dataset antmaze-large-play-v2 --current_exp_label iql_rl_flow_div1_adv_rl_antmaze_tau085 > ./output/slurm_result/iql_rl_flow_div1_adv_rl_antmaze_tau085-antmaze-large-play-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=1
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --reward_tune antmaze --iql_tau 0.8 --critic_type iql --load_critic_model none --batch_size 1024 --divergence_coef 1.0 --energy_scale 10 --dataset antmaze-large-diverse-v2 --current_exp_label iql_rl_flow_div1_adv_rl_antmaze_tau08 > ./output/slurm_result/iql_rl_flow_div1_adv_rl_antmaze_tau08-antmaze-large-diverse-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=1
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --reward_tune antmaze --iql_tau 0.7 --critic_type iql --load_critic_model none --batch_size 1024 --divergence_coef 1.0 --energy_scale 10 --dataset antmaze-large-diverse-v2 --current_exp_label iql_rl_flow_div1_adv_rl_antmaze_tau07 > ./output/slurm_result/iql_rl_flow_div1_adv_rl_antmaze_tau07-antmaze-large-diverse-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=1
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --reward_tune antmaze --iql_tau 0.85 --critic_type iql --load_critic_model none --batch_size 1024 --divergence_coef 1.0 --energy_scale 10 --dataset antmaze-large-diverse-v2 --current_exp_label iql_rl_flow_div1_adv_rl_antmaze_tau085 > ./output/slurm_result/iql_rl_flow_div1_adv_rl_antmaze_tau085-antmaze-large-diverse-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=2
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --reward_tune antmaze --iql_tau 0.8 --critic_type iql --load_critic_model none --batch_size 1024 --divergence_coef 1.0 --energy_scale 10 --dataset antmaze-large-play-v1 --current_exp_label iql_rl_flow_div1_adv_rl_antmaze_tau08 > ./output/slurm_result/iql_rl_flow_div1_adv_rl_antmaze_tau08-antmaze-large-play-v1.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=2
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --reward_tune antmaze --iql_tau 0.7 --critic_type iql --load_critic_model none --batch_size 1024 --divergence_coef 1.0 --energy_scale 10 --dataset antmaze-large-play-v1 --current_exp_label iql_rl_flow_div1_adv_rl_antmaze_tau07 > ./output/slurm_result/iql_rl_flow_div1_adv_rl_antmaze_tau07-antmaze-large-play-v1.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=2
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --reward_tune antmaze --iql_tau 0.85 --critic_type iql --load_critic_model none --batch_size 1024 --divergence_coef 1.0 --energy_scale 10 --dataset antmaze-large-play-v1 --current_exp_label iql_rl_flow_div1_adv_rl_antmaze_tau085 > ./output/slurm_result/iql_rl_flow_div1_adv_rl_antmaze_tau085-antmaze-large-play-v1.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=3
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --reward_tune antmaze --iql_tau 0.8 --critic_type iql --load_critic_model none --batch_size 1024 --divergence_coef 1.0 --energy_scale 10 --dataset antmaze-large-diverse-v1 --current_exp_label iql_rl_flow_div1_adv_rl_antmaze_tau08 > ./output/slurm_result/iql_rl_flow_div1_adv_rl_antmaze_tau08-antmaze-large-diverse-v1.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=3
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --reward_tune antmaze --iql_tau 0.7 --critic_type iql --load_critic_model none --batch_size 1024 --divergence_coef 1.0 --energy_scale 10 --dataset antmaze-large-diverse-v1 --current_exp_label iql_rl_flow_div1_adv_rl_antmaze_tau07 > ./output/slurm_result/iql_rl_flow_div1_adv_rl_antmaze_tau07-antmaze-large-diverse-v1.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=3
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --reward_tune antmaze --iql_tau 0.85 --critic_type iql --load_critic_model none --batch_size 1024 --divergence_coef 1.0 --energy_scale 10 --dataset antmaze-large-diverse-v1 --current_exp_label iql_rl_flow_div1_adv_rl_antmaze_tau085 > ./output/slurm_result/iql_rl_flow_div1_adv_rl_antmaze_tau085-antmaze-large-diverse-v1.txt 2>&1 &
#sleep 2





#export CUDA_VISIBLE_DEVICES=0
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --reward_tune iql_antmaze --iql_tau 0.8 --critic_type iql --load_critic_model none --batch_size 1024 --divergence_coef 1.0 --energy_scale 10 --dataset antmaze-medium-play-v1 --current_exp_label iql_rl_flow_div1_adv_rl_iql_antmaze > ./output/slurm_result/iql_rl_flow_div1_adv_rl_iql_antmaze-antmaze-medium-play-v1.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=0
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --reward_tune iql_antmaze --iql_tau 0.8 --critic_type iql --load_critic_model none --batch_size 1024 --divergence_coef 1.0 --energy_scale 10 --dataset antmaze-medium-diverse-v1 --current_exp_label iql_rl_flow_div1_adv_rl_iql_antmaze > ./output/slurm_result/iql_rl_flow_div1_adv_rl_iql_antmaze-antmaze-medium-diverse-v1.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=0
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --reward_tune iql_antmaze --iql_tau 0.8 --critic_type iql --load_critic_model none --batch_size 1024 --divergence_coef 1.0 --energy_scale 10 --dataset antmaze-large-play-v1 --current_exp_label iql_rl_flow_div1_adv_rl_iql_antmaze > ./output/slurm_result/iql_rl_flow_div1_adv_rl_iql_antmaze-antmaze-large-play-v1.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=1
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --reward_tune iql_antmaze --iql_tau 0.8 --critic_type iql --load_critic_model none --batch_size 1024 --divergence_coef 1.0 --energy_scale 10 --dataset antmaze-large-diverse-v1 --current_exp_label iql_rl_flow_div1_adv_rl_iql_antmaze > ./output/slurm_result/iql_rl_flow_div1_adv_rl_iql_antmaze-maze2d-large-dense-v1.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=1
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --reward_tune cql_antmaze --iql_tau 0.8 --critic_type iql --load_critic_model none --batch_size 1024 --divergence_coef 1.0 --energy_scale 10 --dataset antmaze-medium-play-v1 --current_exp_label iql_rl_flow_div1_adv_rl_cql_antmaze > ./output/slurm_result/iql_rl_flow_div1_adv_rl_cql_antmaze-antmaze-medium-play-v1.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=1
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --reward_tune cql_antmaze --iql_tau 0.8 --critic_type iql --load_critic_model none --batch_size 1024 --divergence_coef 1.0 --energy_scale 10 --dataset antmaze-medium-diverse-v1 --current_exp_label iql_rl_flow_div1_adv_rl_cql_antmaze > ./output/slurm_result/iql_rl_flow_div1_adv_rl_cql_antmaze-antmaze-medium-diverse-v1.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=2
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --reward_tune cql_antmaze --iql_tau 0.8 --critic_type iql --load_critic_model none --batch_size 1024 --divergence_coef 1.0 --energy_scale 10 --dataset antmaze-large-play-v1 --current_exp_label iql_rl_flow_div1_adv_rl_cql_antmaze > ./output/slurm_result/iql_rl_flow_div1_adv_rl_cql_antmaze-antmaze-large-play-v1.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=2
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --reward_tune cql_antmaze --iql_tau 0.8 --critic_type iql --load_critic_model none --batch_size 1024 --divergence_coef 1.0 --energy_scale 10 --dataset antmaze-large-diverse-v1 --current_exp_label iql_rl_flow_div1_adv_rl_cql_antmaze > ./output/slurm_result/iql_rl_flow_div1_adv_rl_cql_antmaze-maze2d-large-dense-v1.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=2
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --reward_tune cql_antmaze --iql_tau 0.8 --critic_type iql --load_critic_model none --batch_size 1024 --divergence_coef 1.0 --energy_scale 10 --dataset antmaze-medium-play-v0 --current_exp_label iql_rl_flow_div1_adv_rl_cql_antmaze > ./output/slurm_result/iql_rl_flow_div1_adv_rl_cql_antmaze-antmaze-medium-play-v0.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=3
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --reward_tune cql_antmaze --iql_tau 0.8 --critic_type iql --load_critic_model none --batch_size 1024 --divergence_coef 1.0 --energy_scale 10 --dataset antmaze-medium-diverse-v0 --current_exp_label iql_rl_flow_div1_adv_rl_cql_antmaze > ./output/slurm_result/iql_rl_flow_div1_adv_rl_cql_antmaze-antmaze-medium-diverse-v0.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=3
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --reward_tune cql_antmaze --iql_tau 0.8 --critic_type iql --load_critic_model none --batch_size 1024 --divergence_coef 1.0 --energy_scale 10 --dataset antmaze-large-play-v0 --current_exp_label iql_rl_flow_div1_adv_rl_cql_antmaze > ./output/slurm_result/iql_rl_flow_div1_adv_rl_cql_antmaze-antmaze-large-play-v0.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=3
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --reward_tune cql_antmaze --iql_tau 0.8 --critic_type iql --load_critic_model none --batch_size 1024 --divergence_coef 1.0 --energy_scale 10 --dataset antmaze-large-diverse-v0 --current_exp_label iql_rl_flow_div1_adv_rl_cql_antmaze > ./output/slurm_result/iql_rl_flow_div1_adv_rl_cql_antmaze-maze2d-large-dense-v0.txt 2>&1 &
#sleep 2


#export CUDA_VISIBLE_DEVICES=0
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --reward_tune cql_antmaze --iql_tau 0.8 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --divergence_coef 1.0 --energy_scale 10 --dataset antmaze-umaze-v2 --current_exp_label iql_rl_flow_div1_adv_rl > ./output/slurm_result/iql_rl_flow_div1_adv_rl-antmaze-umaze-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=0
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --reward_tune cql_antmaze --iql_tau 0.8 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --divergence_coef 1.0 --energy_scale 10 --dataset antmaze-umaze-diverse-v2 --current_exp_label iql_rl_flow_div1_adv_rl > ./output/slurm_result/iql_rl_flow_div1_adv_rl-antmaze-umaze-diverse-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=0
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --reward_tune cql_antmaze --iql_tau 0.8 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --divergence_coef 1.0 --energy_scale 10 --dataset antmaze-medium-play-v2 --current_exp_label iql_rl_flow_div1_adv_rl > ./output/slurm_result/iql_rl_flow_div1_adv_rl-antmaze-medium-play-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=1
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --reward_tune cql_antmaze --iql_tau 0.8 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --divergence_coef 1.0 --energy_scale 10 --dataset antmaze-medium-diverse-v2 --current_exp_label iql_rl_flow_div1_adv_rl > ./output/slurm_result/iql_rl_flow_div1_adv_rl-antmaze-medium-diverse-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=1
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --reward_tune cql_antmaze --iql_tau 0.8 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --divergence_coef 1.0 --energy_scale 10 --dataset antmaze-large-play-v2 --current_exp_label iql_rl_flow_div1_adv_rl > ./output/slurm_result/iql_rl_flow_div1_adv_rl-antmaze-large-play-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=1
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --reward_tune cql_antmaze --iql_tau 0.8 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --divergence_coef 1.0 --energy_scale 10 --dataset antmaze-large-diverse-v2 --current_exp_label iql_rl_flow_div1_adv_rl > ./output/slurm_result/iql_rl_flow_div1_adv_rl-maze2d-large-dense-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=2
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --reward_tune cql_antmaze --iql_tau 0.8 --critic_type isql --load_critic_model isql_rl_flow --batch_size 1024 --divergence_coef 1.0 --energy_scale 10 --update_energy_end_epoch 200 --update_behavior_end_epoch 100 --update_flow_start_epoch 20 --dataset antmaze-umaze-v2 --current_exp_label isql_rl_flow > ./output/slurm_result/isql_rl_flow-antmaze-umaze-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=2
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --reward_tune cql_antmaze --iql_tau 0.8 --critic_type isql --load_critic_model isql_rl_flow --batch_size 1024 --divergence_coef 1.0 --energy_scale 10 --update_energy_end_epoch 200 --update_behavior_end_epoch 100 --update_flow_start_epoch 20 --dataset antmaze-umaze-diverse-v2 --current_exp_label isql_rl_flow > ./output/slurm_result/isql_rl_flow-antmaze-umaze-diverse-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=2
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --reward_tune cql_antmaze --iql_tau 0.8 --critic_type isql --load_critic_model isql_rl_flow --batch_size 1024 --divergence_coef 1.0 --energy_scale 10 --update_energy_end_epoch 200 --update_behavior_end_epoch 100 --update_flow_start_epoch 20 --dataset antmaze-medium-play-v2 --current_exp_label isql_rl_flow > ./output/slurm_result/isql_rl_flow-antmaze-medium-play-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=3
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --reward_tune cql_antmaze --iql_tau 0.8 --critic_type isql --load_critic_model isql_rl_flow --batch_size 1024 --divergence_coef 1.0 --energy_scale 10 --update_energy_end_epoch 200 --update_behavior_end_epoch 100 --update_flow_start_epoch 20 --dataset antmaze-medium-diverse-v2 --current_exp_label isql_rl_flow > ./output/slurm_result/isql_rl_flow-antmaze-medium-diverse-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=3
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --reward_tune cql_antmaze --iql_tau 0.8 --critic_type isql --load_critic_model isql_rl_flow --batch_size 1024 --divergence_coef 1.0 --energy_scale 10 --update_energy_end_epoch 200 --update_behavior_end_epoch 100 --update_flow_start_epoch 20 --dataset antmaze-large-play-v2 --current_exp_label isql_rl_flow > ./output/slurm_result/isql_rl_flow-antmaze-large-play-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=3
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --reward_tune cql_antmaze --iql_tau 0.8 --critic_type isql --load_critic_model isql_rl_flow --batch_size 1024 --divergence_coef 1.0 --energy_scale 10 --update_energy_end_epoch 200 --update_behavior_end_epoch 100 --update_flow_start_epoch 20 --dataset antmaze-large-diverse-v2 --current_exp_label isql_rl_flow > ./output/slurm_result/isql_rl_flow-maze2d-large-dense-v2.txt 2>&1 &
#sleep 2


#export CUDA_VISIBLE_DEVICES=2
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl4 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --update_energy_end_epoch 0 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --divergence_coef 1 --train_with_normed_data True --CEP_dataset_load_mode True --reward_tune iql_antmaze --dataset antmaze-medium-play-v1 --current_exp_label iql_rl_flow_div5_iql_locomotion > ./output/slurm_result/iql_rl_flow_div5_iql_locomotion-antmaze-medium-play-v1.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=2
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl4 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --update_energy_end_epoch 0 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --divergence_coef 1 --train_with_normed_data True --CEP_dataset_load_mode True --reward_tune iql_antmaze --dataset antmaze-medium-diverse-v1 --current_exp_label iql_rl_flow_div5_iql_locomotion > ./output/slurm_result/iql_rl_flow_div5_iql_locomotion-antmaze-medium-diverse-v1.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=3
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl4 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --update_energy_end_epoch 0 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --divergence_coef 1 --train_with_normed_data True --CEP_dataset_load_mode True --reward_tune iql_antmaze --dataset antmaze-large-play-v1 --current_exp_label iql_rl_flow_div5_iql_locomotion > ./output/slurm_result/iql_rl_flow_div1_iql_locomotion-antmaze-large-play-v1.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=3
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl4 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --update_energy_end_epoch 0 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --divergence_coef 1 --train_with_normed_data True --CEP_dataset_load_mode True --reward_tune iql_antmaze --dataset antmaze-large-diverse-v1 --current_exp_label iql_rl_flow_div5_iql_locomotion > ./output/slurm_result/iql_rl_flow_div5_iql_locomotion-antmaze-large-diverse-v1.txt 2>&1 &
#sleep 2


##  todo Description: train mode is flow_constrained_rl4
#
#export CUDA_VISIBLE_DEVICES=0
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl4 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --update_energy_end_epoch 0 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --divergence_coef 1 --CEP_dataset_load_mode True --reward_tune cql_antmaze --dataset antmaze-medium-play-v0 --current_exp_label iql_rl_flow_div1_cql_antmaze > ./output/slurm_result/iql_rl_flow_div1_cql_antmaze-antmaze-medium-play-v0.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=0
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl4 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --update_energy_end_epoch 0 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --divergence_coef 1 --CEP_dataset_load_mode True --reward_tune cql_antmaze --dataset antmaze-medium-diverse-v0 --current_exp_label iql_rl_flow_div1_cql_antmaze > ./output/slurm_result/iql_rl_flow_div1_cql_antmaze-antmaze-medium-diverse-v0.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=0
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl4 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --update_energy_end_epoch 0 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --divergence_coef 1 --CEP_dataset_load_mode True --reward_tune cql_antmaze --dataset antmaze-large-play-v0 --current_exp_label iql_rl_flow_div1_cql_antmaze > ./output/slurm_result/iql_rl_flow_div1_cql_antmaze-antmaze-large-play-v0.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=1
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl4 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --update_energy_end_epoch 0 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --divergence_coef 1 --CEP_dataset_load_mode True --reward_tune cql_antmaze --dataset antmaze-large-diverse-v0 --current_exp_label iql_rl_flow_div1_cql_antmaze > ./output/slurm_result/iql_rl_flow_div1_cql_antmaze-antmaze-large-diverse-v0.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=1
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl4 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --update_energy_end_epoch 0 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --divergence_coef 0.1 --CEP_dataset_load_mode True --reward_tune iql_antmaze --dataset antmaze-medium-play-v0 --current_exp_label iql_rl_flow_div01_iql_antmaze > ./output/slurm_result/iql_rl_flow_div01_iql_antmaze-antmaze-medium-play-v0.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=1
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl4 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --update_energy_end_epoch 0 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --divergence_coef 0.1 --CEP_dataset_load_mode True --reward_tune iql_antmaze --dataset antmaze-medium-diverse-v0 --current_exp_label iql_rl_flow_div01_iql_antmaze > ./output/slurm_result/iql_rl_flow_div01_iql_antmaze-antmaze-medium-diverse-v0.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=2
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl4 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --update_energy_end_epoch 0 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --divergence_coef 0.1 --CEP_dataset_load_mode True --reward_tune iql_antmaze --dataset antmaze-large-play-v0 --current_exp_label iql_rl_flow_div01_iql_antmaze > ./output/slurm_result/iql_rl_flow_div01_iql_antmaze-antmaze-large-play-v0.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=2
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl4 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --update_energy_end_epoch 0 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --divergence_coef 0.1 --CEP_dataset_load_mode True --reward_tune iql_antmaze --dataset antmaze-large-diverse-v0 --current_exp_label iql_rl_flow_div01_iql_antmaze > ./output/slurm_result/iql_rl_flow_div01_iql_antmaze-antmaze-large-diverse-v0.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=2
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl4 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --update_energy_end_epoch 0 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --divergence_coef 0.01 --CEP_dataset_load_mode True --reward_tune iql_locomotion --dataset antmaze-medium-play-v0 --current_exp_label iql_rl_flow_div001_iql_locomotion > ./output/slurm_result/iql_rl_flow_div001_iql_locomotion-antmaze-medium-play-v0.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=3
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl4 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --update_energy_end_epoch 0 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --divergence_coef 0.01 --CEP_dataset_load_mode True --reward_tune iql_locomotion --dataset antmaze-medium-diverse-v0 --current_exp_label iql_rl_flow_div001_iql_locomotion > ./output/slurm_result/iql_rl_flow_div001_iql_locomotion-antmaze-medium-diverse-v0.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=3
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl4 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --update_energy_end_epoch 0 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --divergence_coef 0.01 --CEP_dataset_load_mode True --reward_tune iql_locomotion --dataset antmaze-large-play-v0 --current_exp_label iql_rl_flow_div001_iql_locomotion > ./output/slurm_result/iql_rl_flow_div001_iql_locomotion-antmaze-large-play-v0.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=3
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl4 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --update_energy_end_epoch 0 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --divergence_coef 0.01 --CEP_dataset_load_mode True --reward_tune iql_locomotion --dataset antmaze-large-diverse-v0 --current_exp_label iql_rl_flow_div001_iql_locomotion > ./output/slurm_result/iql_rl_flow_div001_iql_locomotion-antmaze-large-diverse-v0.txt 2>&1 &
#sleep 2

##  todo Description: train mode is flow_constrained_rl4
#export CUDA_VISIBLE_DEVICES=0
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl4 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --update_energy_end_epoch 0 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --divergence_coef 1.0 --dataset antmaze-umaze-v1 --current_exp_label iql_rl_flow_div1_constrained_rl4 > ./output/slurm_result/iql_rl_flow_div1_constrained_rl4-antmaze-umaze-v1.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=0
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl4 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --update_energy_end_epoch 0 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --divergence_coef 1.0 --dataset antmaze-umaze-diverse-v1 --current_exp_label iql_rl_flow_div1_constrained_rl4 > ./output/slurm_result/iql_rl_flow_div1_constrained_rl4-antmaze-umaze-diverse-v1.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=0
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl4 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --update_energy_end_epoch 0 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --divergence_coef 1.0 --dataset antmaze-medium-play-v1 --current_exp_label iql_rl_flow_div1_constrained_rl4 > ./output/slurm_result/iql_rl_flow_div1_constrained_rl4-antmaze-medium-play-v1.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=1
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl4 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --update_energy_end_epoch 0 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --divergence_coef 1.0 --dataset antmaze-medium-diverse-v1 --current_exp_label iql_rl_flow_div1_constrained_rl4 > ./output/slurm_result/iql_rl_flow_div1_constrained_rl4-antmaze-medium-diverse-v1.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=1
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl4 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --update_energy_end_epoch 0 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --divergence_coef 1.0 --dataset antmaze-large-play-v1 --current_exp_label iql_rl_flow_div1_constrained_rl4 > ./output/slurm_result/iql_rl_flow_div1_constrained_rl4-antmaze-large-play-v1.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=1
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl4 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --update_energy_end_epoch 0 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --divergence_coef 1.0 --dataset antmaze-large-diverse-v1 --current_exp_label iql_rl_flow_div1_constrained_rl4 > ./output/slurm_result/iql_rl_flow_div1_constrained_rl4-antmaze-large-diverse-v1.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=2
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl4 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --update_energy_end_epoch 0 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --divergence_coef 0.05 --dataset antmaze-umaze-v1 --current_exp_label iql_rl_flow_div005_constrained_rl4 > ./output/slurm_result/iql_rl_flow_div005_constrained_rl4-antmaze-umaze-v1.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=2
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl4 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --update_energy_end_epoch 0 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --divergence_coef 0.05 --dataset antmaze-umaze-diverse-v1 --current_exp_label iql_rl_flow_div005_constrained_rl4 > ./output/slurm_result/iql_rl_flow_div005_constrained_rl4-antmaze-umaze-diverse-v1.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=2
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl4 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --update_energy_end_epoch 0 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --divergence_coef 0.05 --dataset antmaze-medium-play-v1 --current_exp_label iql_rl_flow_div005_constrained_rl4 > ./output/slurm_result/iql_rl_flow_div005_constrained_rl4-antmaze-medium-play-v1.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=3
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl4 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --update_energy_end_epoch 0 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --divergence_coef 0.05 --dataset antmaze-medium-diverse-v1 --current_exp_label iql_rl_flow_div005_constrained_rl4 > ./output/slurm_result/iql_rl_flow_div005_constrained_rl4-antmaze-medium-diverse-v1.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=3
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl4 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --update_energy_end_epoch 0 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --divergence_coef 0.05 --dataset antmaze-large-play-v1 --current_exp_label iql_rl_flow_div005_constrained_rl4 > ./output/slurm_result/iql_rl_flow_div005_constrained_rl4-antmaze-large-play-v1.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=3
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl4 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --update_energy_end_epoch 0 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --divergence_coef 0.05 --dataset antmaze-large-diverse-v1 --current_exp_label iql_rl_flow_div005_constrained_rl4 > ./output/slurm_result/iql_rl_flow_div005_constrained_rl4-antmaze-large-diverse-v1.txt 2>&1 &
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




wait
