#!/bin/bash
#SBATCH --job-name=pretrain
#SBATCH --cpus-per-task=72
#SBATCH --ntasks=1
#SBATCH --partition=gpujl
#SBATCH --mem=350G
#SBATCH --gres=gpu:4

#  --nodelist=node30

export WANDB_API_KEY='e4f12e79df09008c8e830c15787c73c317d66fd3'
source /home/hujifeng/anaconda3/bin/activate diffusion_env


#  todo Description: train mode is flow_constrained_rl4
export CUDA_VISIBLE_DEVICES=0
echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl4 --divergence_coef 0.1 --critic_type iql --load_critic_model iql_rl_flow --energy_scale 1 --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --dataset hopper-medium-v2 --current_exp_label divergence_coef_01_fcrl4 > ./output/slurm_result/divergence_coef_01_fcrl4-hopper-medium-v2.txt 2>&1 &
sleep 2

export CUDA_VISIBLE_DEVICES=0
echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl4 --divergence_coef 0.3 --critic_type iql --load_critic_model iql_rl_flow --energy_scale 1 --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --dataset hopper-medium-v2 --current_exp_label divergence_coef_03_fcrl4 > ./output/slurm_result/divergence_coef_03_fcrl4-hopper-medium-v2.txt 2>&1 &
sleep 2

export CUDA_VISIBLE_DEVICES=0
echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl4 --divergence_coef 0.8 --critic_type iql --load_critic_model iql_rl_flow --energy_scale 1 --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --dataset hopper-medium-v2 --current_exp_label divergence_coef_08_fcrl4 > ./output/slurm_result/divergence_coef_08_fcrl4-hopper-medium-v2.txt 2>&1 &
sleep 2

export CUDA_VISIBLE_DEVICES=1
echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl4 --divergence_coef 2.0 --critic_type iql --load_critic_model iql_rl_flow --energy_scale 1 --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --dataset hopper-medium-v2 --current_exp_label divergence_coef_2_fcrl4 > ./output/slurm_result/divergence_coef_2_fcrl4-hopper-medium-v2.txt 2>&1 &
sleep 2

export CUDA_VISIBLE_DEVICES=1
echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl4 --divergence_coef 5.0 --critic_type iql --load_critic_model iql_rl_flow --energy_scale 1 --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --dataset hopper-medium-v2 --current_exp_label divergence_coef_5_fcrl4 > ./output/slurm_result/divergence_coef_5_fcrl4-hopper-medium-v2.txt 2>&1 &
sleep 2

export CUDA_VISIBLE_DEVICES=1
echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl4 --divergence_coef 10.0 --critic_type iql --load_critic_model iql_rl_flow --energy_scale 1 --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --dataset hopper-medium-v2 --current_exp_label divergence_coef_10_fcrl4 > ./output/slurm_result/divergence_coef_10_fcrl4-hopper-medium-v2.txt 2>&1 &
sleep 2


export CUDA_VISIBLE_DEVICES=2
echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl4 --divergence_coef 0.1 --critic_type iql --load_critic_model iql_rl_flow --energy_scale 1 --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --dataset hopper-medium-replay-v2 --current_exp_label divergence_coef_01_fcrl4 > ./output/slurm_result/divergence_coef_01_fcrl4-hopper-medium-replay-v2.txt 2>&1 &
sleep 2

export CUDA_VISIBLE_DEVICES=2
echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl4 --divergence_coef 0.3 --critic_type iql --load_critic_model iql_rl_flow --energy_scale 1 --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --dataset hopper-medium-replay-v2 --current_exp_label divergence_coef_03_fcrl4 > ./output/slurm_result/divergence_coef_03_fcrl4-hopper-medium-replay-v2.txt 2>&1 &
sleep 2

export CUDA_VISIBLE_DEVICES=2
echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl4 --divergence_coef 0.8 --critic_type iql --load_critic_model iql_rl_flow --energy_scale 1 --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --dataset hopper-medium-replay-v2 --current_exp_label divergence_coef_08_fcrl4 > ./output/slurm_result/divergence_coef_08_fcrl4-hopper-medium-replay-v2.txt 2>&1 &
sleep 2

export CUDA_VISIBLE_DEVICES=3
echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl4 --divergence_coef 2.0 --critic_type iql --load_critic_model iql_rl_flow --energy_scale 1 --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --dataset hopper-medium-replay-v2 --current_exp_label divergence_coef_2_fcrl4 > ./output/slurm_result/divergence_coef_2_fcrl4-hopper-medium-replay-v2.txt 2>&1 &
sleep 2

export CUDA_VISIBLE_DEVICES=3
echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl4 --divergence_coef 5.0 --critic_type iql --load_critic_model iql_rl_flow --energy_scale 1 --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --dataset hopper-medium-replay-v2 --current_exp_label divergence_coef_5_fcrl4 > ./output/slurm_result/divergence_coef_5_fcrl4-hopper-medium-replay-v2.txt 2>&1 &
sleep 2

export CUDA_VISIBLE_DEVICES=3
echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl4 --divergence_coef 10.0 --critic_type iql --load_critic_model iql_rl_flow --energy_scale 1 --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --dataset hopper-medium-replay-v2 --current_exp_label divergence_coef_10_fcrl4 > ./output/slurm_result/divergence_coef_10_fcrl4-hopper-medium-replay-v2.txt 2>&1 &
sleep 2


##  todo Description: train mode is flow_constrained_rl4
#export CUDA_VISIBLE_DEVICES=0
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl4 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --energy_scale 0.2 --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --dataset hopper-medium-v2 --current_exp_label small_energy_scale_fcrl4 > ./output/slurm_result/small_energy_scale_fcrl4-hopper-medium-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=0
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl4 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --energy_scale 1 --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --large_flow True --flow_step 30 --dataset hopper-medium-v2 --current_exp_label large_flow_large_flow_step_fcrl4 > ./output/slurm_result/large_flow_large_flow_step_fcrl4-hopper-medium-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=0
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl4 --iql_tau 0.9 --critic_type iql --load_critic_model iql_rl_flow --energy_scale 1 --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --dataset hopper-medium-v2 --current_exp_label large_iql_tau_fcrl4 > ./output/slurm_result/large_iql_tau_fcrl4-hopper-medium-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=1
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl4 --iql_tau 0.5 --critic_type ciql --load_critic_model ciql_rl_flow --energy_scale 0.2 --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --dataset hopper-medium-v2 --current_exp_label small_energy_scale_ciql_fcrl4 > ./output/slurm_result/small_energy_scale_ciql_fcrl4-hopper-medium-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=1
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl4 --iql_tau 0.5 --critic_type ciql --load_critic_model ciql_rl_flow --energy_scale 1 --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --large_flow True --flow_step 30 --dataset hopper-medium-v2 --current_exp_label large_flow_large_flow_step_ciql_fcrl4 > ./output/slurm_result/large_flow_large_flow_step_ciql_fcrl4-hopper-medium-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=1
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl4 --iql_tau 0.9 --critic_type ciql --load_critic_model ciql_rl_flow --energy_scale 1 --batch_size 1024 --flow_constrained_rl4_multiple_actions 20 --dataset hopper-medium-v2 --current_exp_label large_iql_tau_ciql_fcrl4 > ./output/slurm_result/large_iql_tau_ciql_fcrl4-hopper-medium-v2.txt 2>&1 &
#sleep 2
#
#
#export CUDA_VISIBLE_DEVICES=2
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl4 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --energy_scale 1 --batch_size 1024 --flow_constrained_rl4_multiple_actions 10 --large_flow True --dataset walker2d-medium-replay-v2 --current_exp_label large_flow_fcrl4 > ./output/slurm_result/large_flow_fcrl4-walker2d-medium-replay-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=2
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl4 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --energy_scale 1 --batch_size 1024 --flow_constrained_rl4_multiple_actions 10 --flow_step 20 --dataset walker2d-medium-replay-v2 --current_exp_label large_flow_step_fcrl4 > ./output/slurm_result/large_flow_step_fcrl4-walker2d-medium-replay-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=2
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl4 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --energy_scale 1 --batch_size 1024 --flow_constrained_rl4_multiple_actions 30 --dataset walker2d-medium-replay-v2 --current_exp_label large_sampled_actions_fcrl4 > ./output/slurm_result/large_sampled_actions_fcrl4-walker2d-medium-replay-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=3
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl4 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --energy_scale 10 --batch_size 1024 --flow_constrained_rl4_multiple_actions 10 --dataset walker2d-medium-replay-v2 --current_exp_label large_energy_scale_fcrl4 > ./output/slurm_result/large_energy_scale_fcrl4-walker2d-medium-replay-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=3
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl4 --iql_tau 0.8 --critic_type iql --load_critic_model iql_rl_flow --energy_scale 1 --batch_size 1024 --flow_constrained_rl4_multiple_actions 10 --dataset walker2d-medium-replay-v2 --current_exp_label large_iql_tau_fcrl4 > ./output/slurm_result/large_iql_tau_fcrl4-walker2d-medium-replay-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=3
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl4 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --energy_scale 10 --batch_size 1024 --flow_constrained_rl4_multiple_actions 10 --large_flow_V True --dataset walker2d-medium-replay-v2 --current_exp_label large_flow_V_fcrl4 > ./output/slurm_result/large_flow_V_fcrl4-walker2d-medium-replay-v2.txt 2>&1 &
#sleep 2

##  todo Description: train mode is flow_constrained_rl5
#export CUDA_VISIBLE_DEVICES=0
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl5 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --energy_scale 10 --batch_size 1024 --flow_constrained_rl5_multiple_actions 10 --large_flow True --dataset halfcheetah-medium-expert-v2 --current_exp_label large_flow_fcrl5 > ./output/slurm_result/large_flow_fcrl5-halfcheetah-medium-expert-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=0
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl5 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --energy_scale 10 --batch_size 1024 --flow_constrained_rl5_multiple_actions 10 --flow_step 20 --dataset halfcheetah-medium-expert-v2 --current_exp_label large_flow_step_fcrl5 > ./output/slurm_result/large_flow_step_fcrl5-halfcheetah-medium-expert-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=0
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl5 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --energy_scale 10 --batch_size 1024 --flow_constrained_rl5_multiple_actions 30 --dataset halfcheetah-medium-expert-v2 --current_exp_label large_sampled_actions_fcrl5 > ./output/slurm_result/large_sampled_actions_fcrl5-halfcheetah-medium-expert-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=1
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl5 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --energy_scale 10 --batch_size 2048 --flow_constrained_rl5_multiple_actions 10 --dataset halfcheetah-medium-expert-v2 --current_exp_label large_batch_size_fcrl5 > ./output/slurm_result/large_batch_size_fcrl5-halfcheetah-medium-expert-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=1
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl5 --iql_tau 0.8 --critic_type iql --load_critic_model iql_rl_flow --energy_scale 10 --batch_size 1024 --flow_constrained_rl5_multiple_actions 10 --dataset halfcheetah-medium-expert-v2 --current_exp_label large_iql_tau_fcrl5 > ./output/slurm_result/large_iql_tau_fcrl5-halfcheetah-medium-expert-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=1
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl5 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --energy_scale 10 --batch_size 1024 --flow_constrained_rl5_multiple_actions 10 --large_flow_V True --dataset halfcheetah-medium-expert-v2 --current_exp_label large_flow_V_fcrl5 > ./output/slurm_result/large_flow_V_fcrl5-halfcheetah-medium-expert-v2.txt 2>&1 &
#sleep 2
#
#
#
#
#
#export CUDA_VISIBLE_DEVICES=2
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl5 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --energy_scale 10 --batch_size 1024 --flow_constrained_rl5_multiple_actions 10 --large_flow True --dataset hopper-medium-expert-v2 --current_exp_label large_flow_fcrl5 > ./output/slurm_result/large_flow_fcrl5-hopper-medium-expert-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=2
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl5 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --energy_scale 10 --batch_size 1024 --flow_constrained_rl5_multiple_actions 10 --flow_step 20 --dataset hopper-medium-expert-v2 --current_exp_label large_flow_step_fcrl5 > ./output/slurm_result/large_flow_step_fcrl5-hopper-medium-expert-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=2
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl5 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --energy_scale 10 --batch_size 1024 --flow_constrained_rl5_multiple_actions 30 --dataset hopper-medium-expert-v2 --current_exp_label large_sampled_actions_fcrl5 > ./output/slurm_result/large_sampled_actions_fcrl5-hopper-medium-expert-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=3
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl5 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --energy_scale 10 --batch_size 2048 --flow_constrained_rl5_multiple_actions 10 --dataset hopper-medium-expert-v2 --current_exp_label large_batch_size_fcrl5 > ./output/slurm_result/large_batch_size_fcrl5-hopper-medium-expert-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=3
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl5 --iql_tau 0.8 --critic_type iql --load_critic_model iql_rl_flow --energy_scale 10 --batch_size 1024 --flow_constrained_rl5_multiple_actions 10 --dataset hopper-medium-expert-v2 --current_exp_label large_iql_tau_fcrl5 > ./output/slurm_result/large_iql_tau_fcrl5-hopper-medium-expert-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=3
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode flow_constrained_rl5 --iql_tau 0.5 --critic_type iql --load_critic_model iql_rl_flow --energy_scale 10 --batch_size 1024 --flow_constrained_rl5_multiple_actions 10 --large_flow_V True --dataset hopper-medium-expert-v2 --current_exp_label large_flow_V_fcrl5 > ./output/slurm_result/large_flow_V_fcrl5-hopper-medium-expert-v2.txt 2>&1 &
#sleep 2

wait