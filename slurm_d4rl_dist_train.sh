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


#  todo Description: Train the flow model with GRPO, where the critic is conservative IQL model
#export CUDA_VISIBLE_DEVICES=0
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 distributional_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --iql_tau 0.5 --divergence_coef 1.0 --critic_type ciql --load_critic_model ciql_rl_flow --batch_size 1024 --dataset halfcheetah-medium-v2 --current_exp_label ciql_rl_dist_flow_grpo > ./output/slurm_result/ciql_rl_dist_flow_grpo-halfcheetah-medium-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=0
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 distributional_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --iql_tau 0.5 --divergence_coef 1.0 --critic_type ciql --load_critic_model ciql_rl_flow --batch_size 1024 --dataset halfcheetah-medium-expert-v2 --current_exp_label ciql_rl_dist_flow_grpo > ./output/slurm_result/ciql_rl_dist_flow_grpo-halfcheetah-medium-expert-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=0
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 distributional_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --iql_tau 0.5 --divergence_coef 1.0 --critic_type ciql --load_critic_model ciql_rl_flow --batch_size 1024 --dataset halfcheetah-medium-replay-v2 --current_exp_label ciql_rl_dist_flow_grpo > ./output/slurm_result/ciql_rl_dist_flow_grpo-halfcheetah-medium-replay-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES = 1
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 distributional_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --iql_tau 0.5 --divergence_coef 1.0 --critic_type ciql --load_critic_model ciql_rl_flow --batch_size 1024 --dataset halfcheetah-full-replay-v2 --current_exp_label ciql_rl_dist_flow_grpo > ./output/slurm_result/ciql_rl_dist_flow_grpo-halfcheetah-full-replay-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES = 1
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 distributional_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --iql_tau 0.5 --divergence_coef 1.0 --critic_type ciql --load_critic_model ciql_rl_flow --batch_size 1024 --dataset hopper-medium-v2 --current_exp_label ciql_rl_dist_flow_grpo > ./output/slurm_result/ciql_rl_dist_flow_grpo-hopper-medium-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES = 1
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 distributional_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --iql_tau 0.5 --divergence_coef 1.0 --critic_type ciql --load_critic_model ciql_rl_flow --batch_size 1024 --dataset hopper-medium-expert-v2 --current_exp_label ciql_rl_dist_flow_grpo > ./output/slurm_result/ciql_rl_dist_flow_grpo-hopper-medium-expert-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES = 2
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 distributional_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --iql_tau 0.5 --divergence_coef 1.0 --critic_type ciql --load_critic_model ciql_rl_flow --batch_size 1024 --dataset hopper-medium-replay-v2 --current_exp_label ciql_rl_dist_flow_grpo > ./output/slurm_result/ciql_rl_dist_flow_grpo-hopper-medium-replay-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES = 2
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 distributional_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --iql_tau 0.5 --divergence_coef 1.0 --critic_type ciql --load_critic_model ciql_rl_flow --batch_size 1024 --dataset hopper-full-replay-v2 --current_exp_label ciql_rl_dist_flow_grpo > ./output/slurm_result/ciql_rl_dist_flow_grpo-hopper-full-replay-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES = 2
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 distributional_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --iql_tau 0.5 --divergence_coef 1.0 --critic_type ciql --load_critic_model ciql_rl_flow --batch_size 1024 --dataset walker2d-medium-v2 --current_exp_label ciql_rl_dist_flow_grpo > ./output/slurm_result/ciql_rl_dist_flow_grpo-walker2d-medium-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES = 3
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 distributional_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --iql_tau 0.5 --divergence_coef 1.0 --critic_type ciql --load_critic_model ciql_rl_flow --batch_size 1024 --dataset walker2d-medium-expert-v2 --current_exp_label ciql_rl_dist_flow_grpo > ./output/slurm_result/ciql_rl_dist_flow_grpo-walker2d-medium-expert-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES = 3
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 distributional_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --iql_tau 0.5 --divergence_coef 1.0 --critic_type ciql --load_critic_model ciql_rl_flow --batch_size 1024 --dataset walker2d-medium-replay-v2 --current_exp_label ciql_rl_dist_flow_grpo > ./output/slurm_result/ciql_rl_dist_flow_grpo-walker2d-medium-replay-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES = 3
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 distributional_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --iql_tau 0.5 --divergence_coef 1.0 --critic_type ciql --load_critic_model ciql_rl_flow --batch_size 1024 --dataset walker2d-full-replay-v2 --current_exp_label ciql_rl_dist_flow_grpo > ./output/slurm_result/ciql_rl_dist_flow_grpo-walker2d-full-replay-v2.txt 2>&1 &
#sleep 2


#  todo Description: Train the flow model with GRPO, where the critic is IQL model
export CUDA_VISIBLE_DEVICES=0
echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
python3 distributional_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --iql_tau 0.5 --divergence_coef 1.0 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --dataset halfcheetah-medium-v2 --current_exp_label iql_rl_dist_flow_grpo > ./output/slurm_result/iql_rl_dist_flow_grpo-halfcheetah-medium-v2.txt 2>&1 &
sleep 2

export CUDA_VISIBLE_DEVICES=0
echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
python3 distributional_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --iql_tau 0.5 --divergence_coef 1.0 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --dataset halfcheetah-medium-expert-v2 --current_exp_label iql_rl_dist_flow_grpo > ./output/slurm_result/iql_rl_dist_flow_grpo-halfcheetah-medium-expert-v2.txt 2>&1 &
sleep 2

export CUDA_VISIBLE_DEVICES=0
echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
python3 distributional_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --iql_tau 0.5 --divergence_coef 1.0 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --dataset halfcheetah-medium-replay-v2 --current_exp_label iql_rl_dist_flow_grpo > ./output/slurm_result/iql_rl_dist_flow_grpo-halfcheetah-medium-replay-v2.txt 2>&1 &
sleep 2

export CUDA_VISIBLE_DEVICES = 1
echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
python3 distributional_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --iql_tau 0.5 --divergence_coef 1.0 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --dataset halfcheetah-full-replay-v2 --current_exp_label iql_rl_dist_flow_grpo > ./output/slurm_result/iql_rl_dist_flow_grpo-halfcheetah-full-replay-v2.txt 2>&1 &
sleep 2

export CUDA_VISIBLE_DEVICES = 1
echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
python3 distributional_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --iql_tau 0.5 --divergence_coef 1.0 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --dataset hopper-medium-v2 --current_exp_label iql_rl_dist_flow_grpo > ./output/slurm_result/iql_rl_dist_flow_grpo-hopper-medium-v2.txt 2>&1 &
sleep 2

export CUDA_VISIBLE_DEVICES = 1
echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
python3 distributional_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --iql_tau 0.5 --divergence_coef 1.0 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --dataset hopper-medium-expert-v2 --current_exp_label iql_rl_dist_flow_grpo > ./output/slurm_result/iql_rl_dist_flow_grpo-hopper-medium-expert-v2.txt 2>&1 &
sleep 2

export CUDA_VISIBLE_DEVICES = 2
echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
python3 distributional_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --iql_tau 0.5 --divergence_coef 1.0 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --dataset hopper-medium-replay-v2 --current_exp_label iql_rl_dist_flow_grpo > ./output/slurm_result/iql_rl_dist_flow_grpo-hopper-medium-replay-v2.txt 2>&1 &
sleep 2

export CUDA_VISIBLE_DEVICES = 2
echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
python3 distributional_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --iql_tau 0.5 --divergence_coef 1.0 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --dataset hopper-full-replay-v2 --current_exp_label iql_rl_dist_flow_grpo > ./output/slurm_result/iql_rl_dist_flow_grpo-hopper-full-replay-v2.txt 2>&1 &
sleep 2

export CUDA_VISIBLE_DEVICES = 2
echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
python3 distributional_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --iql_tau 0.5 --divergence_coef 1.0 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --dataset walker2d-medium-v2 --current_exp_label iql_rl_dist_flow_grpo > ./output/slurm_result/iql_rl_dist_flow_grpo-walker2d-medium-v2.txt 2>&1 &
sleep 2

export CUDA_VISIBLE_DEVICES = 3
echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
python3 distributional_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --iql_tau 0.5 --divergence_coef 1.0 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --dataset walker2d-medium-expert-v2 --current_exp_label iql_rl_dist_flow_grpo > ./output/slurm_result/iql_rl_dist_flow_grpo-walker2d-medium-expert-v2.txt 2>&1 &
sleep 2

export CUDA_VISIBLE_DEVICES = 3
echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
python3 distributional_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --iql_tau 0.5 --divergence_coef 1.0 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --dataset walker2d-medium-replay-v2 --current_exp_label iql_rl_dist_flow_grpo > ./output/slurm_result/iql_rl_dist_flow_grpo-walker2d-medium-replay-v2.txt 2>&1 &
sleep 2

export CUDA_VISIBLE_DEVICES = 3
echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
python3 distributional_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --iql_tau 0.5 --divergence_coef 1.0 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --dataset walker2d-full-replay-v2 --current_exp_label iql_rl_dist_flow_grpo > ./output/slurm_result/iql_rl_dist_flow_grpo-walker2d-full-replay-v2.txt 2>&1 &
sleep 2

wait