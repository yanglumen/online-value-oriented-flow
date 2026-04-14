#!/bin/bash
#SBATCH --job-name=pretrain
#SBATCH --cpus-per-task=144
#SBATCH --ntasks=1
#SBATCH --partition=gpujl
#SBATCH --mem=350G
#SBATCH --gres=gpu:4

#  --nodelist=node30

export WANDB_API_KEY='e4f12e79df09008c8e830c15787c73c317d66fd3'
source /home/hujifeng/anaconda3/bin/activate diffusion_env

#  todo Description: |task: adroit| Pretrain of the value function and behavior model
export CUDA_VISIBLE_DEVICES=0
echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
python3 pretrain_value_models.py --debug_mode False --wandb_log True --iql_tau 0.5 --critic_type iql --batch_size 1024 --dataset pen-cloned-v1 --current_exp_label iql_rl_flow > ./output/slurm_result/iql_rl_flow-pen-cloned-v1.txt 2>&1 &
sleep 2

export CUDA_VISIBLE_DEVICES=0
echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
python3 pretrain_value_models.py --debug_mode False --wandb_log True --iql_tau 0.5 --critic_type iql --batch_size 1024 --dataset pen-expert-v1 --current_exp_label iql_rl_flow > ./output/slurm_result/iql_rl_flow-pen-expert-v1.txt 2>&1 &
sleep 2

export CUDA_VISIBLE_DEVICES=0
echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
python3 pretrain_value_models.py --debug_mode False --wandb_log True --iql_tau 0.5 --critic_type iql --batch_size 1024 --dataset pen-human-v1 --current_exp_label iql_rl_flow > ./output/slurm_result/iql_rl_flow-pen-human-v1.txt 2>&1 &
sleep 2

export CUDA_VISIBLE_DEVICES=1
echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
python3 pretrain_value_models.py --debug_mode False --wandb_log True --iql_tau 0.5 --critic_type iql --batch_size 1024 --dataset relocate-cloned-v1 --current_exp_label iql_rl_flow > ./output/slurm_result/iql_rl_flow-relocate-cloned-v1.txt 2>&1 &
sleep 2

export CUDA_VISIBLE_DEVICES=1
echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
python3 pretrain_value_models.py --debug_mode False --wandb_log True --iql_tau 0.5 --critic_type iql --batch_size 1024 --dataset relocate-expert-v1 --current_exp_label iql_rl_flow > ./output/slurm_result/iql_rl_flow-relocate-expert-v1.txt 2>&1 &
sleep 2

export CUDA_VISIBLE_DEVICES=1
echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
python3 pretrain_value_models.py --debug_mode False --wandb_log True --iql_tau 0.5 --critic_type iql --batch_size 1024 --dataset relocate-human-v1 --current_exp_label iql_rl_flow > ./output/slurm_result/iql_rl_flow-relocate-human-v1.txt 2>&1 &
sleep 2

export CUDA_VISIBLE_DEVICES=2
echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
python3 pretrain_value_models.py --debug_mode False --wandb_log True --iql_tau 0.5 --critic_type iql --batch_size 1024 --dataset hammer-cloned-v1 --current_exp_label iql_rl_flow > ./output/slurm_result/iql_rl_flow-hammer-cloned-v1.txt 2>&1 &
sleep 2

export CUDA_VISIBLE_DEVICES=2
echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
python3 pretrain_value_models.py --debug_mode False --wandb_log True --iql_tau 0.5 --critic_type iql --batch_size 1024 --dataset hammer-expert-v1 --current_exp_label iql_rl_flow > ./output/slurm_result/iql_rl_flow-hammer-expert-v1.txt 2>&1 &
sleep 2

export CUDA_VISIBLE_DEVICES=2
echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
python3 pretrain_value_models.py --debug_mode False --wandb_log True --iql_tau 0.5 --critic_type iql --batch_size 1024 --dataset hammer-human-v1 --current_exp_label iql_rl_flow > ./output/slurm_result/iql_rl_flow-hammer-human-v1.txt 2>&1 &
sleep 2

export CUDA_VISIBLE_DEVICES=3
echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
python3 pretrain_value_models.py --debug_mode False --wandb_log True --iql_tau 0.5 --critic_type iql --batch_size 1024 --dataset door-cloned-v1 --current_exp_label iql_rl_flow > ./output/slurm_result/iql_rl_flow-door-cloned-v1.txt 2>&1 &
sleep 2

export CUDA_VISIBLE_DEVICES=3
echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
python3 pretrain_value_models.py --debug_mode False --wandb_log True --iql_tau 0.5 --critic_type iql --batch_size 1024 --dataset door-expert-v1 --current_exp_label iql_rl_flow > ./output/slurm_result/iql_rl_flow-door-expert-v1.txt 2>&1 &
sleep 2

export CUDA_VISIBLE_DEVICES=3
echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
python3 pretrain_value_models.py --debug_mode False --wandb_log True --iql_tau 0.5 --critic_type iql --batch_size 1024 --dataset door-human-v1 --current_exp_label iql_rl_flow > ./output/slurm_result/iql_rl_flow-door-human-v1.txt 2>&1 &
sleep 2

##  todo Description: |task: antmaze and maze2d| Pretrain of the value function and behavior model
#export CUDA_VISIBLE_DEVICES=0
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 pretrain_value_models.py --debug_mode False --wandb_log True --iql_tau 0.5 --critic_type iql --batch_size 1024 --dataset antmaze-umaze-v1 --current_exp_label iql_rl_flow > ./output/slurm_result/iql_rl_flow-antmaze-umaze-v1.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=0
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 pretrain_value_models.py --debug_mode False --wandb_log True --iql_tau 0.5 --critic_type iql --batch_size 1024 --dataset antmaze-umaze-diverse-v1 --current_exp_label iql_rl_flow > ./output/slurm_result/iql_rl_flow-antmaze-umaze-diverse-v1.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=0
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 pretrain_value_models.py --debug_mode False --wandb_log True --iql_tau 0.5 --critic_type iql --batch_size 1024 --dataset antmaze-medium-play-v1 --current_exp_label iql_rl_flow > ./output/slurm_result/iql_rl_flow-antmaze-medium-play-v1.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=0
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 pretrain_value_models.py --debug_mode False --wandb_log True --iql_tau 0.5 --critic_type iql --batch_size 1024 --dataset antmaze-medium-diverse-v1 --current_exp_label iql_rl_flow > ./output/slurm_result/iql_rl_flow-antmaze-medium-diverse-v1.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=1
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 pretrain_value_models.py --debug_mode False --wandb_log True --iql_tau 0.5 --critic_type iql --batch_size 1024 --dataset antmaze-large-play-v1 --current_exp_label iql_rl_flow > ./output/slurm_result/iql_rl_flow-antmaze-large-play-v1.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=1
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 pretrain_value_models.py --debug_mode False --wandb_log True --iql_tau 0.5 --critic_type iql --batch_size 1024 --dataset antmaze-large-diverse-v1 --current_exp_label iql_rl_flow > ./output/slurm_result/iql_rl_flow-antmaze-large-diverse-v1.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=1
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 pretrain_value_models.py --debug_mode False --wandb_log True --iql_tau 0.5 --critic_type iql --batch_size 1024 --dataset maze2d-umaze-v1 --current_exp_label iql_rl_flow > ./output/slurm_result/iql_rl_flow-maze2d-umaze-v1.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=1
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 pretrain_value_models.py --debug_mode False --wandb_log True --iql_tau 0.5 --critic_type iql --batch_size 1024 --dataset maze2d-medium-v1 --current_exp_label iql_rl_flow > ./output/slurm_result/iql_rl_flow-maze2d-medium-v1.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=1
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 pretrain_value_models.py --debug_mode False --wandb_log True --iql_tau 0.5 --critic_type iql --batch_size 1024 --dataset maze2d-large-v1 --current_exp_label iql_rl_flow > ./output/slurm_result/iql_rl_flow-maze2d-large-v1.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=2
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 pretrain_value_models.py --debug_mode False --wandb_log True --iql_tau 0.5 --critic_type iql --batch_size 1024 --dataset maze2d-umaze-dense-v1 --current_exp_label iql_rl_flow > ./output/slurm_result/iql_rl_flow-maze2d-umaze-dense-v1.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=2
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 pretrain_value_models.py --debug_mode False --wandb_log True --iql_tau 0.5 --critic_type iql --batch_size 1024 --dataset maze2d-medium-dense-v1 --current_exp_label iql_rl_flow > ./output/slurm_result/iql_rl_flow-maze2d-medium-dense-v1.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=2
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 pretrain_value_models.py --debug_mode False --wandb_log True --iql_tau 0.5 --critic_type iql --batch_size 1024 --dataset maze2d-large-dense-v1 --current_exp_label iql_rl_flow > ./output/slurm_result/iql_rl_flow-maze2d-large-dense-v1.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=2
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 pretrain_value_models.py --debug_mode False --wandb_log True --iql_tau 0.5 --critic_type iql --batch_size 1024 --dataset maze2d-umaze-v0 --current_exp_label iql_rl_flow > ./output/slurm_result/iql_rl_flow-maze2d-umaze-v0.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=2
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 pretrain_value_models.py --debug_mode False --wandb_log True --iql_tau 0.5 --critic_type iql --batch_size 1024 --dataset maze2d-medium-v0 --current_exp_label iql_rl_flow > ./output/slurm_result/iql_rl_flow-maze2d-medium-v0.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=3
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 pretrain_value_models.py --debug_mode False --wandb_log True --iql_tau 0.5 --critic_type iql --batch_size 1024 --dataset maze2d-large-v0 --current_exp_label iql_rl_flow > ./output/slurm_result/iql_rl_flow-maze2d-large-v0.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=3
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 pretrain_value_models.py --debug_mode False --wandb_log True --iql_tau 0.5 --critic_type iql --batch_size 1024 --dataset maze2d-umaze-dense-v0 --current_exp_label iql_rl_flow > ./output/slurm_result/iql_rl_flow-maze2d-umaze-dense-v0.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=3
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 pretrain_value_models.py --debug_mode False --wandb_log True --iql_tau 0.5 --critic_type iql --batch_size 1024 --dataset maze2d-medium-dense-v0 --current_exp_label iql_rl_flow > ./output/slurm_result/iql_rl_flow-maze2d-medium-dense-v0.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=3
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 pretrain_value_models.py --debug_mode False --wandb_log True --iql_tau 0.5 --critic_type iql --batch_size 1024 --dataset maze2d-large-dense-v0 --current_exp_label iql_rl_flow > ./output/slurm_result/iql_rl_flow-maze2d-large-dense-v0.txt 2>&1 &
#sleep 2

##  todo Description: Pretrain of the value function and behavior model
#export CUDA_VISIBLE_DEVICES=0
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 pretrain_value_models.py --debug_mode False --wandb_log True --iql_tau 0.5 --critic_type ciql --batch_size 1024 --dataset halfcheetah-medium-v2 --current_exp_label ciql_rl_flow > ./output/slurm_result/ciql_rl_flow-halfcheetah-medium-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=0
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 pretrain_value_models.py --debug_mode False --wandb_log True --iql_tau 0.5 --critic_type ciql --batch_size 1024 --dataset halfcheetah-medium-expert-v2 --current_exp_label ciql_rl_flow > ./output/slurm_result/ciql_rl_flow-halfcheetah-medium-expert-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=0
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 pretrain_value_models.py --debug_mode False --wandb_log True --iql_tau 0.5 --critic_type ciql --batch_size 1024 --dataset halfcheetah-medium-replay-v2 --current_exp_label ciql_rl_flow > ./output/slurm_result/ciql_rl_flow-halfcheetah-medium-replay-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES = 1
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 pretrain_value_models.py --debug_mode False --wandb_log True --iql_tau 0.5 --critic_type ciql --batch_size 1024 --dataset halfcheetah-full-replay-v2 --current_exp_label ciql_rl_flow > ./output/slurm_result/ciql_rl_flow-halfcheetah-full-replay-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES = 1
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 pretrain_value_models.py --debug_mode False --wandb_log True --iql_tau 0.5 --critic_type ciql --batch_size 1024 --dataset hopper-medium-v2 --current_exp_label ciql_rl_flow > ./output/slurm_result/ciql_rl_flow-hopper-medium-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES = 1
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 pretrain_value_models.py --debug_mode False --wandb_log True --iql_tau 0.5 --critic_type ciql --batch_size 1024 --dataset hopper-medium-expert-v2 --current_exp_label ciql_rl_flow > ./output/slurm_result/ciql_rl_flow-hopper-medium-expert-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES = 2
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 pretrain_value_models.py --debug_mode False --wandb_log True --iql_tau 0.5 --critic_type ciql --batch_size 1024 --dataset hopper-medium-replay-v2 --current_exp_label ciql_rl_flow > ./output/slurm_result/ciql_rl_flow-hopper-medium-replay-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES = 2
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 pretrain_value_models.py --debug_mode False --wandb_log True --iql_tau 0.5 --critic_type ciql --batch_size 1024 --dataset hopper-full-replay-v2 --current_exp_label ciql_rl_flow > ./output/slurm_result/ciql_rl_flow-hopper-full-replay-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES = 2
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 pretrain_value_models.py --debug_mode False --wandb_log True --iql_tau 0.5 --critic_type ciql --batch_size 1024 --dataset walker2d-medium-v2 --current_exp_label ciql_rl_flow > ./output/slurm_result/ciql_rl_flow-walker2d-medium-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES = 3
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 pretrain_value_models.py --debug_mode False --wandb_log True --iql_tau 0.5 --critic_type ciql --batch_size 1024 --dataset walker2d-medium-expert-v2 --current_exp_label ciql_rl_flow > ./output/slurm_result/ciql_rl_flow-walker2d-medium-expert-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES = 3
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 pretrain_value_models.py --debug_mode False --wandb_log True --iql_tau 0.5 --critic_type ciql --batch_size 1024 --dataset walker2d-medium-replay-v2 --current_exp_label ciql_rl_flow > ./output/slurm_result/ciql_rl_flow-walker2d-medium-replay-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES = 3
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 pretrain_value_models.py --debug_mode False --wandb_log True --iql_tau 0.5 --critic_type ciql --batch_size 1024 --dataset walker2d-full-replay-v2 --current_exp_label ciql_rl_flow > ./output/slurm_result/ciql_rl_flow-walker2d-full-replay-v2.txt 2>&1 &
#sleep 2

wait