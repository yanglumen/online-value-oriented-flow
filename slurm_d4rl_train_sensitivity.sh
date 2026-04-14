#!/bin/bash
#SBATCH --job-name=pretrain
#SBATCH --cpus-per-task=144
#SBATCH --ntasks=1
#SBATCH --partition=gpujl
#SBATCH --mem=450G
#SBATCH --gres=gpu:4

#  --nodelist=node30

export WANDB_API_KEY='e4f12e79df09008c8e830c15787c73c317d66fd3'
source /home/hujifeng/anaconda3/bin/activate diffusion_env


export CUDA_VISIBLE_DEVICES=0
echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --iql_tau 0.5 --divergence_coef 5.0 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --energy_scale 10 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --dataset hopper-medium-replay-v2 --current_exp_label iql_adv_rl_flow_sta5_escale_10 > ./output/slurm_result/iql_adv_rl_flow_sta5_escale_10-hopper-medium-replay-v2.txt 2>&1 &
sleep 2

export CUDA_VISIBLE_DEVICES=0
echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --iql_tau 0.5 --divergence_coef 1.0 --flow_step 10 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --energy_scale 200 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --dataset halfcheetah-medium-expert-v2 --current_exp_label iql_adv_rl_flow_sta1_escale_200_fstep_10 > ./output/slurm_result/iql_adv_rl_flow_sta1_escale_200_fstep_10-halfcheetah-medium-expert-v2.txt 2>&1 &
sleep 2

export CUDA_VISIBLE_DEVICES=1
echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --iql_tau 0.5 --divergence_coef 2.0 --flow_step 10 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --energy_scale 200 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --dataset halfcheetah-medium-expert-v2 --current_exp_label iql_adv_rl_flow_sta2_escale_200_fstep_10 > ./output/slurm_result/iql_adv_rl_flow_sta2_escale_200_fstep_10-halfcheetah-medium-expert-v2.txt 2>&1 &
sleep 2

export CUDA_VISIBLE_DEVICES=1
echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --iql_tau 0.5 --divergence_coef 3.0 --flow_step 10 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --energy_scale 200 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --dataset halfcheetah-medium-expert-v2 --current_exp_label iql_adv_rl_flow_sta3_escale_200_fstep_10 > ./output/slurm_result/iql_adv_rl_flow_sta3_escale_200_fstep_10-halfcheetah-medium-expert-v2.txt 2>&1 &
sleep 2

export CUDA_VISIBLE_DEVICES=2
echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --iql_tau 0.5 --divergence_coef 5.0 --flow_step 10 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --energy_scale 200 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --dataset halfcheetah-medium-expert-v2 --current_exp_label iql_adv_rl_flow_sta5_escale_200_fstep_10 > ./output/slurm_result/iql_adv_rl_flow_sta5_escale_200_fstep_10-halfcheetah-medium-expert-v2.txt 2>&1 &
sleep 2

export CUDA_VISIBLE_DEVICES=3
echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --iql_tau 0.5 --divergence_coef 1.0 --flow_step 10 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --energy_scale 200 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --dataset halfcheetah-medium-expert-v2 --current_exp_label iql_adv_rl_flow_sta1_escale_200_fstep_20 > ./output/slurm_result/iql_adv_rl_flow_sta1_escale_200_fstep_20-halfcheetah-medium-expert-v2.txt 2>&1 &
sleep 2



#export CUDA_VISIBLE_DEVICES=0
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --iql_tau 0.5 --divergence_coef 5.0 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --energy_scale 10 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --dataset walker2d-medium-replay-v2 --current_exp_label iql_adv_rl_flow_sta5_escale_10 > ./output/slurm_result/iql_adv_rl_flow_sta5_escale_10-walker2d-medium-replay-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=0
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --iql_tau 0.5 --divergence_coef 5.0 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --energy_scale 10 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --dataset hopper-medium-replay-v2 --current_exp_label iql_adv_rl_flow_sta5_escale_10 > ./output/slurm_result/iql_adv_rl_flow_sta5_escale_10-hopper-medium-replay-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=0
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --iql_tau 0.5 --divergence_coef 1.0 --flow_step 20 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --energy_scale 10 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --dataset halfcheetah-medium-v2 --current_exp_label iql_adv_rl_flow_sta1_escale_10_fstep_20 > ./output/slurm_result/iql_adv_rl_flow_sta1_escale_10_fstep_20-halfcheetah-medium-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=1
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --iql_tau 0.5 --divergence_coef 1.0 --flow_step 20 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --energy_scale 10 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --dataset halfcheetah-medium-expert-v2 --current_exp_label iql_adv_rl_flow_sta1_escale_10_fstep_20 > ./output/slurm_result/iql_adv_rl_flow_sta1_escale_10_fstep_20-halfcheetah-medium-expert-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=1
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --iql_tau 0.5 --divergence_coef 1.0 --flow_step 20 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --energy_scale 10 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --dataset halfcheetah-medium-replay-v2 --current_exp_label iql_adv_rl_flow_sta1_escale_10_fstep_20 > ./output/slurm_result/iql_adv_rl_flow_sta1_escale_10_fstep_20-halfcheetah-medium-replay-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=1
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --iql_tau 0.5 --divergence_coef 1.0 --flow_step 20 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --energy_scale 10 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --dataset hopper-medium-v2 --current_exp_label iql_adv_rl_flow_sta1_escale_10_fstep_20 > ./output/slurm_result/iql_adv_rl_flow_sta1_escale_10_fstep_20-hopper-medium-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=2
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --iql_tau 0.5 --divergence_coef 1.0 --flow_step 20 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --energy_scale 10 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --dataset hopper-medium-expert-v2 --current_exp_label iql_adv_rl_flow_sta1_escale_10_fstep_20 > ./output/slurm_result/iql_adv_rl_flow_sta1_escale_10_fstep_20-hopper-medium-expert-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=2
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --iql_tau 0.5 --divergence_coef 1.0 --flow_step 20 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --energy_scale 10 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --dataset hopper-medium-replay-v2 --current_exp_label iql_adv_rl_flow_sta1_escale_10_fstep_20 > ./output/slurm_result/iql_adv_rl_flow_sta1_escale_10_fstep_20-hopper-medium-replay-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=2
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --iql_tau 0.5 --divergence_coef 1.0 --flow_step 20 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --energy_scale 10 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --dataset walker2d-medium-v2 --current_exp_label iql_adv_rl_flow_sta1_escale_10_fstep_20 > ./output/slurm_result/iql_adv_rl_flow_sta1_escale_10_fstep_20-walker2d-medium-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=3
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --iql_tau 0.5 --divergence_coef 1.0 --flow_step 20 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --energy_scale 10 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --dataset walker2d-medium-expert-v2 --current_exp_label iql_adv_rl_flow_sta1_escale_10_fstep_20 > ./output/slurm_result/iql_adv_rl_flow_sta1_escale_10_fstep_20-walker2d-medium-expert-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=3
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --iql_tau 0.5 --divergence_coef 1.0 --flow_step 20 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --energy_scale 10 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --dataset walker2d-medium-replay-v2 --current_exp_label iql_adv_rl_flow_sta1_escale_10_fstep_20 > ./output/slurm_result/iql_adv_rl_flow_sta1_escale_10_fstep_20-walker2d-medium-replay-v2.txt 2>&1 &
#sleep 2




#export CUDA_VISIBLE_DEVICES=0
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --iql_tau 0.5 --divergence_coef 2.0 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --energy_scale 10 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --dataset halfcheetah-medium-expert-v2 --current_exp_label iql_adv_rl_flow_sta2_escale_10 > ./output/slurm_result/iql_adv_rl_flow_sta2_escale_10-halfcheetah-medium-expert-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=0
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --iql_tau 0.5 --divergence_coef 3.0 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --energy_scale 10 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --dataset hopper-medium-expert-v2 --current_exp_label iql_adv_rl_flow_sta3_escale_10 > ./output/slurm_result/iql_adv_rl_flow_sta3_escale_10-hopper-medium-expert-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=0
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --iql_tau 0.5 --divergence_coef 2.0 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --energy_scale 10 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --dataset walker2d-medium-expert-v2 --current_exp_label iql_adv_rl_flow_sta3_escale_10 > ./output/slurm_result/iql_adv_rl_flow_sta3_escale_10-walker2d-medium-expert-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=1
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --iql_tau 0.5 --divergence_coef 3.0 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --energy_scale 10 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --dataset halfcheetah-medium-expert-v2 --current_exp_label iql_adv_rl_flow_sta3_escale_10 > ./output/slurm_result/iql_adv_rl_flow_sta3_escale_10-halfcheetah-medium-expert-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=1
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --iql_tau 0.5 --divergence_coef 5.0 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --energy_scale 10 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --dataset halfcheetah-medium-v2 --current_exp_label iql_adv_rl_flow_sta5_escale_10 > ./output/slurm_result/iql_adv_rl_flow_sta5_escale_10-halfcheetah-medium-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=1
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --iql_tau 0.5 --divergence_coef 5.0 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --energy_scale 10 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --dataset halfcheetah-medium-expert-v2 --current_exp_label iql_adv_rl_flow_sta5_escale_10 > ./output/slurm_result/iql_adv_rl_flow_sta5_escale_10-halfcheetah-medium-expert-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=1
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --iql_tau 0.5 --divergence_coef 5.0 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --energy_scale 10 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --dataset halfcheetah-medium-replay-v2 --current_exp_label iql_adv_rl_flow_sta5_escale_10 > ./output/slurm_result/iql_adv_rl_flow_sta5_escale_10-halfcheetah-medium-replay-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=2
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --iql_tau 0.5 --divergence_coef 5.0 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --energy_scale 10 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --dataset hopper-medium-v2 --current_exp_label iql_adv_rl_flow_sta5_escale_10 > ./output/slurm_result/iql_adv_rl_flow_sta5_escale_10-hopper-medium-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=2
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --iql_tau 0.5 --divergence_coef 5.0 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --energy_scale 10 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --dataset hopper-medium-expert-v2 --current_exp_label iql_adv_rl_flow_sta5_escale_10 > ./output/slurm_result/iql_adv_rl_flow_sta5_escale_10-hopper-medium-expert-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=2
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --iql_tau 0.5 --divergence_coef 5.0 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --energy_scale 10 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --dataset hopper-medium-replay-v2 --current_exp_label iql_adv_rl_flow_sta5_escale_10 > ./output/slurm_result/iql_adv_rl_flow_sta5_escale_10-hopper-medium-replay-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=2
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --iql_tau 0.5 --divergence_coef 5.0 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --energy_scale 10 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --dataset walker2d-medium-v2 --current_exp_label iql_adv_rl_flow_sta5_escale_10 > ./output/slurm_result/iql_adv_rl_flow_sta5_escale_10-walker2d-medium-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=3
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --iql_tau 0.5 --divergence_coef 5.0 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --energy_scale 10 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --dataset walker2d-medium-expert-v2 --current_exp_label iql_adv_rl_flow_sta5_escale_10 > ./output/slurm_result/iql_adv_rl_flow_sta5_escale_10-walker2d-medium-expert-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=3
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --iql_tau 0.5 --divergence_coef 5.0 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --energy_scale 10 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --dataset walker2d-medium-replay-v2 --current_exp_label iql_adv_rl_flow_sta5_escale_10 > ./output/slurm_result/iql_adv_rl_flow_sta5_escale_10-walker2d-medium-replay-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=3
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --iql_tau 0.5 --divergence_coef 2.0 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --energy_scale 10 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --dataset hopper-medium-expert-v2 --current_exp_label iql_adv_rl_flow_sta2_escale_10 > ./output/slurm_result/iql_adv_rl_flow_sta2_escale_10-hopper-medium-expert-v2.txt 2>&1 &
#sleep 2


##  todo Description: train mode is adv_rl_flow
#export CUDA_VISIBLE_DEVICES=0
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --iql_tau 0.5 --divergence_coef 2.0 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --energy_scale 10 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --dataset halfcheetah-medium-v2 --current_exp_label iql_adv_rl_flow_sta2_escale_10 > ./output/slurm_result/iql_adv_rl_flow_sta2_escale_10-halfcheetah-medium-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=0
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --iql_tau 0.5 --divergence_coef 2.0 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --energy_scale 10 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --dataset halfcheetah-medium-expert-v2 --current_exp_label iql_adv_rl_flow_sta2_escale_10 > ./output/slurm_result/iql_adv_rl_flow_sta2_escale_10-halfcheetah-medium-expert-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=0
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --iql_tau 0.5 --divergence_coef 2.0 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --energy_scale 10 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --dataset halfcheetah-medium-replay-v2 --current_exp_label iql_adv_rl_flow_sta2_escale_10 > ./output/slurm_result/iql_adv_rl_flow_sta2_escale_10-halfcheetah-medium-replay-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=0
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --iql_tau 0.5 --divergence_coef 3.0 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --energy_scale 10 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --dataset halfcheetah-medium-v2 --current_exp_label iql_adv_rl_flow_sta3_escale_10 > ./output/slurm_result/iql_adv_rl_flow_sta3_escale_10-halfcheetah-medium-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=0
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --iql_tau 0.5 --divergence_coef 3.0 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --energy_scale 10 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --dataset halfcheetah-medium-expert-v2 --current_exp_label iql_adv_rl_flow_sta3_escale_10 > ./output/slurm_result/iql_adv_rl_flow_sta3_escale_10-halfcheetah-medium-expert-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=1
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --iql_tau 0.5 --divergence_coef 3.0 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --energy_scale 10 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --dataset halfcheetah-medium-replay-v2 --current_exp_label iql_adv_rl_flow_sta3_escale_10 > ./output/slurm_result/iql_adv_rl_flow_sta3_escale_10-halfcheetah-medium-replay-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=1
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --iql_tau 0.5 --divergence_coef 2.0 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --energy_scale 10 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --dataset hopper-medium-v2 --current_exp_label iql_adv_rl_flow_sta2_escale_10 > ./output/slurm_result/iql_adv_rl_flow_sta2_escale_10-hopper-medium-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=1
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --iql_tau 0.5 --divergence_coef 2.0 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --energy_scale 10 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --dataset hopper-medium-expert-v2 --current_exp_label iql_adv_rl_flow_sta2_escale_10 > ./output/slurm_result/iql_adv_rl_flow_sta2_escale_10-hopper-medium-expert-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=1
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --iql_tau 0.5 --divergence_coef 2.0 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --energy_scale 10 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --dataset hopper-medium-replay-v2 --current_exp_label iql_adv_rl_flow_sta2_escale_10 > ./output/slurm_result/iql_adv_rl_flow_sta2_escale_10-hopper-medium-replay-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=1
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --iql_tau 0.5 --divergence_coef 2.0 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --energy_scale 10 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --dataset hopper-medium-v2 --current_exp_label iql_adv_rl_flow_sta3_escale_10 > ./output/slurm_result/iql_adv_rl_flow_sta3_escale_10-hopper-medium-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=2
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --iql_tau 0.5 --divergence_coef 2.0 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --energy_scale 10 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --dataset hopper-medium-expert-v2 --current_exp_label iql_adv_rl_flow_sta3_escale_10 > ./output/slurm_result/iql_adv_rl_flow_sta3_escale_10-hopper-medium-expert-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=2
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --iql_tau 0.5 --divergence_coef 2.0 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --energy_scale 10 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --dataset hopper-medium-replay-v2 --current_exp_label iql_adv_rl_flow_sta3_escale_10 > ./output/slurm_result/iql_adv_rl_flow_sta3_escale_10-hopper-medium-replay-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=2
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --iql_tau 0.5 --divergence_coef 2.0 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --energy_scale 10 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --dataset walker2d-medium-v2 --current_exp_label iql_adv_rl_flow_sta2_escale_10 > ./output/slurm_result/iql_adv_rl_flow_sta2_escale_10-walker2d-medium-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=2
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --iql_tau 0.5 --divergence_coef 2.0 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --energy_scale 10 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --dataset walker2d-medium-expert-v2 --current_exp_label iql_adv_rl_flow_sta2_escale_10 > ./output/slurm_result/iql_adv_rl_flow_sta2_escale_10-walker2d-medium-expert-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=3
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --iql_tau 0.5 --divergence_coef 2.0 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --energy_scale 10 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --dataset walker2d-medium-replay-v2 --current_exp_label iql_adv_rl_flow_sta2_escale_10 > ./output/slurm_result/iql_adv_rl_flow_sta2_escale_10-walker2d-medium-replay-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=3
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --iql_tau 0.5 --divergence_coef 2.0 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --energy_scale 10 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --dataset walker2d-medium-v2 --current_exp_label iql_adv_rl_flow_sta3_escale_10 > ./output/slurm_result/iql_adv_rl_flow_sta3_escale_10-walker2d-medium-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=3
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --iql_tau 0.5 --divergence_coef 2.0 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --energy_scale 10 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --dataset walker2d-medium-expert-v2 --current_exp_label iql_adv_rl_flow_sta3_escale_10 > ./output/slurm_result/iql_adv_rl_flow_sta3_escale_10-walker2d-medium-expert-v2.txt 2>&1 &
#sleep 2
#
#export CUDA_VISIBLE_DEVICES=3
#echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
#python3 multistep_rl_flow_main.py --mode pretrain_guided_flow --debug_mode False --wandb_log True --rl_mode adv_rl --iql_tau 0.5 --divergence_coef 2.0 --critic_type iql --load_critic_model iql_rl_flow --batch_size 1024 --energy_scale 10 --update_energy_end_epoch 200 --update_behavior_end_epoch 200 --update_flow_start_epoch 0 --dataset walker2d-medium-replay-v2 --current_exp_label iql_adv_rl_flow_sta3_escale_10 > ./output/slurm_result/iql_adv_rl_flow_sta3_escale_10-walker2d-medium-replay-v2.txt 2>&1 &
#sleep 2


wait