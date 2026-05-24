nohup python train.py > output.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python train.py --config config.yaml > ./nohup_log/output_260425_v1.log 2>&1 &

sudo ip link delete Meta

python test_model.py \
    --model_path /mnt/sdb11/HK/UAV_SFC_code/DRL/training/checkpoints/ppo_seedNone_20260419_212658/policy_final.pt \
    --config config_small.yaml \
    --num_episodes 3 \
    --device cuda


python test/compare_algorithms.py \
    --model-path /mnt/sdb11/HK/UAV_SFC_code/DRL/training/checkpoints/ppo_seedNone_20260426_192308/policy_final.pt \
    --config-path test/config.yaml \
    --num-episodes 5 \
    --device cpu \
    --output test/comparison_results_10timeslot.csv