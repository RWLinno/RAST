export CUDA_VISIBLE_DEVICES=0,1,2,3
export TQDM_DISABLE=1

# pre-trained baselines
nohup python experiments/train.py -g 3 -c ./baselines/STID/PEMS03.py > ./src/logs/STID_PEMS03_300.log 2>&1 &
nohup python experiments/train.py -g 3 -c ./baselines/STID/PEMS04.py > ./src/logs/STID_PEMS04_300.log 2>&1 &
nohup python experiments/train.py -g 3 -c ./baselines/STID/PEMS07.py > ./src/logs/STID_PEMS07_300.log 2>&1 &
nohup python experiments/train.py -g 3 -c ./baselines/STID/PEMS08.py > ./src/logs/STID_PEMS08_300.log 2>&1 &
nohup python experiments/train.py -g 3 -c ./baselines/STID/SD.py > ./src/logs/STID_SD_300.log 2>&1 &
nohup python experiments/train.py -g 3 -c ./baselines/STID/GBA.py > ./src/logs/STID_GBA_300.log 2>&1 &
nohup python experiments/train.py -g 3 -c ./baselines/STID/GLA.py > ./src/logs/STID_GLA_300.log 2>&1 &

# main results
nohup python experiments/train.py -g 2 -c ./src/scripts/main/RAST_PEMS03.py > ./src/logs/RAST_PEMS03.log 2>&1 &
nohup python experiments/train.py -g 2 -c ./src/scripts/main/RAST_PEMS04.py > ./src/logs/RAST_PEMS04.log 2>&1 &
nohup python experiments/train.py -g 2 -c ./src/scripts/main/RAST_PEMS07.py > ./src/logs/RAST_PEMS07.log 2>&1 &
nohup python experiments/train.py -g 2 -c ./src/scripts/main/RAST_PEMS08.py > ./src/logs/RAST_PEMS08.log 2>&1 &

# large scale
nohup python experiments/train.py -g 1 -c ./src/scripts/main/RAST_SD.py > ./src/logs/RAST_SD.log 2>&1 &
nohup python experiments/train.py -g 1 -c ./src/scripts/main/RAST_GBA.py > ./src/logs/RAST_GBA.log 2>&1 &
nohup python experiments/train.py -g 1 -c ./src/scripts/main/RAST_GLA.py > ./src/logs/RAST_GLA.log 2>&1 &

# STID+RAST
nohup python experiments/train.py -g 3 -c ./src/scripts/pretrain/RAST_STID_PEMS03.py > ./src/logs/RAST_STID_PEMS03.log 2>&1 &
nohup python experiments/train.py -g 3 -c ./src/scripts/pretrain/RAST_STID_PEMS04.py > ./src/logs/RAST_STID_PEMS04.log 2>&1 &
nohup python experiments/train.py -g 3 -c ./src/scripts/pretrain/RAST_STID_PEMS07.py > ./src/logs/RAST_STID_PEMS07.log 2>&1 &
nohup python experiments/train.py -g 3 -c ./src/scripts/pretrain/RAST_STID_PEMS08.py > ./src/logs/RAST_STID_PEMS08.log 2>&1 &
nohup python experiments/train.py -g 2 -c ./src/scripts/pretrain/RAST_STID_SD.py > ./src/logs/RAST_STID_SD.log 2>&1 &
nohup python experiments/train.py -g 2 -c ./src/scripts/pretrain/RAST_STID_GBA.py > ./src/logs/RAST_STID_GBA.log 2>&1 &
nohup python experiments/train.py -g 2 -c ./src/scripts/pretrain/RAST_STID_GLA.py > ./src/logs/RAST_STID_GLA.log 2>&1 &

# components ablation
nohup python experiments/train.py -g 0 -c ./src/scripts/ablation/PEMS04_only_retrieval.py > ./src/logs/PEMS04_only_retrieval.log 2>&1 &
nohup python experiments/train.py -g 1 -c ./src/scripts/ablation/PEMS04_only_query.py > ./src/logs/PEMS04_only_query.log 2>&1 &
nohup python experiments/train.py -g 2 -c ./src/scripts/ablation/PEMS04_without_query_embedding.py > ./src/logs/PEMS04_without_query_embedding.log 2>&1 &
nohup python experiments/train.py -g 3 -c ./src/scripts/ablation/PEMS04_without_retrieval_embedding.py > ./src/logs/PEMS04_without_retrieval_embedding.log 2>&1 &
nohup python experiments/train.py -g 0 -c ./src/scripts/ablation/PEMS04_without_temporal_retrieval.py > ./src/logs/PEMS04_without_temporal_retrieval.log 2>&1 &
nohup python experiments/train.py -g 1 -c ./src/scripts/ablation/PEMS04_without_spatial_retrieval.py > ./src/logs/PEMS04_without_spatial_retrieval.log 2>&1 &

# hyperparam
nohup python experiments/train.py -g 0 -c ./src/scripts/hyperparam/PEMS04_retrieval_32.py > ./src/logs/PEMS04_retrieval_32.log 2>&1 &
nohup python experiments/train.py -g 1 -c ./src/scripts/hyperparam/PEMS04_retrieval_64.py > ./src/logs/PEMS04_retrieval_64.log 2>&1 &
nohup python experiments/train.py -g 2 -c ./src/scripts/hyperparam/PEMS04_retrieval_128.py > ./src/logs/PEMS04_retrieval_128.log 2>&1 &
nohup python experiments/train.py -g 3 -c ./src/scripts/hyperparam/PEMS04_retrieval_256.py > ./src/logs/PEMS04_retrieval_256.log 2>&1 &
nohup python experiments/train.py -g 0 -c ./src/scripts/hyperparam/PEMS04_retrieval_512.py > ./src/logs/PEMS04_retrieval_512.log 2>&1 &

nohup python experiments/train.py -g 0 -c ./src/scripts/hyperparam/PEMS04_query_32.py > ./src/logs/PEMS04_query_32.log 2>&1 &
nohup python experiments/train.py -g 1 -c ./src/scripts/hyperparam/PEMS04_query_64.py > ./src/logs/PEMS04_query_64.log 2>&1 &
nohup python experiments/train.py -g 2 -c ./src/scripts/hyperparam/PEMS04_query_128.py > ./src/logs/PEMS04_query_128.log 2>&1 &
nohup python experiments/train.py -g 3 -c ./src/scripts/hyperparam/PEMS04_query_256.py > ./src/logs/PEMS04_query_256.log 2>&1 &
nohup python experiments/train.py -g 0 -c ./src/scripts/hyperparam/PEMS04_query_512.py > ./src/logs/PEMS04_query_512.log 2>&1 &

nohup python experiments/train.py -g 0 -c ./src/scripts/hyperparam/PEMS04_interval_1.py > ./src/logs/PEMS04_interval_1.log 2>&1 &
nohup python experiments/train.py -g 1 -c ./src/scripts/hyperparam/PEMS04_interval_5.py > ./src/logs/PEMS04_interval_5.log 2>&1 &
nohup python experiments/train.py -g 2 -c ./src/scripts/hyperparam/PEMS04_interval_10.py > ./src/logs/PEMS04_interval_10.log 2>&1 &
nohup python experiments/train.py -g 3 -c ./src/scripts/hyperparam/PEMS04_interval_20.py > ./src/logs/PEMS04_interval_20.log 2>&1 &
nohup python experiments/train.py -g 0 -c ./src/scripts/hyperparam/PEMS04_interval_50.py > ./src/logs/PEMS04_interval_50.log 2>&1 &

nohup python experiments/train.py -g 0 -c ./src/scripts/hyperparam/PEMS04_layers_1.py > ./src/logs/PEMS04_layers_1.log 2>&1 &
nohup python experiments/train.py -g 1 -c ./src/scripts/hyperparam/PEMS04_layers_2.py > ./src/logs/PEMS04_layers_2.log 2>&1 &
nohup python experiments/train.py -g 2 -c ./src/scripts/hyperparam/PEMS04_layers_3.py > ./src/logs/PEMS04_layers_3.log 2>&1 &
nohup python experiments/train.py -g 3 -c ./src/scripts/hyperparam/PEMS04_layers_4.py > ./src/logs/PEMS04_layers_4.log 2>&1 &
nohup python experiments/train.py -g 2 -c ./src/scripts/hyperparam/PEMS04_layers_5.py > ./src/logs/PEMS04_layers_5.log 2>&1 &