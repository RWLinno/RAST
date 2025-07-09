export TQDM_DISABLE=1

nohup python experiments/train.py -c ViST/train_SD_without_CCG.py -g 0 > logs/ViST_SD_without_CCG.txt 2>&1 &

nohup python experiments/train.py -c ViST/train_SD_without_TCG.py -g 1 > logs/ViST_SD_without_TCG.txt 2>&1 &

nohup python experiments/train.py -c ViST/train_SD_without_SCG.py -g 2 > logs/ViST_SD_without_SCG.txt 2>&1 &
