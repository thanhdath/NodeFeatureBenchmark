feat_size=128
mkdir log
for data in cora citeseer pubmed
do

for seed in $(seq 40 42)
do

for init in ssvd1 ssvd0.5 deepwalk graphwave hope ori
do
python -u mmd.py --dataset data/$data \
    --feature_size $feat_size \
    --init $init \
    --seed $seed > log/$data-$init-mmd-seed$seed
done
done
done
