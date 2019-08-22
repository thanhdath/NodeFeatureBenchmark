feat_size=128
mkdir log
for data in MUTAG ENZYMES
do

for seed in $(seq 40 42)
do

python mmd.py --dataset data/$data \
    --feature_size $feat_size \
    --init $init \
    --seed $seed \
    $alg > log/$alg/$data-$init-mmd-seed$seed
done

done
