feat_size=128
mkdir log
for data in ENZYMES DD 
do

for seed in $(seq 40 42)
do
alg=diffpool
for init in hope
do
python graph_classify.py --dataset data/$data \
    --feature_size $feat_size \
    --init $init \
    --seed $seed \
    --cuda \
    $alg --pool_ratio 0.15 --num_pool 1 > log/$alg/$data-$init-seed$seed
done


alg=gin
for init in hope ssvd0.5 ssvd1
do
python graph_classify.py --dataset data/$data \
    --feature_size $feat_size \
    --init $init \
    --seed $seed \
    --cuda \
    $alg > log/$alg/$data-$init-seed$seed
done
done
done
