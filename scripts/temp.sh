feat_size=128
data=ENZYMES
mkdir log

for seed in $(seq 40 44)
do
alg=diffpool
for init in ori-rowsum ori-standard label-standard
do
python graph_classify.py --dataset data/$data \
    --feature_size $feat_size \
    --init $init \
    --seed $seed \
    --cuda \
    $alg --pool_ratio 0.1 --num_pool 1 > log/$alg/$data-$init-seed$seed
done


alg=gin
for init in ori-rowsum ori-standard label-standard
do
python graph_classify.py --dataset data/$data \
    --feature_size $feat_size \
    --init $init \
    --seed $seed \
    --cuda \
    $alg > log/$alg/$data-$init-seed$seed
done
done
