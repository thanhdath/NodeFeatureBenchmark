
for data in cora citeseer pubmed
do
feat_size=128
mkdir log
for seed in $(seq 41 42)
do


alg=dgi
mkdir log/$alg
for init in triangle-standard
do
echo $alg-$init
python -u main.py --dataset data/$data  \
    --feature_size $feat_size \
    --init $init \
    --seed $seed \
    --cuda \
    $alg --self-loop > log/$alg/$data-$init-seed$seed
done # init


alg=graphsage
mkdir log/$alg
for init in triangle-standard
do
echo $alg-$init
python -u main.py --dataset data/$data  \
    --feature_size $feat_size \
    --init $init \
    --seed $seed \
    --cuda \
    $alg --aggregator pool > log/$alg/$data-$init-seed$seed
done # init

done # seed
done