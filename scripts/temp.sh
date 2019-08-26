data=cora
feat_size=128
inits=label
mkdir log
for seed in $(seq 40 42)
do
alg=nope
for init in $inits
do
echo $alg-$init
python -u main.py --dataset data/$data  \
    --feature_size $feat_size \
    --init $init \
    --seed $seed \
    --cuda \
    $alg > log/$alg/$data-$init-seed$seed
done # init

alg=sgc
mkdir log/$alg
for init in $inits
do
echo $alg-$init
python -u main.py --dataset data/$data  \
    --feature_size $feat_size \
    --init $init \
    --seed $seed \
    --cuda \
    $alg > log/$alg/$data-$init-seed$seed
done # init
done

for seed in $(seq 40 42)
do
alg=dgi
mkdir log/$alg
for init in $inits
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
for init in $inits
do
echo $alg-$init
python -u main.py --dataset data/$data  \
    --feature_size $feat_size \
    --init $init \
    --seed $seed \
    --cuda \
    $alg --aggregator mean > log/$alg/$data-$init-seed$seed
done # init

done # seed
