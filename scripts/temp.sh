

feat_size=128
mkdir log
for seed in $(seq 40 42)
do
data=reddit_self_loop
alg=sgc
mkdir log/$alg
for init in deepwalk
do
echo $alg-$init
python -u inductive.py --dataset data/$data  \
    --feature_size $feat_size \
    --init $init \
    --seed $seed \
    --cuda \
    $alg > log/$alg/$data-inductive-$init-seed$seed
done # init


data=reddit_self_loop
alg=dgi
mkdir log/$alg
for init in deepwalk
do
echo $alg-$init
python -u inductive.py --dataset data/$data  \
    --feature_size $feat_size \
    --init $init \
    --seed $seed \
    --cuda \
    $alg --self-loop > log/$alg/$data-inductive-$init-seed$seed
done # init

done
