alias python="python3"
feat_size=128
mkdir log
for seed in $(seq 40 40)
do

alg=sgc
data=reddit_self_loop
mkdir log/$alg
for init in triangle-standard
do
echo $alg-$init
python -u main.py --dataset data/$data  \
    --feature_size $feat_size \
    --init $init \
    --seed $seed \
    --cuda \
    $alg > log/$alg/$data-$init-seed$seed
done # init

done # seed
