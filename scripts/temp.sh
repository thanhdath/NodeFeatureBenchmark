alias python="python3"
feat_size=128
mkdir log
for seed in $(seq 40 40)
do

alg=nope
data=reddit
for init in line
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
