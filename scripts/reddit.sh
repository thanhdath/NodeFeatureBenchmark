alias python="python3"
feat_size=128
mkdir log
for seed in $(seq 40 44)
do
# alg=nope
# data=reddit
# for init in deepwalk hope
# do
# echo $alg-$init
# python -u main.py --dataset data/$data  \
#     --feature_size $feat_size \
#     --init $init \
#     --seed $seed \
#     --cuda \
#     $alg > log/$alg/$data-$init-seed$seed
# done # init

# alg=sgc
# data=reddit_self_loop
# mkdir log/$alg
# for init in deepwalk ssvd0.5 ssvd1 hope
# do
# echo $alg-$init
# python -u main.py --dataset data/$data  \
#     --feature_size $feat_size \
#     --init $init \
#     --seed $seed \
#     --cuda \
#     $alg > log/$alg/$data-$init-seed$seed
# done # init

alg=dgi
data=reddit_self_loop
mkdir log/$alg
for init in ori ori-rowsum ori-standard deepwalk hope ssvd0.5 ssvd1
do
echo $alg-$init
python -u main.py --dataset data/$data  \
    --feature_size $feat_size \
    --init $init \
    --seed $seed \
    --cuda \
    $alg --self-loop > log/$alg/$data-$init-seed$seed
done # init


# alg=graphsage
# data=reddit
# mkdir log/$alg
# for init in ori ori-rowsum ori-standard deepwalk hope ssvd0.5 ssvd1
# do
# echo $alg-$init
# python -u main.py --dataset data/$data  \
#     --feature_size $feat_size \
#     --init $init \
#     --seed $seed \
#     --cuda \
#     $alg --aggregator pool > log/$alg/$data-$init-seed$seed
# done # init

done # seed
