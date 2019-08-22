# data=youtube
# feat_size=128
# mkdir log
# for seed in $(seq 40 42)
# do
# alg=nope
# for init in deepwalk hope line gf
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
# mkdir log/$alg
# for init in uniform deepwalk ssvd0.5 ssvd1 hope line gf pagerank-standard degree-standard triangle-standard kcore-standard egonet-standard clique-standard coloring-standard
# do
# echo $alg-$init
# python -u main.py --dataset data/$data  \
#     --feature_size $feat_size \
#     --init $init \
#     --seed $seed \
#     --cuda \
#     $alg > log/$alg/$data-$init-seed$seed
# done # init
# done
# 

data=youtube
feat_size=128
mkdir log
for seed in $(seq 40 42)
do
alg=dgi
mkdir log/$alg
for init in uniform deepwalk ssvd0.5 ssvd1 hope line gf pagerank-standard degree-standard triangle-standard kcore-standard egonet-standard clique-standard coloring-standard
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
for init in uniform deepwalk ssvd0.5 ssvd1 hope line gf pagerank-standard degree-standard triangle-standard kcore-standard egonet-standard clique-standard coloring-standard
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