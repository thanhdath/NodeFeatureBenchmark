feat_size=128
mkdir log
for data in DD
do

for seed in $(seq 40 44)
do
alg=diffpool
for init in degree-standard uniform deepwalk kcore-standard egonet-standard pagerank-standard ori ori-rowsum ori-standard label label-standard
do
python graph_classify.py --dataset data/$data \
    --feature_size $feat_size \
    --init $init \
    --seed $seed \
    --cuda \
    $alg --pool_ratio 0.15 --num_pool 1 > log/$alg/$data-$init-seed$seed
done


# alg=gin
# for init in ori ori-rowsum ori-standard label label-standard degree-standard uniform deepwalk ssvd0.5 ssvd1 hope triangle-standard egonet-standard kcore-standard pagerank-standard coloring-standard clique-standard identity
# do
# python graph_classify.py --dataset data/$data \
#     --feature_size $feat_size \
#     --init $init \
#     --seed $seed \
#     --cuda \
#     $alg > log/$alg/$data-$init-seed$seed
# done
done
done
