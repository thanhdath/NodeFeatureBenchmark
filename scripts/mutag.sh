feat_size=128
mkdir log
for data in MUTAG DD FIRSTMM_DB
do

for seed in $(seq 40 44)
do
# alg=diffpool
# data=ENZYMES
# for init in ori ori-rowsum ori-standard label label-standard degree-standard uniform deepwalk ssvd0.5 ssvd1 hope triangle-standard egonet-standard kcore-standard pagerank-standard coloring-standard clique-standard identity
# do
# python graph_classify.py --dataset data/$data \
#     --feature_size $feat_size \
#     --init $init \
#     --seed $seed \
#     --cuda \
#     $alg --pool_ratio 0.1 --num_pool 1 > log/$alg/$data-$init-seed$seed
# done


alg=gin
for init in ori ori-rowsum ori-standard label label-standard degree-standard uniform deepwalk ssvd0.5 ssvd1 hope triangle-standard egonet-standard kcore-standard pagerank-standard coloring-standard clique-standard identity
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
