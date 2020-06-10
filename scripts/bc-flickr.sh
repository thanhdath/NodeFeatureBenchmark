mkdir log
data=cora
seed=40
feat_size=128
init=ssvd0.5
echo $init
python -u main.py --dataset data/$data  \
    --feature_size $feat_size \
    --init $init \
    --seed $seed \
    --cuda --aggregator mean

# inits="ssvd0.5 ssvd1 hope line gf triangles-standard kcore-standard egonet-standard pagerank-standard coloring-standard clique-standard deepwalk"
