workers=6
data=ENZYMES
feat_dim=128
seed=40
mkdir log/diffpool
for init in ori degree uniform deepwalk ssvd0.5 ssvd1 hope triangle egonet kcore pagerank coloring clique identity node2vec
do 
echo $init
PYTHONPATH=diffpool:. python diffpool/train.py \
    --data $data \
    --pool_ratio 0.10 \
    --num_pool 1 \
    --cuda 0 \
    --norm_features \
    --init $init \
    --num_workers $workers \
    --train-ratio 0.7 \
    --test-ratio 0.2 \
    --feature_size $feat_dim \
    --seed $seed > log/diffpool/$data-$init-seed40
done


data=DD
feat_dim=128
seed=40
mkdir log/diffpool
for init in ori degree uniform deepwalk ssvd0.5 ssvd1 hope triangle egonet kcore pagerank coloring clique identity node2vec
do 
echo $init
PYTHONPATH=diffpool:. python diffpool/train.py \
    --data $data \
    --pool_ratio 0.15 \
    --num_pool 1 \
    --cuda 0 \
    --norm_features \
    --init $init \
    --num_workers $workers \
    --train-ratio 0.7 \
    --test-ratio 0.2 \
    --feature_size $feat_dim \
    --seed $seed > log/diffpool/$data-$init-seed40
done
