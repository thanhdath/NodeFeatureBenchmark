mkdir log
alg=gin
mkdir log/$alg 

for data in ENZYMES DD
do
for init in ori degree uniform deepwalk ssvd0.5 ssvd1 hope triangle egonet kcore pagerank coloring clique identity label node2vec
do
echo $init
PYTHONPATH=gin:. python -u gin/train.py \
    --data ENZYMES \
    --init $init \
    --device 0 \
    --feature_size 128 \
    --norm_features \
    --seed 40 > log/$alg/$data-$init-seed40
done
done
