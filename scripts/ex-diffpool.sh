workers=7
data=ENZYMES
feat_dim=128
mkdir log/diffpool
for init in ori degree uniform deepwalk ssvd0.5 ssvd1 hope triangle egonet kcore pagerank coloring clique identity node2vec
do 
echo $init
PYTHONPATH=diffpool:. python -u diffpool/train.py \
    --bmname=$data \
    --assign-ratio=0.1 \
    --cuda=0 \
    --num-classes=6 \
    --num_workers $workers \
    --method=soft-assign \
    --train-ratio 0.7 \
    --test-ratio 0.2 \
    --init $init \
    --linkpred \
    --input-dim $feat_dim > log/diffpool/$data-$init-seed40
done
