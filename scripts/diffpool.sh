init=ori
input_dim=128
workers=12

PYTHONPATH=diffpool:. python diffpool/train.py \
    --bmname=ENZYMES \
    --assign-ratio=0.1 \
    --hidden-dim=30 \
    --output-dim=30 \
    --max_nodes=10000000 \
    --epochs 1000 \
    --cuda=1 \
    --num-classes=6 \
    --num-workers $workers \
    --method=soft-assign \
    --train-ratio 0.7 \
    --test-ratio 0.2 \
    --init $init \
    --input_dim $feat_dim 