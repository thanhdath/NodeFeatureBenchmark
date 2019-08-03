alg=sgc
seed=40
data=cora
init=uniform

python -u main.py --data data/${data} \
    --alg ${alg} \
    --init ${init} \
    --feature_size 128 \
    --train_features \
    --norm_features \
    --seed ${seed} > log/$alg/$data-$init-seed$seed
