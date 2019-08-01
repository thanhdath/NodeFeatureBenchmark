alg=sgc
seed=40
data=bc
init=clique

python -u main.py --data data/${data} \
    --alg ${alg} \
    --init ${init} \
    --feature_size 128 \
    --seed ${seed} > log/$alg/$data-$init-seed$seed &
