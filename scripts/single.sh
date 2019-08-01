alg=sgc
seed=40
data=cora
init=ssvd

python -u main.py --data data/${data} \
    --alg ${alg} \
    --init ${init} \
    --feature_size 10 \
    --seed ${seed}
