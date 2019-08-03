feat_dim=128
alg=logistic
log/${alg}
for seed in $(seq 40 40)
do
    for data in cora bc pubmed reddit
    do
        for init in ori deepwalk hope node2vec
        do
            python -u main.py --data data/${data} \
                --alg ${alg} \
                --init ${init} \
                --feature_size ${feat_dim} \
                --seed ${seed} \
                --norm_features > log/${alg}/${data}-${init}-seed${seed}
        done
    done
done
