feat_dim=128
alg=logistic
mkdir log/${alg}
for seed in $(seq 40 50)
do
    for data in cora pubmed
    do
        for init in ori-pass ori-standard ori-rowsum deepwalk-pass hope-pass node2vec-pass
        do
            python -u main.py --data data/${data} \
                --alg ${alg} \
                --init ${init} \
                --feature_size ${feat_dim} \
                --seed ${seed} > log/${alg}/${data}-${init}-seed${seed}
        done
    done
done
