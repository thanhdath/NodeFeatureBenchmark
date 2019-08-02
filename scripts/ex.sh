mkdir "log" 
feat_dim=128

for alg in sgc
do
    log/${alg}
    for seed in $(seq 40 40)
    do
        for data in bc pubmed
        do
            for init in ori degree uniform deepwalk node2vec ssvd0.5 ssvd1 hope triangle egonet kcore pagerank coloring clique identity
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
done
