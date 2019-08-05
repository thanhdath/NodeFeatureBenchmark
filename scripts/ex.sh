mkdir "log" 
feat_dim=128
for alg in sgc
do
    mkdir log/${alg}
    for seed in $(seq 40 40)
    do
        for data in cora pubmed
        do
            for init in ori degree uniform deepwalk ssvd0.5 ssvd1 hope triangle egonet kcore pagerank coloring clique identity node2vec
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
