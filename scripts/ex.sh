mkdir log 
feat_dim=128
for alg in sgc
do
    mkdir log/${alg}
    for seed in $(seq 40 45)
    do
        for data in cora bc pubmed
        do
            for init in ori degree uniform deepwalk node2vec hope triangle egonet kcore pagerank coloring clique
            do
                python -u main.py --data data/${data} \
                    --alg ${alg} \
                    --init ${init} \
                    --feature_size ${feat_dim} \
                    --seed ${seed} > log/${alg}/${data}-${init}-seed${seed}
            done
        done
    done
done
