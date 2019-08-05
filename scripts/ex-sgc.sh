mkdir "log" 
feat_dim=128
alg=sgc
mkdir log/${alg}
for seed in $(seq 40 45)
do
    for data in cora pubmed
    do
        for init in ori-pass ori-rowsum ori-standard degree-standard uniform-pass deepwalk-pass ssvd0.5-pass ssvd1-pass hope-pass triangle-standard egonet-standard kcore-standard pagerank-standard coloring-standard clique-standard identity-pass node2vec-pass
        do
            python -u main.py --data data/${data} \
                --alg ${alg} \
                --init ${init} \
                --feature_size ${feat_dim} \
                --seed ${seed} > log/${alg}/${data}-${init}-seed${seed}
        done
    done
done
