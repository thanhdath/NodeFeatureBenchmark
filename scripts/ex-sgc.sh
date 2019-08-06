mkdir "log" 
feat_dim=128
alg=sgc
mkdir log/${alg}
for seed in $(seq 40 45)
do
    for data in cora
    do
        for init in ori ori-rowsum ori-standard degree-standard uniform deepwalk ssvd0.5 ssvd1 hope triangle-standard egonet-standard kcore-standard pagerank-standard coloring-standard clique-standard identity node2vec
        do
            python -u main.py --data data/${data} \
                --alg ${alg} \
                --init ${init} \
                --feature_size ${feat_dim} \
                --seed ${seed} \
                --shuffle > log/${alg}/${data}-${init}-seed${seed}
        done
    done
done
