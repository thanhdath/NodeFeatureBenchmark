mkdir "/content/drive/My Drive/log" 
feat_dim=128

for alg in dgi
do
    mkdir /content/drive/My\ Drive/log/${alg}
    for seed in $(seq 40 40)
    do
        for data in reddit
        do
            for init in ori degree uniform deepwalk node2vec svd0.5 svd1 hope triangle egonet kcore pagerank coloring clique identity
            do
                python -u main.py --data data/${data} \
                    --alg ${alg} \
                    --init ${init} \
                    --feature_size ${feat_dim} \
                    --seed ${seed} > /content/drive/My\ Drive/log/${alg}/${data}-${init}-seed${seed}
            done
        done
    done
done