mkdir "log" 
feat_dim=128
alg=dgi
mkdir log/${alg}
for seed in $(seq 40 40)
do
    for data in cora bc pubmed
    do
        for init in ori degree uniform deepwalk ssvd0.5 ssvd1 hope triangle egonet kcore pagerank coloring clique identity node2vec
        do
            python -u -m dgi.train \
                --data data/$data \
                --gpu 0 \
                --init $init \
                --norm_features \
                --seed $seed > log/${alg}/${data}-${init}-seed${seed}
        done
    done
done
