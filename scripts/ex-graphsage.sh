mkdir "log" 
feat_dim=128
alg=graphsage
mkdir log/${alg}
for seed in $(seq 40 40)
do
    for data in cora bc pubmed
    do
        for init in ori degree uniform deepwalk ssvd0.5 ssvd1 hope triangle egonet kcore pagerank coloring clique identity node2vec
        do
            python -u -m graphsage.graphsage \
                --data data/$data --aggregator-type pool \
                --feature_size $feat_dim \
                --gpu 0 \
                --init $init \
                --norm_features \
                --seed $seed > log/${alg}/${data}-${init}-seed${seed}
        done
    done
done


mkdir "log" 
feat_dim=128
alg=graphsage
mkdir log/${alg}
for seed in $(seq 40 50)
do
    for data in cora pubmed
    do
        for init in ori ori-standard ori-rowsum
        do
            python -u -m graphsage.graphsage \
                --data data/$data --aggregator-type pool \
                --feature_size $feat_dim \
                --gpu 0 \
                --init $init \
                --seed $seed \
                --shuffle > log/${alg}/${data}-${init}-seed${seed}
        done
    done
done
