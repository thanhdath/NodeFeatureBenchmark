mkdir log 
feat_dim=128
alg=logistic
mkdir log/${alg}
seed=40
data=reddit 
for init in ori deepwalk hope node2vec 
do 
    python -u main.py --data data/${data} \
                    --alg ${alg} \
                    --init ${init} \
                    --feature_size ${feat_dim} \
                    --seed ${seed} > log/${alg}/${data}-${init}-seed${seed}
done


for alg in sgc
do
    mkdir log/${alg}
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
                    --seed ${seed} > log/${alg}/${data}-${init}-seed${seed}
            done
        done
    done
done