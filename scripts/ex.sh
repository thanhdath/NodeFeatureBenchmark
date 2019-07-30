mkdir log 
for alg in sgc
do
    mkdir log/${alg}
    for seed in $(seq 40 50)
    do
        for data in cora bc arxiv
        do
            for init in degree uniform identity 
            do
                python -u main.py --data data/${data} \
                    --alg ${alg} \
                    --init ${init} \
                    --feature_size 10 \
                    --seed ${seed} > log/${alg}/${data}-${init}-seed${seed}
            done
        done
    done
done
