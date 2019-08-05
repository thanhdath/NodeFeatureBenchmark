mkdir "log" 
feat_dim=128
for alg in sgc logistic
do
    mkdir log/${alg}-nonorm
    for seed in $(seq 40 40)
    do
        for data in reddit
        do
            for init in ori
            do
                python -u main.py --data data/${data} \
                    --alg ${alg} \
                    --init ${init} \
                    --feature_size ${feat_dim} \
                    --seed ${seed}  > log/${alg}-nonorm/${data}-${init}-seed${seed}
            done
        done
    done
done
