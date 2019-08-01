alg=logistic
seed=40
mkdir log/$alg
for data in cora
do
for init in ori deepwalk hope node2vec
do
python -u main.py --data data/${data} \
    --alg ${alg} \
    --init ${init} \
    --feature_size 128 \
    --seed ${seed} > log/$alg/$data-$init-seed$seed
done
done 
