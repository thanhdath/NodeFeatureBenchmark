for data in bc
do
mkdir log
for feat_size in 128 256 512
do

# for seed in $(seq 40 42)
# do
#     alg=nope
#     mkdir log/$alg
#     for init in struc2vec
#     do
#     echo $data-$alg-$init-$feat_size
#     python -u main.py --dataset data/$data  \
#         --feature_size $feat_size \
#         --init $init \
#         --seed $seed \
#         --cuda \
#         $alg > log/$alg/$data-$init-dim$feat_size-seed$seed
#     done # init
# done

# for seed in $(seq 40 42)
# do
#     alg=sgc
#     mkdir log/$alg
#     for init in struc2vec
#     do
#     echo $data-$alg-$init-$feat_size
#     python -u main.py --dataset data/$data  \
#         --feature_size $feat_size \
#         --init $init \
#         --seed $seed \
#         --cuda \
#         $alg > log/$alg/$data-$init-dim$feat_size-seed$seed
#     done # init
# done

# for seed in $(seq 40 42)
# do
#     alg=dgi
#     mkdir log/$alg
#     for init in deepwalk ssvd0.5 ssvd1 hope
#     do
#     echo $data-$alg-$init-$feat_size
#     python -u main.py --dataset data/$data  \
#         --feature_size $feat_size \
#         --init $init \
#         --seed $seed \
#         --cuda \
#         $alg --self-loop > log/$alg/$data-$init-dim$feat_size-seed$seed
#     done # init
# done

for seed in $(seq 40 42)
do
    alg=graphsage
    mkdir log/$alg
    for init in deepwalk ssvd0.5 ssvd1 hope
    do
    echo $data-$alg-$init-$feat_size
    python -u main.py --dataset data/$data  \
        --feature_size $feat_size \
        --init $init \
        --seed $seed \
        --cuda \
        $alg --aggregator mean > log/$alg/$data-$init-dim$feat_size-seed$seed
    done # init
done # seed

done

done 
