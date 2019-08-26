# alias python="python3"
feat_size=128
mkdir log
# for seed in $(seq 40 42)
# do
# alg=nope
# data=reddit
# for init in deepwalk hope
# do
# echo $alg-$init
# python -u main.py --dataset data/$data  \
#     --feature_size $feat_size \
#     --init $init \
#     --seed $seed \
#     --cuda \
#     $alg > log/$alg/$data-$init-seed$seed
# done # init

for seed in $(seq 40 41)
do
    alg=sgc
    data=reddit_self_loop
    mkdir log/$alg
    for init in label
    do
    echo $alg-$init
    python -u main.py --dataset data/$data  \
        --feature_size $feat_size \
        --init $init \
        --seed $seed \
        --cuda \
        $alg > log/$alg/$data-$init-seed$seed
    done # init
done

for seed in $(seq 40 41)
do
    alg=dgi
    data=reddit_self_loop
    mkdir log/$alg
    for init in label
    do
    echo $alg-$init
    python -u main.py --dataset data/$data  \
        --feature_size $feat_size \
        --init $init \
        --seed $seed \
        --cuda \
        $alg --self-loop > log/$alg/$data-$init-seed$seed
    done # init
done 

# for seed in $(seq 40 41)
# do
#     alg=graphsage
#     data=reddit_self_loop
#     mkdir log/$alg
#     for init in ori ori-rowsum ori-standard
#     do
#     echo $alg-$init
#     python -u main.py --dataset data/$data  \
#         --feature_size $feat_size \
#         --init $init \
#         --seed $seed \
#         --cuda \
#         $alg --aggregator mean > log/$alg/$data-$init-seed$seed
#     done # init
# done # seed
