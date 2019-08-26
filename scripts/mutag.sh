feat_size=128
mkdir log
for data in MUTAG
do
for seed in $(seq 40 41)
do
alg=diffpool
for init in triangle-standard
do
python -u graph_classify.py --dataset data/$data \
    --feature_size $feat_size \
    --init $init \
    --seed $seed \
    --cuda \
    $alg --pool_ratio 0.15 --num_pool 1 > log/$alg/$data-$init-seed$seed
done
# done
# done


# alg=gin
# for init in degree-standard uniform deepwalk ssvd0.5 ssvd1 hope gf triangle-standard kcore-standard egonet-standard clique-standard graphwave
# do
# python -u graph_classify.py --dataset data/$data \
#     --feature_size $feat_size \
#     --init $init \
#     --seed $seed \
#     --cuda \
#     $alg > log/$alg/$data-$init-seed$seed
# done

# alg=simple
# mkdir log/$alg
# for init in degree uniform deepwalk ssvd0.5 ssvd1 hope triangle kcore egonet coloring clique graphwave ori ori-rowsum label gf
# do
# python -u graph_classify.py --dataset data/$data \
#     --feature_size $feat_size \
#     --init $init \
#     --seed $seed \
#     --cuda \
#     $alg --operator mean > log/$alg/$data-$init-seed$seed
# done
done
done
