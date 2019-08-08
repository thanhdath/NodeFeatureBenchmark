data=pubmed
feat_size=128
mkdir log
for seed in $(seq 40 49)
do
# alg=nope
# for init in ori ori-rowsum ori-standard deepwalk hope
# do
# echo $alg-$init
# python -u main.py --dataset data/$data  \
#     --feature_size $feat_size \
#     --init $init \
#     --seed $seed \
#     --cuda \
#     --logreg-wc 5e-5 --logreg-bias \
#     $alg > log/$alg/$data-$init-seed$seed
# done

# for alg in sgc 
# do
# mkdir log/$alg
# for init in ori ori-rowsum ori-standard degree-standard uniform deepwalk ssvd0.5 ssvd1 hope triangle-standard egonet-standard kcore-standard pagerank-standard coloring-standard clique-standard identity node2vec
# do
# echo $alg-$init
# python -u main.py --dataset data/$data  \
#     --feature_size $feat_size \
#     --init $init \
#     --seed $seed \
#     --cuda \
#     --logreg-wc 5e-5 --logreg-bias \
#     $alg > log/$alg/$data-$init-seed$seed
# done # init
# done # alg

alg=dgi
mkdir log/$alg
for init in ori ori-rowsum ori-standard degree-standard uniform deepwalk ssvd0.5 ssvd1 hope triangle-standard egonet-standard kcore-standard pagerank-standard coloring-standard clique-standard identity
do
echo $alg-$init
python -u main.py --dataset data/$data  \
    --feature_size $feat_size \
    --init $init \
    --seed $seed \
    --cuda \
    $alg > log/$alg/$data-$init-seed$seed
done # init

done # seed
