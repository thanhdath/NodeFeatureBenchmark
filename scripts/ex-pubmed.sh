data=pubmed
feat_size=128

for seed in $(seq 40 49)
do
for alg in nope sgc 
do
for init in ori ori-rowsum ori-standard degree-standard uniform deepwalk ssvd0.5 ssvd1 hope triangle-standard egonet-standard kcore-standard pagerank-standard coloring-standard clique-standard identity node2vec
do
echo $alg-$init
python -u main.py --dataset $data  \
    --feature_size $feat_size \
    --init $init \
    --seed $seed \
    --cuda \
    --logreg-wc 5e-5 --bias \
    $alg > log/$alg/$data-$init-seed$seed
done # init
done # alg
done # seed
