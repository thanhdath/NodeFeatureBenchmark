data=cora
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
    $alg > log/$alg/$data-$init-seed$seed
done # init
done # alg
done # seed

# python3 sgc.py --dataset cora --gpu 0
# python3 sgc.py --dataset citeseer --weight-decay 5e-5 --n-epochs 150 --bias --gpu 0
# python3 sgc.py --dataset pubmed --weight-decay 5e-5 --bias --gpu 0