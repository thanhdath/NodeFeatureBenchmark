for data in bc flickr 
do
feat_size=128
mkdir log
for seed in $(seq 40 44)
do
alg=nope
for init in deepwalk hope line gf
do
echo $alg-$init
python -u main.py --dataset data/$data  \
    --feature_size $feat_size \
    --init $init \
    --seed $seed \
    --cuda \
    $alg > log/$alg/$data-$init-seed$seed
done # init

alg=sgc
mkdir log/$alg
for init in degree-standard uniform deepwalk ssvd0.5 ssvd1 hope line gf triangles-standard kcore-standard egonet-standard pagerank-standard coloring-standard clique-standard identity
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

for seed in $(seq 40 44)
do
alg=dgi
mkdir log/$alg
for init in degree-standard uniform deepwalk ssvd0.5 ssvd1 hope line gf triangles-standard kcore-standard egonet-standard pagerank-standard coloring-standard clique-standard identity
do
echo $alg-$init
python -u main.py --dataset data/$data  \
    --feature_size $feat_size \
    --init $init \
    --seed $seed \
    --cuda \
    $alg --self-loop > log/$alg/$data-$init-seed$seed
done # init


alg=graphsage
mkdir log/$alg
for init in degree-standard uniform deepwalk ssvd0.5 ssvd1 hope line gf triangles-standard kcore-standard egonet-standard pagerank-standard coloring-standard clique-standard identity
do
echo $alg-$init
python -u main.py --dataset data/$data  \
    --feature_size $feat_size \
    --init $init \
    --seed $seed \
    --cuda \
    $alg --aggregator pool > log/$alg/$data-$init-seed$seed
done # init

done # seed

done