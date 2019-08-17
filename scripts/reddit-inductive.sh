

feat_size=128
mkdir log
for seed in $(seq 40 42)
do
data=reddit_self_loop
alg=sgc
mkdir log/$alg
for init in ori ori-rowsum ori-standard degree-standard uniform deepwalk ssvd0.5 ssvd1 hope line gf triangle-standard egonet-standard kcore-standard pagerank-standard coloring-standard clique-standard identity
do
echo $alg-$init
python -u inductive.py --dataset data/$data  \
    --feature_size $feat_size \
    --init $init \
    --seed $seed \
    --cuda \
    $alg > log/$alg/$data-inductive-$init-seed$seed
done # init
done

data=reddit_self_loop
alg=dgi
mkdir log/$alg
for init in ori ori-rowsum ori-standard degree-standard uniform deepwalk ssvd0.5 ssvd1 hope line gf triangle-standard egonet-standard kcore-standard pagerank-standard coloring-standard clique-standard identity
do
echo $alg-$init
python -u inductive.py --dataset data/$data  \
    --feature_size $feat_size \
    --init $init \
    --seed $seed \
    --cuda \
    $alg --self-loop > log/$alg/$data-$init-seed$seed
done # init


data=reddit
alg=graphsage
mkdir log/$alg
for init in ori ori-rowsum ori-standard degree-standard uniform deepwalk ssvd0.5 ssvd1 hope line gf triangle-standard egonet-standard kcore-standard pagerank-standard coloring-standard clique-standard identity
do
echo $alg-$init
python -u inductive.py --dataset data/$data  \
    --feature_size $feat_size \
    --init $init \
    --seed $seed \
    --cuda \
    $alg --aggregator mean > log/$alg/$data-$init-seed$seed
done # init

done
