seed=40
data=ppi
mkdir log/inductive-sgc
for init in ori degree uniform deepwalk ssvd0.5 ssvd1 hope triangle egonet pagerank coloring clique node2vec
do
python -u -m inductive.main \
    --data data/$data \
    --init $init \
    --feature_size 128 \
    --seed $seed \
    --norm_features > log/inductive-sgc/${data}-${init}-seed${seed}
done

for init in ori degree uniform deepwalk ssvd0.5 ssvd1 hope triangle egonet kcore pagerank coloring clique identity node2vec
