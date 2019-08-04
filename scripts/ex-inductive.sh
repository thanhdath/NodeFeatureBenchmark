seed=40
data=ppi
mkdir log/inductive-sgc-norm
for init in ori degree uniform deepwalk ssvd0.5 ssvd1 hope triangle egonet pagerank coloring clique node2vec
do
python -u -m inductive.main \
    --data data/$data \
    --init $init --norm_features > log/inductive-sgc-norm/${data}-${init}-seed${seed}
done

for init in ori degree uniform deepwalk ssvd0.5 ssvd1 hope triangle egonet kcore pagerank coloring clique identity node2vec
