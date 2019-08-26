feat_size=128
mkdir log
for data in homo-sapiens 
do
    for seed in $(seq 40 42)
    do
        alg=nope
        mkdir log/$lag
        for init in deepwalk hope line gf
        do
        echo $alg-$init
        python -u main.py --dataset data/$data  \
            --feature_size $feat_size \
            --init $init \
            --seed $seed \
            --cuda --logreg-epochs 300 \
            $alg > log/$alg/$data-$init-seed$seed
        done # init
    done

    for seed in $(seq 40 42)
    do
        alg=sgc
        mkdir log/$alg
        for init in degree-standard uniform deepwalk ssvd0.5 ssvd1 hope line gf triangle-standard kcore-standard egonet-standard pagerank-standard coloring-standard clique-standard identity
        do
        echo $alg-$init
        python -u main.py --dataset data/$data  \
            --feature_size $feat_size \
            --init $init \
            --seed $seed \
            --cuda --logreg-epochs 300 \
            $alg > log/$alg/$data-$init-seed$seed
        done # init
    done

    for seed in $(seq 40 42)
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
            --cuda --logreg-epochs 300 \
            $alg --self-loop > log/$alg/$data-$init-seed$seed
        done # init
    done

    for seed in $(seq 40 42)
    do
        alg=graphsage
        mkdir log/$alg
        for init in ssvd0.5 ssvd1 hope line gf triangles-standard kcore-standard egonet-standard pagerank-standard coloring-standard clique-standard deepwalk
        do
        echo $alg-$init
        python -u main.py --dataset data/$data  \
            --feature_size $feat_size \
            --init $init \
            --seed $seed \
            --cuda --logreg-epochs 300 \
            $alg --aggregator mean > log/$alg/$data-$init-seed$seed
        done # init
    done

done
