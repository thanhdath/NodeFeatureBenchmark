
mkdir logs 
mkdir logs/graph-from-features
data=NELL
for i in knn sigmoid
do 
echo $i
    # python -u main.py --dataset temp/$data-gen-$i \
    #     --init ori --seed 40 --cuda sgc > logs/graph-from-features/sgc-$data-$i.log
    # python -u main.py --dataset temp/$data-gen-$i \
    #     --init ori --seed 40 --cuda dgi > logs/graph-from-features/dgi-$data-$i.log
    python -u main.py --dataset temp/$data-gen-$i \
        --init ori --seed 40 --cuda graphsage --aggregator mean > logs/graph-from-features/graphsage-$data-$i.log
done 


mkdir logs 
mkdir logs/graph-from-features
for data in NELL
do
    for i in knn sigmoid
    do 
    echo $i
        python -u main.py --dataset temp/$data-gen-$i \
            --init ori --seed 40 --cuda gat > logs/graph-from-features/gat-$data-$i.log
    done 
done 

mkdir logs/gat
for data in NELL
do
    for i in ori deepwalk hope
    do 
    echo $i
        python -u main.py --dataset data/$data \
            --init $i --seed 40 --cuda gat > logs/gat/$data-$i.log
    done 
done 
