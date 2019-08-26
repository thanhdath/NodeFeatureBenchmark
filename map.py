import argparse
import numpy as np
import networkx as nx
from dataloader import PPIDataset, RedditInductiveDataset, CitationInductiveDataloader, DefaultInductiveDataloader
from features_init import lookup as lookup_feature_init
import torch
import random
from dgl.data import citation_graph as citegrh
from parser import *
from algorithms.node_embedding import SGC, DGIAPI, GraphsageInductive
from algorithms.logreg_inductive import LogisticRegressionInductive
import os
from muse.evaluation import Evaluator
from muse.models import build_model
from muse.trainer import Trainer
from muse.utils import initialize_exp
from types import SimpleNamespace

def parse_args():
    parser = argparse.ArgumentParser(
        description="Node feature initialization benchmark.")
    parser.add_argument('--dataset', default="data/ppi")
    parser.add_argument('--init', default="ori")
    parser.add_argument('--feature_size', default=128, type=int)
    # args.add_argument('--train_features', action='store_true')
    parser.add_argument('--shuffle', action='store_true',
                        help="Whether shuffle features or not.")
    parser.add_argument('--seed', type=int, default=40)
    parser.add_argument('--verbose', type=int, default=1)
    parser.add_argument('--cuda', action='store_true')

    # for logistic regression
    parser.add_argument('--logreg-bias', action='store_true',
                        dest='logreg_bias', help="Whether use bias in logistic regression or not.")
    parser.add_argument('--logreg-wc', dest='logreg_weight_decay', type=float,
                        default=5e-6, help="Weight decay for logistic regression.")
    parser.add_argument('--logreg-epochs',
                        dest='logreg_epochs', default=200, type=int)

    subparsers = parser.add_subparsers(dest="alg",
                                       help='Choose 1 of the GNN algorithm from: sgc, dgi, graphsage, nope.')
    add_sgc_parser(subparsers)
    add_nope_parser(subparsers)
    add_dgi_parser(subparsers)
    add_graphsage_parser(subparsers)
    return parser.parse_args()


def get_algorithm(args, train_data, train_features, val_data=None, val_features=None, 
    test_data=None, test_features=None):
    if args.alg == "sgc":
        return SGC(train_data, train_features, degree=args.degree, cuda=args.cuda)
    elif args.alg == "dgi":
        return DGIAPI(train_data, train_features, cuda=args.cuda)
    elif args.alg == "graphsage":
        return GraphsageInductive(train_data, val_data, test_data, train_features, val_features,
            test_features, cuda=args.cuda, aggregator=args.aggregator)
    else:
        raise NotImplementedError

def add_weight(subgraph):
    for n1, n2 in subgraph.edges():
        subgraph[n1][n2]['weight'] = 1
    return subgraph

def get_feature_initialization(args, graph, mode, inplace=True):
    elms = args.init.split("-")
    if len(elms) < 2:
        init = elms[0]
        normalizer = "pass"
    else:
        init, normalizer = elms[:2]
    if init not in lookup_feature_init:
        raise NotImplementedError
    kwargs = {}
    if init == "ori":
        kwargs = {"feature_path": args.dataset+"/{}_feats.npy".format(mode)}
    elif init == "ssvd0.5":
        init = "ssvd"
        kwargs = {"alpha": 0.5}
    elif init == "ssvd1":
        init = "ssvd"
        kwargs = {"alpha": 1}
    elif init in ["gf", "node2vec"]:
        add_weight(graph)

    if "reddit" in args.dataset and init == "deepwalk":
        graph.build_neibs_dict()

    init_feature = lookup_feature_init[init](**kwargs)
    return init_feature.generate(graph, args.feature_size,
                                 inplace=inplace, normalizer=normalizer, verbose=args.verbose,
                                 shuffle=args.shuffle)


def dict2arr(dictt, graph):
    """ 
    Note: always sort graph nodes
    """
    dict_arr = torch.FloatTensor([dictt[int(x)] for x in graph.nodes()])
    return dict_arr


def load_data(dataset):
    data_name = dataset.split('/')[-1]
    if data_name == "ppi":
        return PPIDataset("train"), PPIDataset("valid"), PPIDataset("test")
    elif "reddit" in data_name:
        return (RedditInductiveDataset("train", self_loop=("self_loop" in data_name)), 
            RedditInductiveDataset("valid", self_loop=("self_loop" in data_name)), 
            RedditInductiveDataset("test", self_loop=("self_loop" in data_name)))
    elif data_name in "citeseer pubmed".split():
        return (CitationInductiveDataloader(dataset, "train"), 
            CitationInductiveDataloader(dataset, "valid"), 
            CitationInductiveDataloader(dataset, "test"), )
    elif data_name in "cora".split():
        return (DefaultInductiveDataloader(dataset, "train"),
            DefaultInductiveDataloader(dataset, "valid"),
            DefaultInductiveDataloader(dataset, "test"))

def load_features(mode, graph, args):
    inits_one = "ori ori-rowsum ori-standard degree-standard triangle-standard kcore-standard egonet-standard clique-standard coloring-standard".split()
    if args.init in inits_one:
        load_seed = 40
    else:
        load_seed=  args.seed

    feat_file = 'feats/{}-{}-{}-seed{}.npz'.format(args.dataset.split('/')[-1], 
        mode, args.init, load_seed)
    if args.shuffle:
        features = get_feature_initialization(args, graph, mode, inplace=False)
    else:
        if os.path.isfile(feat_file):
            features = np.load(feat_file, allow_pickle=True)['features'][()]
        else:
            print(feat_file, "not found ")
            features = get_feature_initialization(args, graph, mode, inplace=False)
            if not os.path.isdir('feats'):
                os.makedirs('feats')
            try:
                np.savez_compressed(feat_file, features=features)
            except: 
                pass
    features = dict2arr(features, graph)
    return features

def main(args):
    train_data, val_data, test_data = load_data(args.dataset)
    train_features = load_features('train', train_data.graph, args)
    val_features = load_features('valid', val_data.graph, args)
    test_features = load_features('test', test_data.graph, args)

    # 
    anchors = list(train_data.graph.nodes() )
    train_features_dict = {node: train_features[i].numpy() for i,node in enumerate(train_data.graph.nodes())}
    val_features_dict = {node: val_features[i].numpy() for i,node in enumerate(val_data.graph.nodes())}
    test_features_dict = {node: test_features[i].numpy() for i,node in enumerate(test_data.graph.nodes())}
    val_features_dict, train_features_dict = aligned_emb(val_features_dict, train_features_dict, anchors, args.feature_size)
    test_features_dict, train_features_dict = aligned_emb(test_features_dict, train_features_dict, anchors, args.feature_size)

    val_features = dict2arr(val_features_dict, val_data.graph)
    test_features = dict2arr(test_features_dict, test_data.graph)


    use_default_classifier = False
    if args.alg == "sgc":
        # aggregate only -> create train val test alg
        train_alg = get_algorithm(args, train_data, train_features) 
        train_embs = train_alg.train()
        val_alg = get_algorithm(args, val_data, val_features)
        val_embs = val_alg.train()[val_data.mask]
        test_alg = get_algorithm(args, test_data, test_features)
        test_embs = test_alg.train()[test_data.mask]
        val_labels = val_data.labels
        test_labels = test_data.labels
        use_default_classifier = True
    elif args.alg == "dgi":
        alg = get_algorithm(args, train_data, train_features)
        train_embs = alg.train()
        torch.cuda.empty_cache()
        val_embs = alg.get_embeds(val_features, val_data.graph)[val_data.mask]
        torch.cuda.empty_cache()
        test_embs = alg.get_embeds(test_features, test_data.graph)[test_data.mask]
        val_labels = val_data.labels
        test_labels = test_data.labels
        torch.cuda.empty_cache()
        use_default_classifier = True
    elif args.alg == "graphsage":
        alg = get_algorithm(args, train_data, train_features, val_data, val_features,
            test_data, test_features)
        alg.train()

    if use_default_classifier:
        print("Using default logistic regression")
        classifier = LogisticRegressionInductive(train_embs, val_embs, test_embs, 
            train_data.labels, val_labels, test_labels,
            epochs=args.logreg_epochs, weight_decay=args.logreg_weight_decay,
            bias=args.logreg_bias, cuda=args.cuda, 
            multiclass=train_data.multiclass)

def init_environment(args):
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    random.seed(args.seed)


def save_embeddings(vectors, lang, size, log_dir):
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    from muse.dictionary import Dictionary
    embeddings = []
    word2id = {}
    for i, word in enumerate(vectors.keys()):
        embeddings.append(vectors[word])
        word2id[word] = i
    embeddings = torch.FloatTensor(embeddings)
    id2word = {i: w for w, i in word2id.items()}
    dico = Dictionary(id2word, word2id, lang)
    file_path = log_dir + '/' + lang + '.pth'
    torch.save({'dico': dico, 'vectors': embeddings}, file_path)
    return file_path

def aligned_emb(src_emb, trg_emb, anchors, dim_size):
    write_dict("train.dict", anchors, ".")
    save_embeddings(src_emb, "embsrc", None, ".")
    save_embeddings(trg_emb, "embbase", None, ".")
    src_emb, tgr_emb = map_emb(exp_id="src", exp_name="src", src_lang="embsrc",
            tgt_lang="embbase",
            dico_train="train.dict",
            dico_eval="val.dict",
            src_emb="embsrc.pth",
            tgt_emb="embbase.pth", 
            emb_dim=dim_size,
            cuda=True)
    return src_emb, trg_emb

def write_dict(filename, anchors, log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    with open(log_dir + '/' + filename, 'w') as f:
        for anchor in anchors:
            f.write(str(anchor) + " " + str(anchor) + "\n")

def map_emb(seed=42, verbose=1, exp_path="", exp_name="test", exp_id="test", cuda=False,
            src_lang='emb0', tgt_lang='emb1', max_vocab=-1, n_refinement=3,
            dico_train="cora_train.dict", dico_eval="cora_val.dict",
            dico_method='csls_knn_10', dico_build="S2T&T2S",
            dico_threshold=0, dico_max_rank=0, dico_min_size=0, dico_max_size=0,
            src_emb="cora_emb_space0.emb", tgt_emb="cora_emb_space1.emb",
            emb_dim=128, normalize_embeddings=""):
    p = {}
    p['seed'] = seed
    p['verbose'] = verbose
    p['exp_path'] = exp_path
    p['exp_name'] = exp_name
    p['exp_id'] = exp_id
    p['cuda'] = cuda
    p['src_lang'] = src_lang
    p['tgt_lang'] = tgt_lang
    p['max_vocab'] = max_vocab
    p['n_refinement'] = n_refinement
    p['dico_train'] = dico_train
    p['dico_eval'] = dico_eval
    p['dico_method'] = dico_method
    p['dico_build'] = dico_build
    p['dico_threshold'] = dico_threshold
    p['dico_max_rank'] = dico_max_rank
    p['dico_min_size'] = dico_min_size
    p['dico_max_size'] = dico_max_size
    p['src_emb'] = src_emb
    p['tgt_emb'] = tgt_emb
    p['emb_dim'] = emb_dim
    p['normalize_embeddings'] = normalize_embeddings
    params = SimpleNamespace(**p)

    VALIDATION_METRIC_SUP = 'precision_at_1-csls_knn_10'
    VALIDATION_METRIC_UNSUP = 'mean_cosine-csls_knn_10-S2T-10000'
    logger = initialize_exp(params)
    src_emb, tgt_emb, mapping, _ = build_model(params, False)
    trainer = Trainer(src_emb, tgt_emb, mapping, None, params)
    evaluator = Evaluator(trainer)
    trainer.load_training_dico(params.dico_train)
    VALIDATION_METRIC = VALIDATION_METRIC_UNSUP if params.dico_train == 'identical_char' else VALIDATION_METRIC_SUP
    logger.info("Validation metric: %s" % VALIDATION_METRIC)

    """
    Learning loop for Procrustes Iterative Learning
    """
    for n_iter in range(params.n_refinement + 1):
        logger.info('Starting iteration %i...' % n_iter)
        trainer.procrustes()
        logger.info('End of iteration %i.\n\n' % n_iter)

    mapped_src, mapped_tgt = trainer.export(save=False)
    src_emb = {params.src_dico[i]: mapped_src[i].tolist() for i in range(len(params.src_dico))}
    tgt_emb = {params.tgt_dico[i]: mapped_tgt[i].tolist() for i in range(len(params.tgt_dico))}
    return src_emb, tgt_emb

if __name__ == '__main__':
    args = parse_args()
    init_environment(args)
    print(args)
    main(args)
