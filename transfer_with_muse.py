import argparse
import numpy as np
import networkx as nx
from dataloader import DefaultDataloader, CitationDataloader, RedditDataset, NELLDataloader
from features_init import lookup as lookup_feature_init
import torch
import random
from dgl.data import citation_graph as citegrh
from parser import *
from algorithms.node_embedding import SGC, Nope, DGIAPI, GraphsageAPI
from algorithms.node_embedding.graphsagetf.api import Graphsage
from algorithms.logistic_regression import LogisticRegressionPytorch
import os
from sklearn.decomposition import PCA
from muse.evaluation import Evaluator
from muse.models import build_model
from muse.trainer import Trainer
from muse.utils import initialize_exp
from types import SimpleNamespace

def parse_args():
    parser = argparse.ArgumentParser(
        description="Node feature initialization benchmark.")
    parser.add_argument('--dataset', default="data/cora")
    parser.add_argument('--init', default="ori")
    parser.add_argument('--feature_size', default=128, type=int)
    parser.add_argument('--learn-features', dest='learnable_features',
                        action='store_true')
    parser.add_argument('--shuffle', action='store_true',
                        help="Whether shuffle features or not.")
    parser.add_argument('--seed', type=int, default=40)
    parser.add_argument('--verbose', type=int, default=1)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--pca', action='store_true', 
        help="Whether to reduce feature dimention to feature_size. Use for original features, identity features.")

    # for logistic regression
    parser.add_argument('--logreg-bias', action='store_true',
                        dest='logreg_bias', help="Whether use bias in logistic regression or not.")
    parser.add_argument('--logreg-wc', dest='logreg_weight_decay', type=float,
                        default=5e-6, help="Weight decay for logistic regression.")
    parser.add_argument('--logreg-epochs',
                        dest='logreg_epochs', default=300, type=int)

    subparsers = parser.add_subparsers(dest="alg",
                                       help='Choose 1 of the GNN algorithm from: sgc, dgi, graphsage, nope.')
    add_sgc_parser(subparsers)
    add_nope_parser(subparsers)
    add_dgi_parser(subparsers)
    add_graphsage_parser(subparsers)
    return parser.parse_args()


def get_algorithm(args, data, features):
    if args.alg == "sgc":
        return SGC(data, features, degree=args.degree, cuda=args.cuda)
    elif args.alg == "nope":
        return Nope(features)
    elif args.alg == "dgi":
        return DGIAPI(data, features, self_loop=args.self_loop, cuda=args.cuda,
                      learnable_features=args.learnable_features, 
                      epochs=args.epochs,
                      suffix="{}-{}-{}".format(args.dataset.split('/')[-1], args.init, args.seed),
                      load_model=args.load_model)
    elif args.alg == "graphsage":
        if features.shape[0] > 60000:
            return Graphsage(data, features, max_degree=args.max_degree, samples_1=args.samples_1)
        else:
            return GraphsageAPI(data, features, cuda=args.cuda, 
                                aggregator=args.aggregator,
                                learnable_features=args.learnable_features, 
                                suffix="{}-{}-{}".format(args.dataset.split('/')[-1], args.init, args.seed),
                                load_model=args.load_model)
    else:
        raise NotImplementedError


def add_weight(subgraph):
    for n1, n2 in subgraph.edges():
        subgraph[n1][n2]['weight'] = 1
    return subgraph


def get_feature_initialization(args, data, inplace=True, input_graph=False):
    if input_graph:
        graph = data
    else:
        graph = data.graph
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
        kwargs = {"feature_path": args.dataset+"/features.npz"}
    elif init == "label":
        kwargs = {"label_path": args.dataset+"/labels.npz"}
    elif init == "ssvd0.5":
        init = "ssvd"
        kwargs = {"alpha": 0.5}
    elif init == "ssvd1":
        init = "ssvd"
        kwargs = {"alpha": 1}
    elif init in ["node2vec"]:
        add_weight(graph)

    if "reddit" in args.dataset:
        if init == "deepwalk":
            graph.build_neibs_dict()
        elif init in "pagerank triangle kcore".split():
            kwargs = {"use_networkit": True}
            graph = data.graph_networkit()
            inplace = False
            print("Warning: Init using {} will set inplace = False".format(init))
        elif init in "egonet coloring clique graphwave".split():
            graph = data.graph_networkx()
            inplace = False
            print("Warning: Init using {} will set inplace = False".format(init))

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
    if data_name in ["citeseer", "pubmed"]:
        return CitationDataloader(dataset)
    elif data_name == "reddit":
        return RedditDataset(self_loop=False)
    elif data_name == "reddit_self_loop":
        return RedditDataset(self_loop=True)
    elif data_name == "NELL":
        return NELLDataloader(dataset)
    else:
        # cora bc flickr wiki youtube homo-sapiens
        return DefaultDataloader(dataset)


def main(args):
    data = load_data(args.dataset)
    inplace = "reddit" not in args.dataset

    inits_one = "degree-standard triangle-standard kcore-standard egonet-standard clique-standard coloring-standard".split()
    if args.init in inits_one:
        load_seed = 40
    else:
        load_seed = args.seed

    feat_file = 'feats/{}-{}-seed{}-dim{}.npz'.format(args.dataset.split('/')[-1], args.init,
                                                      load_seed, args.feature_size)

    if args.shuffle:
        features = get_feature_initialization(args, data, inplace=inplace)
    else:
        if os.path.isfile(feat_file):
            features = np.load(feat_file, allow_pickle=True)['features'][()]
        else:
            features = get_feature_initialization(args, data, inplace=inplace)
            if not os.path.isdir('feats'):
                os.makedirs('feats')
            if args.init not in ["identity"]:
                np.savez_compressed(feat_file, features=features)
    # features = dict2arr(features, data.graph)

    # inits_fixed_dim = "ori label identity".split()
    # init, norm = (args.init+"-0").split("-")[:2]
    # if args.pca and init in inits_fixed_dim:
    #     print("Perform PCA to reduce feature dimention from {} to {}.".format(features.shape[1], args.feature_size))
    #     pca = PCA(n_components=args.feature_size)
    #     pca.fit(features.numpy())
    #     features = torch.FloatTensor(pca.transform(features.numpy()))

    # align features to load model feature space
    if hasattr(args, 'load_model') and args.load_model is not None:
        from_data = args.load_model.split("-")[3]
        from_init = args.load_model.split("-")[4]
        # from_seed = args.load_model.split("-")[5]
        from_features_file = 'feats/{}-{}-seed{}-dim128.npz'.format(from_data, from_init, load_seed)
        from_features = np.load(from_features_file, allow_pickle=True)['features'][()]
        dataname = args.dataset.split('/')[-1]
        anchors_file = "anchors-{}-{}.dict".format(dataname, from_data)
        print("Aligned features from {} to {} space".format(dataname, from_data))
        features, from_features = aligned_emb(features, from_features, anchors_file, args.feature_size)
    features = dict2arr(features, data.graph)

    alg = get_algorithm(args, data, features)

    embeds = alg.train()

    if args.alg in ["sgc", "dgi", "nope"]:
        print("Using default logistic regression")
        torch.cuda.empty_cache()
        classifier = LogisticRegressionPytorch(embeds,
                                               data.labels, data.train_mask, data.val_mask, data.test_mask,
                                               epochs=args.logreg_epochs, weight_decay=args.logreg_weight_decay,
                                               bias=args.logreg_bias, cuda=args.cuda,
                                               multiclass=data.multiclass)


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

def aligned_emb(src_emb, trg_emb, anchors_file, dim_size):
    # write_dict("train.dict", anchors, ".")
    save_embeddings(src_emb, "embsrc", None, ".")
    save_embeddings(trg_emb, "embbase", None, ".")
    src_emb, tgr_emb = map_emb(exp_id="src", exp_name="src", src_lang="embsrc",
            tgt_lang="embbase",
            dico_train=anchors_file,
            dico_eval="val.dict",
            src_emb="embsrc.pth",
            tgt_emb="embbase.pth", 
            emb_dim=dim_size,
            cuda=True)
    return src_emb, trg_emb

# def write_dict(filename, anchors, log_dir):
#     if not os.path.exists(log_dir):
#         os.makedirs(log_dir)
#     with open(log_dir + '/' + filename, 'w') as f:
#         for n1, n2 in anchors:
#             f.write(str(anchor) + " " + str(anchor) + "\n")

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
    if args.dataset.endswith("/"):
        args.dataset = args.dataset[:-1]
    print(args)
    main(args)
