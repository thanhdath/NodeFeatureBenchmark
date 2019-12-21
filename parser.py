
def add_sgc_parser(subparsers):
    parser = subparsers.add_parser('sgc', help='SGC algorithm')
    parser.add_argument('--degree', default=2, type=int,
                        help="aggregate k-hop neighbors.")
    return parser


def add_nope_parser(subparsers):
    parser = subparsers.add_parser('nope', help='Raw features only.')
    return parser


def add_dgi_parser(subparsers):
    parser = subparsers.add_parser('dgi', help='DGI algorithm.')
    parser.add_argument('--self-loop', dest='self_loop', action='store_true',
                        help="Whether add self loop or not.")
    parser.add_argument('--load-model', dest='load_model',
                        help="Path to pretrain embeds model.")
    parser.add_argument('--epochs', type=int, default=300,
                        help="Number of epochs for training embeds.")
    return parser


def add_graphsage_parser(subparsers):
    parser = subparsers.add_parser('graphsage', help='DGI algorithm.')
    parser.add_argument('--aggregator', default="mean",
                        help="Aggregator type (mean or pooling)")
    parser.add_argument('--load-model', dest='load_model',
                        help="Path to pretrain embeds model.")
    parser.add_argument('--max_degree', default=25, type=int,
                        help="Max degree for neighbors sampling.")
    parser.add_argument('--samples_1', default=25, type=int, help="")
    return parser

def add_gat_parser(subparsers):
    parser = subparsers.add_parser('gat', help='GAT algorithm.')
    parser.add_argument('--num-heads', default=8, type=int)
    parser.add_argument('--num-layers', default=1, type=int)
    parser.add_argument('--num-out-heads', default=1, type=int)
    parser.add_argument('--num-hidden', default=8, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    return parser

def add_diffpool_parser(subparsers):
    parser = subparsers.add_parser('diffpool', help='Diffpool algorithm.')
    parser.add_argument('--pool_ratio', type=float, default=0.1)
    parser.add_argument('--num_pool', type=int, default=1)
    return parser


def add_gin_parser(subparsers):
    parser = subparsers.add_parser('gin', help='Diffpool algorithm.')
    parser.add_argument('--graph_pooling_type', default="sum",
                        choices=["sum", "mean", "max"])
    parser.add_argument('--neighbor_pooling_type', default="sum",
                        choices=["sum", "mean", "max"])
    parser.add_argument('--learn_eps', action="store_true",
                        help='learn the epsilon weighting')
    parser.add_argument('--degree_as_tag', action="store_true",
                        help='take the degree of nodes as input feature')
    return parser


def add_simple_graph_emb_parse(subparsers):
    parser = subparsers.add_parser('simple', help='Simple graph embedding algorithm.')
    parser.add_argument('--operator', default="sum", choices=["sum", "mean", "max"])
    parser.add_argument('--l2_norm', action='store_true')
    return parser
