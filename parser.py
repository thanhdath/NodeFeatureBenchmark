
def add_sgc_parser(subparsers):
    parser = subparsers.add_parser('sgc', help='SGC algorithm')
    parser.add_argument('--degree', default=2, type=int, help="aggregate k-hop neighbors.")
    return parser

def add_nope_parser(subparsers):
    parser = subparsers.add_parser('nope', help='Raw features only.')
    return parser

def add_dgi_parser(subparsers):
    parser = subparsers.add_parser('dgi', help='DGI algorithm.')
    parser.add_argument('--self-loop', dest='self_loop', action='store_true',
        help="Whether add self loop or not.")
    return parser

def add_graphsage_parser(subparsers):
    parser = subparsers.add_parser('graphsage', help='DGI algorithm.')
    parser.add_argument('--aggregator', help="Aggregator type (mean or pool)")
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
