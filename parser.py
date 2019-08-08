
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