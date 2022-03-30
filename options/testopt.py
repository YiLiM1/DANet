import argparse

def _get_test_opt():
    parser = argparse.ArgumentParser(description = 'Evaluate performance of DANet')
    parser.add_argument('--backbone', default='EfficientNet', help='select a network as backbone')
    parser.add_argument('--testlist_path',default='./data/nyu2_test.csv',required=False, help='the path of testlist')
    parser.add_argument('--batch_size', type=int, default=1, help='testing batch size')
    parser.add_argument('--root_path', default='./',required=False, help="the root path of dataset")
    parser.add_argument('--loadckpt', default='./results/Efficientnet_best.pt',required=False, help="the path of the loaded model")
    parser.add_argument('--threshold', type=float, default=1.0, help="threshold of the pixels on edges")
    parser.add_argument('--pretrained_dir', type=str,default='./pretrained', required=False, help="the path of pretrained models")
    parser.add_argument('--max_depth', type=float, help='maximum depth in estimation', default=10)
    parser.add_argument('--min_depth', type=float, help='minimum depth in estimation', default=1e-3)
    parser.add_argument('--height', type=float, help='feature height', default=8)
    parser.add_argument('--width', type=float, help='feature width', default=10)
    # parse arguments
    return parser.parse_args()
