# @copyright CEA-LIST/DIASI/SIALV/LVA (2022)
# @author CEA-LIST/DIASI/SIALV/LVA <quentin.bouniot@cea.fr>
# @license CECILL

import torch

from utils import fix_seed

from mini_maml import test as test_maml
from mini_protonet import test as test_proto

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('Mini-Imagenet evaluation of Few Shot Learner')

    parser.add_argument('method', type=str)
    parser.add_argument('--filename', '-f', type=str,
        help='File name of checkpoints')

    parser.add_argument('--folder', type=str, default='/home/qbouniot/Documents/Data/Datasets/',
        help='Path to the folder the data is downloaded to.')
    parser.add_argument('--num-shots', type=int, default=1,
        help='Number of examples per class (k in "k-shot", default: 1).')
    parser.add_argument('--num-ways', type=int, default=5,
        help='Number of classes per task (N in "N-way", default: 5).')
    parser.add_argument('--num-query', type=int, default=15,
        help='Number of query images (test images) per task (default: 15).')
    parser.add_argument('--val-shots', type=int, default=1,
        help='Number of examples per class during validation (k in "k-shot", default: 1).')
    parser.add_argument('--dataset', type=str, default='mini',
        help='Dataset for training/testing (choice: mini/tiered, default: mini)')
    parser.add_argument('--model', type=str, default='conv4',
        help='Backbone model (choice: conv4/resnet, default: conv4) ')
    
    parser.add_argument('--embedding-size', type=int, default=64,
        help='Dimension of the embedding/latent space (default: 64).')
    parser.add_argument('--hidden-size', type=int, default=64,
        help='Number of channels for each convolutional layer (default: 64).')

    parser.add_argument('--output-folder', type=str, default='./checkpoints/',
        help='Path to the output folder for saving the model (optional).')
    parser.add_argument('--results-folder', type=str, default='./results/',
        help='Path to save the results of the evaluation')
    parser.add_argument('--save', action='store_true',
        help='Save checkpoints or not')
    parser.add_argument('--all_results', action='store_true',
        help='Print results for all episodes at meta-test time.')

    parser.add_argument('--s_ratio', action='store_true',
        help='Use the ratio of singular values with the symmetric matrix WW_t in the outer loss')
    parser.add_argument('--s_norm', action='store_true',
        help='Minimize the norm of singular values in the outer loss')
    parser.add_argument('--norm', action='store_true')
    parser.add_argument('--s_ent', action='store_true',
        help='Use entropy of singular values')
    parser.add_argument('--lbda', type=float, default=1,
        help="Loss regularization strength")
    
    parser.add_argument('--first-order', action='store_true',
        help='Use the first-order approximation of MAML.')
    parser.add_argument('--step-size', type=float, default=0.01,
        help='Step-size for the gradient step for adaptation (default: 0.01).')
    parser.add_argument('--weight-decay', type=float, default=0.001,
        help='Weight decay in the inner loop')
    parser.add_argument('--test-batch-size', type=int, default=4,
        help='Number of tasks in a testing batch (default: 4).')
    parser.add_argument('--batch-size', type=int, default=4,
        help='Number of tasks in a testing batch (default: 4).')
    parser.add_argument('--nb-test-task-update', type=int, default=10,
        help='Number of updates for each testing task (in inner loop)')
    parser.add_argument('--num-test-ep', type=int, default=600,
        help='Number of batches of episodes the model is tested over (defaul: 1000)')
    parser.add_argument('--num-workers', type=int, default=6,
        help='Number of workers for data loading (default: 1).')
    parser.add_argument('--download', action='store_true',
        help='Download the dataset in the data folder.')
    parser.add_argument('--use_cuda', action='store_true',
        help='Use CUDA if available.')
    parser.add_argument('--gpu', type=int, default=0,
        help='Which gpu to use if CUDA is enabled (default: 0)')
    parser.add_argument('--multi-gpu', action='store_true',
        help='Use multiple GPUs')
    parser.add_argument('--seed', type=int, default=None,
        help='Set the seed for the training and testing')

    parser.add_argument('--num-batches', type=int, default=0)
    parser.add_argument('--val-interval', type=int, default=0)

    args = parser.parse_args()
    args.device = torch.device(f'cuda:{args.gpu}' if args.use_cuda
        and torch.cuda.is_available() else 'cpu')

    if args.seed is not None:
        fix_seed(args.seed) 

    args.writer = None
    args.test_only = True

    if args.method == 'maml':
        if args.test_batch_size != args.batch_size:
            args.batch_size = args.test_batch_size
        test_maml(args)

    elif args.method == 'protonet':
        if args.test_batch_size != args.batch_size:
            args.batch_size = args.test_batch_size
        test_proto(args)
    