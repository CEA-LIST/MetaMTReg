# @copyright CEA-LIST/DIASI/SIALV/LVA (2022)
# @author CEA-LIST/DIASI/SIALV/LVA <quentin.bouniot@cea.fr>
# @license CECILL
 
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from tqdm import tqdm
import numpy as np

from datasets.cropdisease import CropDisease

from torchmeta.datasets import MiniImagenet, TieredImagenet
from torchmeta.datasets.helpers import helper_with_default, omniglot
from torchmeta.utils.data import BatchMetaDataLoader
from torchmeta.utils.gradient_based import gradient_update_parameters

from torch.utils.tensorboard import SummaryWriter

from models.Maml import ConvolutionalNeuralNetwork
from utils import fix_seed, get_accuracy, save_model_best_acc, load_model, save_model

def train(args):

    if args.dataset == 'mini':
        train_dataset = helper_with_default(MiniImagenet,
                                    folder=args.folder,
                                    shots=args.num_shots,
                                    ways=args.num_ways,
                                    shuffle=True,
                                    test_shots=15,
                                    meta_train=True,
                                    download=args.download,
                                    seed=args.seed,
                                    transform=Compose([
                                        Resize((84,84)),
                                        ToTensor(),
                                        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                                    ]))
    elif args.dataset == 'tiered':

        train_dataset = helper_with_default(TieredImagenet,
                                    folder=args.folder,
                                    shots=args.num_shots,
                                    ways=args.num_ways,
                                    shuffle=True,
                                    test_shots=15,
                                    meta_train=True,
                                    download=args.download,
                                    seed=args.seed,
                                    transform=Compose([
                                        Resize((84,84)),
                                        ToTensor(),
                                        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                                    ]))
    elif args.dataset=='omni':
        train_dataset = omniglot(args.folder,
                                    shots=args.num_shots,
                                    ways=args.num_ways,
                                    shuffle=True,
                                    test_shots=15,
                                    meta_train=True,
                                    seed=args.seed,
                                    download=args.download)

    train_dataloader = BatchMetaDataLoader(train_dataset,
                                     batch_size=args.batch_size,
                                     shuffle=True,
                                     num_workers=args.num_workers,
                                     pin_memory=True)

    if args.dataset=='mini':
        val_dataset = helper_with_default(MiniImagenet,
                                        folder=args.folder,
                                        shots=args.num_shots,
                                        ways=args.num_ways,
                                        shuffle=True,
                                        test_shots=15,
                                        meta_val=True,
                                        download=args.download,
                                        seed=args.seed,
                                        transform=Compose([
                                            Resize((84,84)),
                                            ToTensor(),
                                            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                                        ]))
    elif args.dataset == 'tiered':
        val_dataset = helper_with_default(TieredImagenet,
                                        folder=args.folder,
                                        shots=args.num_shots,
                                        ways=args.num_ways,
                                        shuffle=True,
                                        test_shots=15,
                                        meta_val=True,
                                        download=args.download,
                                        seed=args.seed,
                                        transform=Compose([
                                            Resize((84,84)),
                                            ToTensor(),
                                            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                                        ]))

    elif args.dataset == 'omni':
        val_dataset = omniglot(folder=args.folder,
                                shots=args.num_shots,
                                ways=args.num_ways,
                                shuffle=True,
                                test_shots=15,
                                meta_val=True,
                                seed=args.seed,
                                download=args.download)
    
    val_dataloader = BatchMetaDataLoader(val_dataset,
                                    batch_size=args.batch_size,
                                    shuffle=True,
                                    num_workers=args.num_workers,
                                    pin_memory=True)

    if args.dataset == 'omni':
        in_channels = 1
    else:
        in_channels = 3

    model = ConvolutionalNeuralNetwork(in_channels,
                                    int(args.num_ways),
                                    hidden_size=args.hidden_size,
                                    embedding_size=args.embedding_size)
    model.to(device=args.device)
    model = model.train()
    meta_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_acc_mean = 0
    acc_mean = 0
    acc_std = 0

    writer = SummaryWriter(comment= '_' + args.filename)
    for arg in vars(args):
        writer.add_text(str(arg), str(getattr(args,arg)))

    # Training loop
    with tqdm(train_dataloader, total=args.num_batches) as pbar:
        for batch_idx, batch in enumerate(pbar):
            if batch_idx >= args.num_batches:
                break
            
            model.zero_grad()

            train_inputs, train_targets = batch['train']
            train_inputs = train_inputs.to(device=args.device)
            train_targets = train_targets.to(device=args.device)

            test_inputs, test_targets = batch['test']
            test_inputs = test_inputs.to(device=args.device)
            test_targets = test_targets.to(device=args.device)

            outer_loss = torch.tensor(0., device=args.device)
            accuracy = torch.tensor(0., device=args.device)
            this_ratio = None
            for task_idx, (train_input, train_target, test_input,
                    test_target) in enumerate(zip(train_inputs, train_targets,
                    test_inputs, test_targets)):

                for i in range(args.nb_task_update):
                    if i==0:
                        train_logit = model(train_input)
                        params = None
                    else:
                        train_logit = model(train_input, params=params)
                    inner_loss = F.cross_entropy(train_logit, train_target)

                    model.zero_grad()
                    params = gradient_update_parameters(model,
                                                        inner_loss,
                                                        params=params,
                                                        step_size=args.step_size,
                                                        first_order=args.first_order)

                test_logit = model(test_input, params=params)
                outer_loss += F.cross_entropy(test_logit, test_target)
                
                if args.s_ratio or args.s_norm:
                    linear_weights = model.classifier.weight
                    
                    u,s,v = torch.svd(torch.matmul(linear_weights, linear_weights.t()))
                    this_ratio = s[0] / s[-1]
                    s_norm = torch.norm(s).to(args.device)
                    w_norm = torch.mean(torch.norm(linear_weights, dim=1))
                    if args.s_ratio:
                        outer_loss.add_(args.lbda1*this_ratio)
                    if args.s_norm:
                        outer_loss.add_(args.lbda2*s_norm)


                with torch.no_grad():
                    accuracy += get_accuracy(test_logit, test_target)
                    if this_ratio is None:
                        linear_weights = model.classifier.weight
                        u,s,v = torch.svd(torch.matmul(linear_weights, linear_weights.t()))
                        this_ratio = s[0] / s[-1]
                        s_norm = torch.norm(s)
                        w_norm = torch.mean(torch.norm(linear_weights, dim=1))

            outer_loss.div_(args.batch_size)
            accuracy.div_(args.batch_size)

            outer_loss.backward()
            meta_optimizer.step()

            log = {
                'accuracy': f'{accuracy.item():.4f}',
                'val_mean': f'{acc_mean:.4f}',
                'val_std': f'{acc_std:.4f}'
            }

            pbar.set_postfix(log)
            writer.add_scalar('Training/Accuracy', accuracy.item(), batch_idx)
            if this_ratio is not None:
                writer.add_scalar('Training/last_ratio', this_ratio.item(), batch_idx)
                writer.add_scalar('Training/si_max', s[0].item(), batch_idx)
                writer.add_scalar('Training/si_min', s[-1].item(), batch_idx)
                writer.add_scalar('Training/s_norm', s_norm.item(), batch_idx)
                writer.add_scalar('Training/w_norm', w_norm.item(), batch_idx)

            if batch_idx%args.val_interval == 0 and batch_idx > 0:
                model = model.eval()
                acc_all = []
                with tqdm(val_dataloader, total=min(len(val_dataloader), args.num_test_ep)) as val_pbar:
                    for val_idx, val_batch in enumerate(val_pbar):
                        if val_idx >= args.num_test_ep:
                            break

                        support_inputs, support_targets = val_batch['train']
                        support_inputs = support_inputs.to(device=args.device)
                        support_targets = support_targets.to(device=args.device)

                        query_inputs, query_targets = val_batch['test']
                        query_inputs = query_inputs.to(device=args.device)
                        query_targets = query_targets.to(device=args.device)
                        
                        acc = torch.tensor(0., device=args.device)
                        for task_idx, (support_input, support_target, query_input,
                                query_target) in enumerate(zip(support_inputs, support_targets,
                                query_inputs, query_targets)):


                            for i in range(args.nb_test_task_update):
                                if i==0:
                                    support_logit = model(support_input)
                                    params = None
                                else:
                                    support_logit = model(support_input, params=params)
                            
                                inner_loss = F.cross_entropy(support_logit, support_target)
                                
                                model.zero_grad()
                                params = gradient_update_parameters(model,
                                                                    inner_loss,
                                                                    params=params,
                                                                    step_size=args.step_size,
                                                                    first_order=args.first_order)

                            with torch.no_grad():
                                query_logit = model(query_input, params=params)
                                acc_this = get_accuracy(query_logit, query_target)
                            acc_all.append(acc_this)
                            acc += acc_this
                        
                        acc.div_(args.batch_size)
                        
                        val_pbar.set_postfix(accuracy='{0:.4f}'.format(acc.item()))
                acc_all = torch.stack(acc_all)
                acc_mean = torch.mean(acc_all)
                acc_std  = torch.std(acc_all)
                writer.add_scalar('Val/Mean accuracy', acc_mean.item(), batch_idx)
                writer.add_scalar('Val/Std accuracy', acc_std.item(), batch_idx)
                model = model.train()

                # Save model
                if args.output_folder is not None:
                    if args.save_all:
                        save_model(acc_mean, model, meta_optimizer, batch_idx, args.output_folder, args.filename[:-3] + "_" + str(batch_idx) + ".th")
                    if args.save:
                        best_acc_mean = save_model_best_acc(acc_mean, best_acc_mean, model, meta_optimizer, batch_idx, args.output_folder, args.filename)
                else:
                    print("output folder not defined")

    args.writer = writer
        
            
def test(args):

    if args.dataset == 'mini':
        dataset = helper_with_default(MiniImagenet,
                                        folder=args.folder,
                                        shots=args.num_shots,
                                        ways=args.num_ways,
                                        shuffle=True,
                                        test_shots=15,
                                        meta_test=True,
                                        download=args.download,
                                        seed=args.seed,
                                        transform=Compose([
                                            Resize((84,84)),
                                            ToTensor(),
                                            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                                        ]))
    elif args.dataset == 'tiered':
        dataset = helper_with_default(TieredImagenet,
                                    folder=args.folder,
                                    shots=args.num_shots,
                                    ways=args.num_ways,
                                    shuffle=True,
                                    test_shots=15,
                                    meta_test=True,
                                    download=args.download,
                                    seed=args.seed,
                                    transform=Compose([
                                        Resize((84,84)),
                                        ToTensor(),
                                        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                                    ]))
    elif args.dataset == 'omni':
        dataset = omniglot(folder=args.folder,
                            shots=args.num_shots,
                            ways=args.num_ways,
                            shuffle=True,
                            test_shots=15,
                            meta_test=True,
                            seed=args.seed,
                            download=args.download)

    elif args.dataset == 'crop':
        dataset = helper_with_default(CropDisease,
                                    folder=args.folder,
                                    shots=args.num_shots,
                                    ways=args.num_ways,
                                    shuffle=True,
                                    test_shots=15,
                                    meta_test=True,
                                    download=args.download,
                                    seed=args.seed,
                                    transform=Compose([
                                        Resize((84,84)),
                                        ToTensor(),
                                        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                                    ]))
    
    dataloader = BatchMetaDataLoader(dataset,
                                     batch_size=args.batch_size,
                                     shuffle=True,
                                     num_workers=args.num_workers,
                                     pin_memory=True,
                                     drop_last=True)
    
    if args.dataset == 'omni':
        in_channels = 1
    else:
        in_channels = 3

    model = ConvolutionalNeuralNetwork(in_channels,
                                    int(args.num_ways),
                                    hidden_size=args.hidden_size,
                                    embedding_size=args.embedding_size)
    model.to(device=args.device)

    # Load model
    model, _, best_acc, episode = load_model(model, None, args.output_folder, args.filename, args.device)
    print(f"Model loaded at episode {episode}")

    model = model.eval()
    acc_all = []

    for arg in vars(args):
        print(str(arg), str(getattr(args,arg)))
    

    with tqdm(total=args.num_test_ep) as pbar:
        batch_idx = 0
        while batch_idx < args.num_test_ep:
            for batch in dataloader:
                if batch_idx >= args.num_test_ep:
                    break
                else:
                    batch_idx += 1
                
                support_inputs, support_targets = batch['train']
                support_inputs = support_inputs.to(device=args.device)
                support_targets = support_targets.to(device=args.device)

                query_inputs, query_targets = batch['test']
                query_inputs = query_inputs.to(device=args.device)
                query_targets = query_targets.to(device=args.device)
                
                acc = torch.tensor(0., device=args.device)
                for task_idx, (support_input, support_target, query_input,
                        query_target) in enumerate(zip(support_inputs, support_targets,
                        query_inputs, query_targets)):

                    for i in range(args.nb_test_task_update):
                        if i==0:
                            support_logit = model(support_input)
                            params = None
                        else:
                            support_logit = model(support_input, params=params)
                    
                        inner_loss = F.cross_entropy(support_logit, support_target)
                        
                        model.zero_grad()
                        params = gradient_update_parameters(model,
                                                            inner_loss,
                                                            params=params,
                                                            step_size=args.step_size,
                                                            first_order=args.first_order)

                    with torch.no_grad():
                        query_logit = model(query_input, params=params)
                        acc_this = get_accuracy(query_logit, query_target)
                    acc_all.append(acc_this)
                    acc += acc_this
                
                acc.div_(args.batch_size)
                
                pbar.set_postfix(accuracy='{0:.4f}'.format(acc.item()))
                pbar.update()
            
    acc_all = torch.stack(acc_all)
    acc_mean = torch.mean(acc_all)
    acc_std  = torch.std(acc_all)
    print(f'{args.num_test_ep} Test Acc = {acc_mean} +- {1.96* acc_std/np.sqrt(args.num_test_ep)}')
    if args.writer is not None:
        hparams_dict = {arg:str(getattr(args,arg)) for arg in vars(args)}
        metric_dict = {'hparam/Test mean accuracy': acc_mean.item(), 'hparam/Test std accuracy': acc_std.item()}
        args.writer.add_hparams(hparams_dict, metric_dict)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('Model-Agnostic Meta-Learning (MAML) Mini/TieredImageNet')

    parser.add_argument('--folder', type=str, default='/home/qbouniot/Documents/Data/Datasets/',
        help='Path to the folder the data is downloaded to.')
    parser.add_argument('--num-shots', type=int, default=1,
        help='Number of examples per class (k in "k-shot", default: 1).')
    parser.add_argument('--num-ways', type=int, default=5,
        help='Number of classes per task (N in "N-way", default: 5).')
    parser.add_argument('--dataset', type=str, default='mini',
        help='Dataset for training/testing (choice: mini/tiered/omni, default: mini)')
    parser.add_argument('--model', type=str, default='conv4',
        help='Backbone model (choice: conv4/resnet, default: conv4) ')

    parser.add_argument('--use_ratio', action='store_true',
        help='Use the ratio of singular values in the loss')
    parser.add_argument('--s_ratio', action='store_true')
    parser.add_argument('--s_norm', action='store_true',
        help='Minimize the norm of singular values in the outer loss')

    parser.add_argument('--lbda1', type=float, default=1.,
        help='Hyperparameter for the SV ratio')
    parser.add_argument('--lbda2', type=float, default=1.,
        help="Hyperparameter for the norm")

    parser.add_argument('--first-order', action='store_true',
        help='Use the first-order approximation of MAML.')

    parser.add_argument('--lr', type=float, default=1e-3,
        help='Meta-learning rate (for the outer loop)')
    parser.add_argument('--step-size', type=float, default=0.4,
        help='Step-size for the gradient step for adaptation (default: 0.4).')
    parser.add_argument('--nb-task-update', type=int, default=1,
        help='Number of updates for each training task (in inner loop)')
    parser.add_argument('--nb-test-task-update', type=int, default=10,
        help='Number of updates for each testing task (in inner loop)')
    parser.add_argument('--hidden-size', type=int, default=64,
        help='Number of channels for each convolutional layer (default: 64).')
    parser.add_argument('--embedding-size', type=int, default=1600,
        help='Embedding dimension before the classifier (default: 1600)')

    parser.add_argument('--output-folder', type=str, default='./checkpoints/',
        help='Path to the output folder for saving the model (optional).')
    parser.add_argument('--save', action='store_true',
        help='Save best checkpoint or not')
    parser.add_argument('--save_all', action='store_true',
        help='Save all checkpoints or not')
    parser.add_argument('--filename', '-f', type=str,
        help='File name of checkpoints')
        
    parser.add_argument('--batch-size', type=int, default=16,
        help='Number of tasks in a mini-batch of tasks (default: 16).')
    parser.add_argument('--num-batches', type=int, default=100,
        help='Number of batches the model is trained over (default: 100).')
    parser.add_argument('--num-test-ep', type=int, default=600,
        help='Number of batches of episodes the model is tested over (defaul: 1000)')
    parser.add_argument('--val-interval', type=int, default=1000,
        help='Number of training batches between validation')
    parser.add_argument('--num-workers', type=int, default=4,
        help='Number of workers for data loading (default: 1).')
    parser.add_argument('--download', action='store_true',
        help='Download the dataset in the data folder.')
    parser.add_argument('--use_cuda', action='store_true',
        help='Use CUDA if available.')
    parser.add_argument('--gpu', type=int, default=0,
        help='Which gpu to use if CUDA is enabled (default: 0)')
    parser.add_argument('--seed', type=int, default=None,
        help='Set the seed for the training and testing')

    args = parser.parse_args()
    args.device = torch.device(f'cuda:{args.gpu}' if args.use_cuda
        and torch.cuda.is_available() else 'cpu')

    if args.seed is not None:
        fix_seed(args.seed)                                     

    train(args)
    if args.save:
        test(args)