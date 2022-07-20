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

from torch.utils.tensorboard import SummaryWriter

from models.ProtoNet import PrototypicalNetwork
from torchmeta.utils.prototype import get_prototypes
from utils import fix_seed, get_accuracy_proto, save_model_best_acc, load_model, prototypical_loss, save_model

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

    elif args.dataset == 'omni':
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

    if args.dataset == 'mini':

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
        in_channel = 1
    else:
        in_channel = 3

    model = PrototypicalNetwork(in_channel,
                                hidden_size=args.hidden_size)
    model.to(device=args.device)
    model = model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

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

            this_ratio = None

            train_embeddings = model(train_inputs)
            test_embeddings = model(test_inputs)

            prototypes = get_prototypes(train_embeddings, train_targets, train_dataset.num_classes_per_task)

            if args.norm:
                prototypes = F.normalize(prototypes, p=2, dim=2)
                
            loss = prototypical_loss(prototypes, test_embeddings, test_targets)

            if args.s_ent:
                u,s,v = torch.svd(torch.matmul(prototypes, prototypes.transpose(1,2)))
                this_ratio = torch.mean(s[:,0] / s[:,-1])
                s_norm = torch.mean(torch.norm(s, dim=1))
                w_norm = torch.mean(torch.norm(prototypes, p=2, dim=2))
                if args.s_ent:
                    entropy = torch.mean(torch.sum(F.softmax(s.sqrt(), dim=1) * F.log_softmax(s.sqrt(), dim=1), dim=1))
                    loss.add_(args.lbda * entropy)

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                accuracy = get_accuracy_proto(prototypes, test_embeddings, test_targets)
                if this_ratio is None:
                    u,s,v = torch.svd(torch.matmul(prototypes, prototypes.transpose(1,2)))
                    this_ratio = torch.mean(s[:,0] / s[:,-1])
                    s_norm = torch.mean(torch.norm(s, dim=1))
                    w_norm = torch.mean(torch.norm(prototypes, p=2, dim=2))

            log = {
                'accuracy': f'{accuracy.item():.4f}',
                'val_mean': f'{acc_mean:.4f}',
                'val_std': f'{acc_std:.4f}'
            }
            if args.s_ent:
                log['entropy'] = f'{entropy.item():.4f}'

            pbar.set_postfix(log)
            writer.add_scalar('Training/Accuracy', accuracy.item(), batch_idx)
            if this_ratio is not None:
                writer.add_scalar('Training/last_ratio', this_ratio.item(), batch_idx)
                writer.add_scalar('Training/si_max', torch.mean(s[:,0]).item(), batch_idx)
                writer.add_scalar('Training/si_min', torch.mean(s[:,-1]).item(), batch_idx)
                writer.add_scalar('Training/s_norm', s_norm.item(), batch_idx)
                writer.add_scalar('Training/w_norm', w_norm.item(), batch_idx)

            # Validation loop
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
                        
                        support_embeddings = model(support_inputs)
                        query_embeddings = model(query_inputs)

                        prototypes = get_prototypes(support_embeddings, support_targets, val_dataset.num_classes_per_task)
                            
                        if args.norm:
                            prototypes = F.normalize(prototypes, p=2, dim=2)
                        
                        with torch.no_grad():
                            acc_this = get_accuracy_proto(prototypes, query_embeddings, query_targets)
                        acc_all.append(acc_this)
                                                
                        val_pbar.set_postfix(accuracy='{0:.4f}'.format(acc_this.item()))
                acc_all = torch.stack(acc_all)
                acc_mean = torch.mean(acc_all)
                acc_std  = torch.std(acc_all)
                writer.add_scalar('Val/Mean accuracy', acc_mean.item(), batch_idx)
                writer.add_scalar('Val/Std accuracy', acc_std.item(), batch_idx)
                model = model.train()

                # Save model
                if args.output_folder is not None:
                    if args.save:
                        best_acc_mean = save_model_best_acc(acc_mean, best_acc_mean, model, optimizer, batch_idx, args.output_folder, args.filename)
                    if args.save_all:
                        save_model(acc_mean, model, optimizer, batch_idx, args.output_folder, args.filename[:-3] + "_" + str(batch_idx) + ".th")
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
        dataset = omniglot(args.folder,
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
        in_channel = 1
    else:
        in_channel = 3

    model = PrototypicalNetwork(in_channel,
                                args.hidden_size)
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
                
                support_embeddings = model(support_inputs)
                query_embeddings = model(query_inputs)

                prototypes = get_prototypes(support_embeddings, support_targets, dataset.num_classes_per_task)

                if args.norm:
                    prototypes = F.normalize(prototypes, p=2, dim=2)
                    
                with torch.no_grad():
                    acc_this = get_accuracy_proto(prototypes, query_embeddings, query_targets)
                acc_all.append(acc_this)
                            
                pbar.set_postfix(accuracy='{0:.4f}'.format(acc_this.item()))
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

    parser = argparse.ArgumentParser('Prototypical Network MiniImageNet')

    parser.add_argument('--folder', type=str, default='/home/qbouniot/Documents/Data/Datasets/',
        help='Path to the folder the data is downloaded to.')
    parser.add_argument('--num-shots', type=int, default=1,
        help='Number of examples per class (k in "k-shot", default: 1).')
    parser.add_argument('--num-ways', type=int, default=5,
        help='Number of classes per task (N in "N-way", default: 5).')
    parser.add_argument('--dataset', type=str, default='mini',
        help='Dataset for training/testing (choice: mini/tiered/omni, default: mini)')

    parser.add_argument('--s_ent', action='store_true',
        help='Use entropy of singular values')
    parser.add_argument('--norm', action='store_true',
        help='Normalize prototypes before computing prototypical loss')
    parser.add_argument('--lbda', type=float, default=1,
        help="Loss regularization strength")

    parser.add_argument('--lr', type=float, default=1e-3,
        help='Meta-learning rate (for the outer loop)')
    parser.add_argument('--hidden-size', type=int, default=64,
        help='Number of channels for each convolutional layer (default: 64).')

    parser.add_argument('--output-folder', type=str, default='./checkpoints/',
        help='Path to the output folder for saving the model (optional).')
    parser.add_argument('--save', action='store_true',
        help='Save checkpoints or not')
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
        help='Number of workers for data loading (default: 4).')
    parser.add_argument('--download', action='store_true',
        help='Download the Omniglot dataset in the data folder.')
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