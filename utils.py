# @copyright CEA-LIST/DIASI/SIALV/LVA (2022)
# @author CEA-LIST/DIASI/SIALV/LVA <quentin.bouniot@cea.fr>
# @license CECILL

import torch
import os

import torch.nn.functional as F

import numpy as np
import random

def get_accuracy(logits, targets):
    """Compute the accuracy (after adaptation) of MAML on the test/query points
    Parameters
    ----------
    logits : `torch.FloatTensor` instance
        Outputs/logits of the model on the query points. This tensor has shape
        `(num_examples, num_classes)`.
    targets : `torch.LongTensor` instance
        A tensor containing the targets of the query points. This tensor has 
        shape `(num_examples,)`.
    Returns
    -------
    accuracy : `torch.FloatTensor` instance
        Mean accuracy on the query points
    """
    _, predictions = torch.max(logits, dim=-1)
    return torch.mean(predictions.eq(targets).float())

def get_accuracy_proto(prototypes, embeddings, targets):
    """Compute the accuracy of the prototypical network on the test/query points.
    Parameters
    ----------
    prototypes : `torch.FloatTensor` instance
        A tensor containing the prototypes for each class. This tensor has shape 
        `(meta_batch_size, num_classes, embedding_size)`.
    embeddings : `torch.FloatTensor` instance
        A tensor containing the embeddings of the query points. This tensor has 
        shape `(meta_batch_size, num_examples, embedding_size)`.
    targets : `torch.LongTensor` instance
        A tensor containing the targets of the query points. This tensor has 
        shape `(meta_batch_size, num_examples)`.
    Returns
    -------
    accuracy : `torch.FloatTensor` instance
        Mean accuracy on the query points.
    """
    sq_distances = torch.sum((prototypes.unsqueeze(1)
        - embeddings.unsqueeze(2)) ** 2, dim=-1)

    _, predictions = torch.min(sq_distances, dim=-1)
    
    return torch.mean(predictions.eq(targets).float())

def prototypical_loss(prototypes, embeddings, targets, **kwargs):
    """Compute the loss (i.e. negative log-likelihood) for the prototypical 
    network, on the test/query points.
    Parameters
    ----------
    prototypes : `torch.FloatTensor` instance
        A tensor containing the prototypes for each class. This tensor has shape 
        `(batch_size, num_classes, embedding_size)`.
    embeddings : `torch.FloatTensor` instance
        A tensor containing the embeddings of the query points. This tensor has 
        shape `(batch_size, num_examples, embedding_size)`.
    targets : `torch.LongTensor` instance
        A tensor containing the targets of the query points. This tensor has 
        shape `(batch_size, num_examples)`.
    Returns
    -------
    loss : `torch.FloatTensor` instance
        The negative log-likelihood on the query points.
    """

    squared_distances = torch.sum((prototypes.unsqueeze(2)
        - embeddings.unsqueeze(1)) ** 2, dim=-1)

    return F.cross_entropy(-squared_distances, targets, **kwargs)

def load_model(net, optim, save_path, filename, device):
    """Loads parameters of a saved model into a new model as well as the optimizer
    state.

    Args:
        net (torch.Module): A model with the same architecture than the saved model
        optim (torch.optim): The same optimizer used during training
        save_path (str): The path of the folder with the saved model
        filename (str): The name of the file containing the parameters
        device (cuda device): The cuda device that will contain the model

    Returns:
        net: The model with the saved parameters
        optim: The optimizer with the saved state
        best_acc: The accuracy of the saved model
        epoch: The current epoch of the saved model
    """
    state = torch.load(os.path.join(save_path, filename), map_location=device)
    net.load_state_dict(state['net'])
    best_acc = state['acc']
    epoch = state['epoch']
    try :
        optim_state = state['optim']
    except KeyError:
        optim_state = None

    if optim_state and optim:
        optim.load_state_dict(optim_state)
    
    return net, optim, best_acc, epoch

def save_model_best_acc(acc, best_acc, net, optim, epoch, save_path, filename):
    """Save a model and its optimizer if the current accuracy is better than best one.

    Args:
        acc (int): Accuracy of the current model
        best_acc (int): Best accuracy of the saved model
        net (torch.Module): The model that will be saved
        optim (torch.optim): The optimizer of the model
        epoch (int): The current epoch of training
        save_path (str): The path of the folder to save to
        filename (str): The name of the file that will contain the parameters

    Returns:
        best_acc: The new best accuracy of the newly saved model (best_acc = acc)
    """
    if acc > best_acc:
        print('Saving ...')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'optim': optim.state_dict()
        }
        torch.save(state, os.path.join(save_path, filename))
        best_acc = acc
    return best_acc

def save_model(acc, net, optim, epoch, save_path, filename):
    """Save a model and its optimizer

    Args:
        acc (int): Accuracy of the current model
        net (torch.Module): The model that will be saved
        optim (torch.optim): The optimizer of the model
        epoch (int): The current epoch of training
        save_path (str): The path of the folder to save to
        filename (str): The name of the file that will contain the parameters
    """
    print('Saving ...')
    state = {
        'net': net.state_dict(),
        'acc': acc,
        'epoch': epoch,
        'optim': optim.state_dict()
    }
    torch.save(state, os.path.join(save_path, filename))


def fix_seed(seed, deterministic=False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:                                             
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
