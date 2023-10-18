import numpy as np
import torch
import pickle
import pdb
from utils.utils import *
import copy
import os
from datasets.dataset_generic import save_splits
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score
from models.model_attention_mil import MIL_Attention_fc

from utils.fl_utils import sync_models, federated_averaging

class Accuracy_Logger(object):
    """Accuracy logger"""
    def __init__(self, n_classes):
        super(Accuracy_Logger, self).__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
    
    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)

    def log_batch(self, count, correct, c):
        self.data[c]["count"] += count
        self.data[c]["correct"] += correct
    
    def get_summary(self, c):
        count = self.data[c]["count"] 
        correct = self.data[c]["correct"]
        
        if count == 0: 
            acc = None
        else:
            acc = float(correct) / count
        
        return acc, correct, count

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, stop_epoch=50, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt'):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss

def train_fl(datasets, cur, args):
    """   
        train for a single fold
    """
    # number of institutions

    print('\nTraining Fold {}!'.format(cur))
    writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    from tensorboardX import SummaryWriter
    writer = SummaryWriter(writer_dir, flush_secs=15)
    print('\nInit train/val/test splits...', end=' ')
    train_splits, val_split, test_split = datasets
    num_insti = len(train_splits)
    # save_splits(datasets, ['train', 'val', 'test'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    print('Done!')
    for idx in range(num_insti):
        print("Worker_{} Training on {} samples".format(idx,len(train_splits[idx])))
        print("Validating on {} samples".format(len(val_split)))
        print("Testing on {} samples".format(len(test_split)))

    print('\nInit loss function...', end=' ')
    loss_fn = nn.CrossEntropyLoss()
    print('Done!')
    
    print('\nInit Model...', end=' ')
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes}
    if args.model_type =='attention_mil':
        if args.model_size is not None:
            model_dict.update({"size_arg": args.model_size})
        
        model = MIL_Attention_fc(**model_dict)
        worker_models =[MIL_Attention_fc(**model_dict) for idx in range(num_insti)]

    
    else: # args.model_type == 'mil'
        raise NotImplementedError
    
    sync_models(model, worker_models)   
    device_counts = torch.cuda.device_count()
    if device_counts > 1:
        device_ids = [idx % device_counts for idx in range(num_insti)]
    else:
        device_ids = [0]*num_insti
    
    model.relocate(device_id=0)
    for idx in range(num_insti):
        worker_models[idx].relocate(device_id=device_ids[idx])

    print('Done!')
    print_network(model)
    print('\nInit optimizer ...', end=' ')
    worker_optims = [get_optim(worker_models[i], args) for i in range(num_insti)]
    print('Done!')
    
    print('\nInit Loaders...', end=' ')
    train_loaders = []
    for idx in range(num_insti):
        train_loaders.append(get_split_loader(train_splits[idx], training=True, testing = args.testing, 
                                              weighted = args.weighted_sample))
    val_loader = get_split_loader(val_split, testing = args.testing)
    test_loader = get_split_loader(test_split, testing = args.testing)

    if args.weighted_fl_avg:
        weights = np.array([len(train_loader) for train_loader in train_loaders]) 
        weights = weights / weights.sum()
    else:
        weights = None

    print('Done!')

    print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping:
        early_stopping = EarlyStopping(patience = 10, stop_epoch= 20, verbose = True)

    else:
        early_stopping = None
    print('Done!')

    for epoch in range(args.max_epochs):        
        train_loop_fl(epoch, model, worker_models, train_loaders, worker_optims, 
                     args.n_classes, writer, loss_fn)

        if (epoch + 1) % args.E == 0:
            best_model, best_model_index,all_val_loss = find_top_model(worker_models, val_loader,args.n_classes,loss_fn)
            model, worker_models = federated_averaging(model, worker_models, best_model_index, all_val_loss, weights, args)
            sync_models(model, worker_models)  

        stop = validate(cur, epoch, model, val_loader, args.n_classes, 
                early_stopping, writer, loss_fn, args.results_dir)
        
        if stop: 
            break

    if args.early_stopping:
        model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))
    else:
        torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))

    _, val_error, val_auc, _,_,_= summary(model, val_loader, args.n_classes)
    print('Val error: {:.4f}, ROC AUC: {:.4f}'.format(val_error, val_auc))

    results_dict, test_error, test_auc, acc_logger,test_bacc,test_F1 = summary(model, test_loader, args.n_classes)
    print('Test error: {:.4f}, ROC AUC: {:.4f},Test bacc: {:.4f},Test F1: {:.4f}'.format(test_error, test_auc,test_bacc,test_F1))

    for i in range(args.n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))

        if writer:
            writer.add_scalar('final/test_class_{}_acc'.format(i), acc, 0)

    if writer:
        writer.add_scalar('final/val_error', val_error, epoch)
        writer.add_scalar('final/val_auc', val_auc, epoch)
        writer.add_scalar('final/test_error', test_error, epoch)
        writer.add_scalar('final/test_auc', test_auc, epoch)
    
    writer.close()
    return results_dict, test_auc, val_auc, test_error, val_error,test_bacc,test_F1


def train(datasets, cur, args):
    """   
        train for a single fold
    """
    print('\nTraining Fold {}!'.format(cur))
    writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)
    
    from tensorboardX import SummaryWriter
    writer = SummaryWriter(writer_dir, flush_secs=15)

    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split, test_split = datasets
    save_splits(datasets, ['train', 'val', 'test'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))
    print("Testing on {} samples".format(len(test_split)))

    print('\nInit loss function...', end=' ')
    loss_fn = nn.CrossEntropyLoss()
    print('Done!')
    
    print('\nInit Model...', end=' ')
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes}
    if args.model_type =='attention_mil':
        if args.model_size is not None:
            model_dict.update({"size_arg": args.model_size})
        model = MIL_Attention_fc(**model_dict)
    
    else: 
        raise NotImplementedError
    
    model.relocate()
    print('Done!')
    print_network(model)

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    print('Done!')
    
    print('\nInit Loaders...', end=' ')
    train_loader = get_split_loader(train_split, training=True, testing = args.testing, weighted = args.weighted_sample)
    val_loader = get_split_loader(val_split,  testing = args.testing)
    test_loader = get_split_loader(test_split, testing = args.testing)
    print('Done!')

    print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping:
        early_stopping = EarlyStopping(patience = 10, stop_epoch=20, verbose = True)

    else:
        early_stopping = None
    print('Done!')

    for epoch in range(args.max_epochs):        
        train_loop(epoch, model, train_loader, optimizer, args.n_classes, writer, loss_fn)
        stop = validate(cur, epoch, model, val_loader, args.n_classes, 
            early_stopping, writer, loss_fn, args.results_dir)
        
        if stop: 
            break

    if args.early_stopping:
        model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))
    else:
        torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))

    _, val_error, val_auc, _,_,_= summary(model, val_loader, args.n_classes)
    print('Val error: {:.4f}, ROC AUC: {:.4f}'.format(val_error, val_auc))

    results_dict, test_error, test_auc, acc_logger,test_bacc,test_F1 = summary(model, test_loader, args.n_classes)
    print('Test error: {:.4f}, ROC AUC: {:.4f},Test bacc: {:.4f},Test F1: {:.4f}'.format(test_error, test_auc,test_bacc,test_F1))

    for i in range(args.n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))

        if writer and acc is not None:
            writer.add_scalar('final/test_class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('final/val_error', val_error, epoch)
        writer.add_scalar('final/val_auc', val_auc, epoch)
        writer.add_scalar('final/test_error', test_error, epoch)
        writer.add_scalar('final/test_auc', test_auc, epoch)
    
    writer.close()
    return results_dict, test_auc, val_auc, test_error, val_error,test_bacc,test_F1

def train_loop_fl(epoch, model, worker_models, worker_loaders, worker_optims, n_classes, writer = None, loss_fn = None):

    num_insti = len(worker_models)    
    model.train()
    
    # for idx in range(num_insti):
    #     worker_models[idx].train()
   
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    
    train_loss = 0.
    train_error = 0.

    print('\n')
    for idx in range(len(worker_loaders)):
        # pdb.set_trace()
        if worker_models[idx].device is not None:
            model_device = worker_models[idx].device
        else:
            model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        for batch_idx, (data, label) in enumerate(worker_loaders[idx]):
            data, label = data.to(model_device), label.to(model_device)
            logits, Y_prob, Y_hat, _, _ = worker_models[idx](data)

            acc_logger.log(Y_hat, label)
            loss = loss_fn(logits, label)
            loss_value = loss.item()
            train_loss += loss_value
            if (batch_idx + 1) % 5 == 0:
                print('batch {}, loss: {:.4f}, '.format(batch_idx, loss_value),
                      'label: {}, bag_size: {}'.format(label.item(), data.size(0)))

            error = calculate_error(Y_hat, label)
            train_error += error

            # backward pass
            loss.backward()
            # step
            worker_optims[idx].step()
            worker_optims[idx].zero_grad()


    # calculate loss and error for epoch
    train_loss /= np.sum(len(worker_loaders[i]) for i in range(num_insti))
    train_error /= np.sum(len(worker_loaders[i]) for i in range(num_insti))
    print('Epoch: {}, train_loss: {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)


def train_loop(epoch, model, loader, optimizer, n_classes, writer = None, loss_fn = None):   
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    train_loss = 0.
    train_error = 0.

    print('\n')
    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)

        logits, Y_prob, Y_hat, _, _ = model(data)
        
        acc_logger.log(Y_hat, label)
        loss = loss_fn(logits, label)
        loss_value = loss.item()
        
        train_loss += loss_value
        if (batch_idx + 1) % 5 == 0:
            print('batch {}, loss: {:.4f}, label: {}, bag_size: {}'.format(batch_idx, loss_value, label.item(), data.size(0)))
           
        error = calculate_error(Y_hat, label)
        train_error += error
        
        # backward pass
        loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)

    print('Epoch: {}, train_loss: {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)

   
def validate(cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir=None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    val_loss = 0.
    val_error = 0.
    
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))

    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(loader):
            data, label = data.to(device), label.to(device)

            logits, Y_prob, Y_hat, _, _ = model(data)

            acc_logger.log(Y_hat, label)
            
            loss = loss_fn(logits, label)

            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()
            
            val_loss += loss.item()
            error = calculate_error(Y_hat, label)
            val_error += error
            

    val_error /= len(loader)
    val_loss /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
    
    else:
        auc = roc_auc_score(labels, prob, multi_class='ovr')
    
    
    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('val/class_{}_acc'.format(i), acc, epoch)
     

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False


def summary(model, loader, n_classes):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        slide_id = slide_ids.iloc[batch_idx]
        with torch.no_grad():
            logits, Y_prob, Y_hat, _, _ = model(data)

        acc_logger.log(Y_hat, label)
        probs = Y_prob.cpu().numpy()
        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()
        
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
        error = calculate_error(Y_hat, label)
        test_error += error

    test_error /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(all_labels, all_probs[:, 1])

    else:
        auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')

    bacc = balanced_accuracy_score(all_labels, np.argmax(all_probs, axis=1))
    f1 = f1_score(all_labels, np.argmax(all_probs, axis=1), average='weighted')

    return patient_results, test_error, auc, acc_logger, bacc, f1

def find_top_model(worker_models, val_loader,n_classes,loss_fn):
    """
    Selects the best-performing model among multiple models based on their performance on validation set
    :param worker_models: List of trained models from different workers
    :param val_loader: Data loader for validation set
    :return: The best-performing model and its index in worker_models
    """
    # Record the performance of each model
    model_performance = []
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    all_val_loss = 0.
    # Traverse each model, evaluate its performance on the validation set, and record it in model_performance
    for model in worker_models:
        val_loss = 0.

        model.eval()
        with torch.no_grad():
            num_val_samples = len(val_loader.dataset) // 10  # Use one-fifth of the data for validation
            for i, (data, label) in enumerate(val_loader):
               if i >= num_val_samples:
                break
               
               data, label = data.to(model.device), label.to(model.device)
               logits, Y_prob, Y_hat, _, _ = model(data)
               acc_logger.log(Y_hat, label)
               loss = loss_fn(logits, label)
               loss_value = loss.item()
               val_loss += loss_value
            
            val_loss /= num_val_samples

        model_performance.append({'model': model, 'val_loss': val_loss})
        all_val_loss += val_loss
    # Select the best-performing model based on loss
    sorted_models = sorted(model_performance, key=lambda x: x['val_loss'])
    top_model = sorted_models[0]['model']

    # Find the index of the top model in the worker_models list
    top_model_index = worker_models.index(top_model)

    return top_model, top_model_index, all_val_loss