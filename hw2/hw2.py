import os
import argparse
import numpy as np
import itertools
import random
from time import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import pandas as pd

def read_data(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        user_list = [list(map(lambda a: int(a), line.split(',')[1].split())) for line in lines[1:]]
        num_item = max([max(l) for l in user_list])+1
        num_user = len(user_list)
    return user_list, num_item, num_user

def train_val_split(data, ratio=0.1):
    train, val = [], []
    for l in data:
        val.append(np.random.choice(l, size=int(ratio*len(l)), replace=False))
        train.append(np.setdiff1d(l, val[-1]))
    return train, val

class Datapair(Dataset):
    def __init__(self, user_list, num_item, num_user, duplicating, num_neg):
        super(Datapair, self).__init__()
        self.num_item = num_item
        self.num_user = num_user
        self.items = np.arange(num_item)
        self.num_neg = num_neg
        
        self.user_list = user_list
        self.user_complement_list = [np.setdiff1d(self.items, item_list) for item_list in self.user_list]
        self.user_pair = [(user, item) for _ in range(duplicating) for (user, item_list) in enumerate(user_list) for item in item_list]
    def __len__(self):    
        return len(self.user_pair)
    def __getitem__(self, idx):
        u, i = self.user_pair[idx]
        if self.num_neg > 1:
            return u, i, np.random.choice(self.user_complement_list[u], size=self.num_neg, replace=False)
        return u, i, np.random.choice(self.user_complement_list[u])
    
class MF(nn.Module):
    def __init__(self, num_user, num_item, dim):
        super(MF, self).__init__()
        self.W = nn.Parameter(torch.empty(num_user, dim))
        self.H = nn.Parameter(torch.empty(num_item, dim))
        nn.init.xavier_normal_(self.W.data)
        nn.init.xavier_normal_(self.H.data)

    def forward(self):
        raiseNotImplementedError()
    
    def predict(self, u):
        u = self.W[u].reshape(1, -1)
        scores = torch.mm(u, self.H.t())

        return scores.detach().cpu().numpy().squeeze()

class BPR(MF):
    def __init__(self, num_user, num_item, dim):
        super(BPR, self).__init__(num_user, num_item, dim)

    def forward(self, u, i, j):
        # u: bsize, i: bsize, j: bsize * num_neg
        u, i, j = self.W[u], self.H[i], self.H[j]
        # u: bsize * hidden_dim, i: bsize * hidden_dim, j: bsize * num_neg * hidden_dim
        ui = torch.einsum('ij,ij->i', u, i)
        uj = torch.einsum('ij,ij->i', u, j)
        log_prob = F.logsigmoid(ui - uj).sum()
        return -log_prob 

class BCE(MF):
    def __init__(self, num_user, num_item, dim):
        super(BCE, self).__init__(num_user, num_item, dim)

    def forward(self, u, i, j):
        j = j.reshape(j.shape[0], -1)
        u, i, j = self.W[u], self.H[i], self.H[j].reshape(j.shape[0], j.shape[1], -1)
        ui = torch.einsum('ij,ij->i', u, i)
        uj = torch.einsum('ij,ibj->ib', u, j).flatten()
        li = torch.ones_like(ui)
        lj = torch.zeros_like(uj)

        criterion = nn.BCEWithLogitsLoss(reduction='sum')
        return (criterion(ui, li) + j.shape[1]*criterion(uj, lj)) / (j.shape[1]+1)

def train_step(model, optimizer, dloader, device):
    model.train()
    total_loss = 0
    print_interval = int(len(dloader)/100)
    for idx, (u, i, j) in enumerate(dloader):
        if (idx+1) % print_interval == 0:
            print('    Step: %03d%%' % (100 * (idx+1)/len(dloader)), end='\r')
        u, i, j = u.to(device), i.to(device), j.to(device)
        optimizer.zero_grad()
        loss = model(u, i, j)
        loss.backward()
        optimizer.step()

        total_loss += loss.detach().item()
    return total_loss

def predict(models, dset):
    if not isinstance(models, list):
        models = [models]
    result = []
    for user in range(dset.num_user):
        scores = []
        
        for model in models:
            model.eval()
            with torch.no_grad():
                scores.append(model.predict(user)[dset.user_complement_list[user]])
        scores = np.mean(scores, axis=0)
        ranking = np.argsort(scores)[-50:][::-1].astype(int)
        result.append(dset.user_complement_list[user][ranking])
    return result

def AveP(pred, ans):
    is_relevence = np.array([1 if p in ans else 0 for p in pred])
    return ((np.arange(sum(is_relevence))+1) / (np.where(is_relevence > 0)[0]+1)).sum() / len(ans)

def evaluate(pred, ans):
    map = 0
    for p, a in zip(pred, ans):
        map += AveP(p, a)
    return map/len(ans)

def train(model, optimizer, train_set, train_loader, val_list, args):
    start = time()
    best_val_score = 0
    for epoch in range(args.num_epoch):
        print('Epoch %02d/%02d' % (epoch+1, args.num_epoch))
        train_loss = train_step(model, optimizer, train_loader, args.device) / len(train_loader)

        if args.mode == 'validation':
            # For validation
            pred = predict(model, train_set)
            val_score = evaluate(pred, val_list)

            if val_score > best_val_score:
                best_val_score = val_score
                torch.save(model.state_dict(), os.path.join(args.model_path, '%s_0.pt' % (args.config)))
            print('Epoch %02d/%02d, train_loss: %.4f, val_score: %.4f, total_time: %06.2f' % (epoch+1, args.num_epoch, train_loss, val_score, time() - start))
        else:
            if args.save_interval > 0 and ((epoch+1) % args.save_interval == 0):
                torch.save(model.state_dict(), os.path.join(args.model_path, '%s_%d.pt' % (args.config, epoch+1)))
            print('Epoch %03d/%03d, train_loss: %.4f, total_time: %06.2f' % (epoch+1, args.num_epoch, train_loss, time() - start))
        scheduler.step(train_loss)
    if args.mode != 'validation':
        torch.save(model.state_dict(), os.path.join(args.model_path, '%s_0.pt' % (args.config)))

def arguments_parsing():
    parser = argparse.ArgumentParser()
    # hyper-parameters
    parser.add_argument('-s', '--seed', type=int, default=1126)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('-e', '--num_epoch', type=int, default=14)
    parser.add_argument('-bs', '--batch_size', type=int, default=1024)
    parser.add_argument('-dup', '--duplicating', type=int, default=10)
    parser.add_argument('-w', '--weight_decay', type=float, default=1e-3)

    # model
    parser.add_argument('-d', '--hidden_dim', type=int, default=512)
    parser.add_argument('-l', '--loss', type=str, choices=['BPR', 'BCE'], default='BPR')
    parser.add_argument('-dev', '--device', type=int, default=0)
    parser.add_argument('-nn', '--num_neg', type=int, default=1)

    # I/O
    parser.add_argument('-dp', '--data_path', type=str, default='./data/train.csv')
    parser.add_argument('-mp', '--model_path', type=str, default='./model')
    parser.add_argument('-mi', '--model_idx', type=str, default='0')
    parser.add_argument('-si', '--save_interval', type=int, default=0)
    parser.add_argument('-o', '--output_file', type=str, default='./output.csv')

    # Mode
    parser.add_argument('-m', '--mode', type=str, choices=['training', 'validation', 'testing'], default='testing')

    return parser.parse_args()

if __name__ == '__main__':
    args = arguments_parsing()
    args.device = torch.device('cpu') if args.device < 0 else torch.device('cuda:%d' % (args.device))

    pkey = ['seed', 'learning_rate', 'num_epoch', 'batch_size', 'duplicating', 'weight_decay', 'hidden_dim', 'loss', 'num_neg']
    args.config = str({k:vars(args)[k] for k in pkey})

    # Reproduction Settings

    torch.cuda.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    user_list, num_item, num_user = read_data(args.data_path)
    if args.mode == 'validation':
        train_list, val_list = train_val_split(user_list)
    else:
        train_list, val_list = user_list, None
    train_set = Datapair(train_list, num_item, num_user, args.duplicating, args.num_neg)

    if args.mode != 'testing':
        train_loader = DataLoader(train_set, shuffle=True, batch_size=args.batch_size, num_workers=4)

        if args.loss == 'BPR':
            model = BPR(num_user, num_item, args.hidden_dim).to(args.device)
        else:
            model = BCE(num_user, num_item, args.hidden_dim).to(args.device)

        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=2)

        train(model, optimizer, train_set, train_loader, val_list, args)
    
    # Load models
    model_idx = [int(x) for x in args.model_idx.split(',')]

    if args.loss == 'BPR':
        best_models = [BPR(num_user, num_item, args.hidden_dim) for _ in range(len(model_idx))]
    else:
        best_models = [BCE(num_user, num_item, args.hidden_dim) for _ in range(len(model_idx))]

    for idx, model in zip(model_idx, best_models):
        model.load_state_dict(torch.load(os.path.join(args.model_path, '%s_%d.pt' % (args.config, idx)), map_location=args.device))
        model.to(args.device)

    pred = predict(best_models, train_set)
    result = [[i, ' '.join([str(x) for x in p])] for i, p in enumerate(pred)]
    pd.DataFrame(result).to_csv(args.output_file, header=['UserId','ItemId'], index=False)
