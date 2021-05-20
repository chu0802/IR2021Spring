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
    def __init__(self, user_list, num_item, num_user, duplicating):
        super(Datapair, self).__init__()
        self.num_item = num_item
        self.num_user = num_user
        self.items = np.arange(num_item)
        
        self.user_list = user_list
        self.user_complement_list = [np.setdiff1d(self.items, item_list) for item_list in self.user_list]
        self.user_pair = [(user, item) for _ in range(duplicating) for (user, item_list) in enumerate(user_list) for item in item_list]
    def __len__(self):    
        return len(self.user_pair)
    def __getitem__(self, idx):
        u, i = self.user_pair[idx]
        return u, i, np.random.choice(self.user_complement_list[u])
    
class BPR(nn.Module):
    def __init__(self, num_user, num_item, dim, reg=0.05):
        super(BPR, self).__init__()
        self.W = nn.Parameter(torch.empty(num_user, dim))
        self.H = nn.Parameter(torch.empty(num_item, dim))
        nn.init.xavier_normal_(self.W.data)
        nn.init.xavier_normal_(self.H.data)
        
    def forward(self, u, i, j):
        u, i, j = self.W[u, :], self.H[i, :], self.H[j, :]
        ui = torch.einsum('ij,ij->i', u, i)
        uj = torch.einsum('ij,ij->i', u, j)
        
        log_prob = F.logsigmoid(ui - uj).sum()
        return -log_prob 
    
    def predict(self, u):
        u = self.W[u].reshape(1, -1)
        scores = torch.mm(u, self.H.t())

        return scores.detach().cpu().numpy().squeeze()


def train_step(model, optimizer, dloader):
    model.train()
    total_loss = 0
    print_interval = int(len(dloader)/100)
    for idx, (u, i, j) in enumerate(dloader):
        if (idx+1) % print_interval == 0:
            print('    Step: %03d%%' % (100 * (idx+1)/len(dloader)), end='\r')
        u, i, j = u.cuda(), i.cuda(), j.cuda()
        optimizer.zero_grad()
        loss = model(u, i, j)
        loss.backward()
        optimizer.step()

        total_loss += loss.detach().item()
    return total_loss

def val_step(model, dloader):
    total_loss = 0
    with torch.no_grad():
        for u, i, j in enumerate(dloader):
            u, i, j = u.cuda(), i.cuda(), j.cuda()
            loss = model(u, i, j)

            total_loss += loss.item()
    return total_loss

def predict(model, dset):
    model.eval()
    result = []
    with torch.no_grad():
        for user in range(dset.num_user):
            scores = model.predict(user)[dset.user_complement_list[user]]
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

def train(model, optimizer, train_set, train_loader, num_epoch, val_list=None, validation=True):
    start = time()
    best_val_score = 0
    for epoch in range(num_epoch):
        print('Epoch %02d/%02d' % (epoch+1, num_epoch))
        train_loss = train_step(model, optimizer, train_loader) / len(train_loader)

        if validation:
            # For validation
            pred = predict(model, train_set)
            val_score = evaluate(pred, val_list)

            if val_score > best_val_score:
                best_val_score = val_score
                torch.save(model.state_dict(), './model/best_model.pt')
            print('Epoch %02d/%02d, train_loss: %.4f, val_score: %.4f, total_time: %06.2f' % (epoch+1, num_epoch, train_loss, val_score, time() - start))
        else:
            print('Epoch %03d/%03d, train_loss: %.4f, total_time: %06.2f' % (epoch+1, num_epoch, train_loss, time() - start))
        scheduler.step(train_loss)
    if not validation:
        torch.save(model.state_dict(), './model/best_model.pt')

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

    # I/O
    parser.add_argument('-dp', '--data_path', type=str, default='./data/train.csv')
    parser.add_argument('-mp', '--model_path', type=str, default='./model/best_model.pt')
    parser.add_argument('-o', '--output_file', type=str, default='./output.csv')

    # Mode
    parser.add_argument('-m', '--mode', type=str, choices=['training', 'validation', 'testing'], default='testing')

    return parser.parse_args()

if __name__ == '__main__':
    args = arguments_parsing()

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
    train_set = Datapair(train_list, num_item, num_user, args.duplicating)

    if args.mode != 'testing':
        train_loader = DataLoader(train_set, shuffle=True, batch_size=args.batch_size, num_workers=4)

        model = BPR(num_user, num_item, args.hidden_dim).cuda()
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=2)

        train(model, optimizer, train_set, train_loader, args.num_epoch, val_list, args.validation)

    best_model = BPR(num_user, num_item, args.hidden_dim)
    best_model.load_state_dict(torch.load(args.model_path))
    
    pred = predict(best_model, train_set)
    result = [[i, ' '.join([str(x) for x in p])] for i, p in enumerate(pred)]
    pd.DataFrame(result).to_csv(args.output_file, header=['UserId','ItemId'], index=False)
