import sys
sys.path.append("/home/disk3/code/SocialBotTrain")
import numpy as np
import pandas as pd
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import json
from tqdm import tqdm
import torch
import torch.nn as nn
import time
import datetime
import random
import pickle
import torch.utils.data as Data
from model.Propagation import *
from utils.dataloader import *
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

Nclass = 2

class Trainer(nn.Module):
    def __init__(self, Prop_cell, cls, batch_size=5, grad_accum_cnt=4):
        super(Trainer, self).__init__()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.prop_cell = Prop_cell.to(self.device)
        self.cls = cls.to(self.device)
        self.loss_fn = nn.NLLLoss()
        self.batch_size = batch_size
        self.grad_accum_cnt = grad_accum_cnt

    def forward(self, user):
        hidden = self.prop_cell(torch.Tensor(user).to(self.device))
        pred = self.cls(hidden).softmax(dim=1)
        return pred

    def loss(self, user, y):
        pred_y = self.forward(user)
        loss = self.loss_fn(pred_y.to(self.device).log(), y.to(self.device))
        acc = accuracy_score(y.cpu().numpy(), pred_y.argmax(dim=1).cpu().numpy())
        macro_f1= f1_score(y.cpu().numpy(), pred_y.argmax(dim=1).cpu().numpy(), average='macro')
        return loss, acc, macro_f1

    def train_iters(self, train_set, test_set, dev_set, valid_every=20, max_epochs=10, lr_discount=1.0, 
                    best_valid_acc=0.0, best_test_acc=0.0, best_test_f1=0.0, best_valid_test_acc=0.0, best_valid_test_f1=0.0, 
                    log_dir="/home/disk3/code/socialTrian/logs/", model_file=""):

        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True, collate_fn=train_set.collate_fn)
        dev_loader = DataLoader(dev_set, batch_size=self.batch_size, shuffle=True, collate_fn=dev_set.collate_fn)
        te_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=True, collate_fn=test_set.collate_fn)

        optimizer = torch.optim.Adam([
            {'params': self.prop_cell.parameters(), 'lr': 1e-2 * lr_discount / self.grad_accum_cnt,'weight_decay': 0},
            {'params': self.cls.parameters(), 'lr': 1e-2 * lr_discount / self.grad_accum_cnt,'weight_decay': 0}
        ]
        )

        optimizer.zero_grad()
        self.train()
        sum_loss, sum_acc, sum_f1 = 0.0, 0.0, 0.0

        #3  train&eval
        losses = []

        for epoch in range(max_epochs):
            for step, batch_data in enumerate(train_loader):
                loss, acc, macro_f1 = self.loss(batch_data[0], batch_data[1].to(self.device))
                loss.backward()
                torch.cuda.empty_cache()
                if step % self.grad_accum_cnt == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                torch.cuda.empty_cache()
                sum_loss += loss
                sum_acc += acc
                sum_f1 += macro_f1
                if (step + 1) % self.grad_accum_cnt == 0:
                    print('%6d | %6d  [%3d | %3d], loss/acc = %6.8f/%6.7f, macro_f1 = %6.7f , best_valid_acc:%6.7f ' % (
                                    step, len(train_loader),
                                    epoch + 1, max_epochs,
                                    sum_loss / self.grad_accum_cnt, sum_acc / self.grad_accum_cnt, sum_f1 / self.grad_accum_cnt,
                                    best_valid_acc
                                )
                    )
                    sum_loss, sum_acc, sum_f1 = 0.0, 0.0, 0.0
                losses.append(loss)
            
                ## cal loss & evaluate
                if (step + 1) % (valid_every * self.grad_accum_cnt) == 0:
                    val_loss, val_acc, val_f1, _ = self.valid(dev_loader)
                    test_loss, test_acc, test_f1, test_res = self.valid(te_loader)
                    self.train()
                    if best_test_f1 < test_f1:
                        best_test_acc = test_acc
                        best_test_f1 = test_f1
                        best_cls_res = test_res
                    print('##### %6d | %6d, [%3d | %3d], val_loss|val_acc|val_f1 = %6.8f/%6.7f/%6.7f, te_loss|te_acc|test_f1 = %6.8f/%6.7f/%6.7f, best_valid_acc/related_test_acc/related_test_f1= %6.7f/%6.7f/%6.7f, best_test_acc/best_test_f1=%6.7f/%6.7f' % (
                        step, len(train_loader),
                        epoch + 1, max_epochs,
                        val_loss, val_acc, val_f1,
                        test_loss, test_acc, test_f1, 
                        best_valid_acc, best_valid_test_acc, best_valid_test_f1,
                        best_test_acc, best_test_f1
                        )
                    )
                    if val_acc > best_valid_acc:
                        best_valid_acc = val_acc
                        best_valid_test_acc = test_acc
                        best_valid_test_f1 = test_f1
                        self.save_model(model_file)
            print("epoch=%d: loss=%f" % (epoch+1, torch.mean(torch.tensor(losses))))
            losses = []
        return best_valid_acc, best_test_acc, best_test_f1, best_valid_test_acc, best_valid_test_f1, best_cls_res

    def valid(self, data_loader, pretrained_file=None, all_metrics=False):
        self.eval()
        if pretrained_file is not None and os.path.exists(pretrained_file):
            self.load_model(pretrained_file)
        labels = []
        preds = []
        with torch.no_grad():
            for batch_data in data_loader:
                label = batch_data[1]
                pred = self.forward(batch_data[0])
                torch.cuda.empty_cache()
                preds.append(pred)
                labels.append(label)
            pred_tensor = torch.cat(preds, dim=0)
            label_tensor = torch.cat(labels, dim=0)
            val_acc = accuracy_score(label_tensor, pred_tensor.cpu().argmax(dim=1))
            #print("label",label_tensor)
            #print("pred",pred_tensor.cpu().argmax(dim=1))
            val_loss = self.loss_fn(pred_tensor.to(self.device).log(), label_tensor.to(self.device))
            val_f1 = f1_score(label_tensor, pred_tensor.cpu().argmax(dim=1), average='macro')
            val_res = classification_report(label_tensor, pred_tensor.argmax(dim=1).cpu(), digits = 3)#0:unrelated  1:discuss  2:agree  3:disagree
        return val_loss, val_acc, val_f1, val_res

    def save_model(self, model_file):
        torch.save(
            {
                #"sent2vec": self.sent2vec.state_dict(),
                "prop_cell": self.prop_cell.state_dict(),
                "cls": self.cls.state_dict()
            },
            model_file
        )

    def load_model(self, model_file):
        if os.path.exists(model_file):
            checkpoint = torch.load(model_file)
            #self.sent2vec.load_state_dict(checkpoint['sent2vec'])
            self.cls.load_state_dict(checkpoint["cls"])
            self.prop_cell.load_state_dict(checkpoint['prop_cell'])
        else:
            print("Error: pretrained file %s is not existed!" % model_file)
            sys.exit()


# 1 load data
def loadData():
    trainPath = './data/train1.csv'
    testPath= './data/test1.csv'
    devPath= './data/dev1.csv'

    train_dataset = BotLoader(trainPath)
    test_dataset = BotLoader(testPath)
    dev_dataset = BotLoader(devPath)

    return train_dataset, test_dataset, dev_dataset

#加载模型
def obtain_RD(): 
    Prop_cell = BD(15, 15, Nclass)
    cls = nn.Linear(15, 2)
    model = Trainer(Prop_cell, cls, batch_size=5, grad_accum_cnt=4)
    return model

#1  load Data
train_set, test_set, dev_set = loadData()


#2  ini model
t0 = time.time()
model = obtain_RD()
t1 = time.time()
print('Recursive model established,', (t1 - t0) / 60)

#3  train&eval #choose a task randomly
best_valid_acc, best_test_acc, best_test_f1, best_valid_test_acc, best_valid_test_f1, best_cls_res = model.train_iters(train_set, test_set, dev_set,
        valid_every=2500, max_epochs=100, lr_discount=1.0, 
        best_valid_acc=0.0, best_test_acc=0.0, best_test_f1=0.0, best_valid_test_acc=0.0, best_valid_test_f1=0.0, log_dir="./logs/", model_file="./bd_model_tmp.pkl")

print('、')
print('best_test_acc', best_test_acc) 
print('best_test_f1',  best_test_f1)
print('best_valid_acc:', best_valid_acc)
print('best_valid_test_acc', best_valid_test_acc) 
print('best_valid_test_f1', best_valid_test_f1)
print(best_cls_res)
