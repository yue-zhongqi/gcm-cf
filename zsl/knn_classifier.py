
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import util
from sklearn.preprocessing import MinMaxScaler 
import sys
import copy
import pdb


class KNNClassifier():
    def __init__(self, train_x, train_l, x, netDec=None, dec_size=4096, dec_hidden_size=4096, batch_size=100):
        self.train_x = train_x.clone()
        self.train_l = train_l.clone()
        self.x = x
        self.netDec = netDec
        self.input_dim = train_x.size(1)
        self.batch_size = batch_size
        self.cuda = True
        if self.netDec:
            self.netDec.eval()
            self.input_dim = self.input_dim + dec_size
            self.input_dim += dec_hidden_size
            self.train_x = self.compute_dec_out(self.train_x, self.input_dim)
            self.x = self.compute_dec_out(self.x, self.input_dim)

    def fit(self):
        preds = []
        for i in range(self.x.shape[0]):
            dist = ((self.train_x - self.x[i]) ** 2).mean(dim=1)
            pred_l = self.train_l[torch.argmin(dist)]
            preds.append(pred_l.item())
        return preds

    def compute_dec_out(self, test_X, new_size):
        start = 0
        ntest = test_X.size()[0]
        new_test_X = torch.zeros(ntest,new_size).cuda()
        for i in range(0, ntest, self.batch_size):
            end = min(ntest, start + self.batch_size)
            if self.cuda:
                inputX = Variable(test_X[start:end].cuda())
            else:
                inputX = Variable(test_X[start:end])
            with torch.no_grad():
                feat1 = self.netDec(inputX)
                feat2 = self.netDec.getLayersOutDet()
            new_test_X[start:end] = torch.cat([inputX,feat1,feat2],dim=1).data
            start = end
        return new_test_X