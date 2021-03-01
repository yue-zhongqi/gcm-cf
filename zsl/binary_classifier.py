
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import util
from sklearn.preprocessing import MinMaxScaler
from model import Causal_Norm_Classifier
import sys
import copy
import pdb

class BINARY_CLASSIFIER:
    # train_Y is interger 
    def __init__(self, _train_X, _train_Y, data_loader, _nclass, _cuda, _lr=0.001, _beta1=0.5, _nepoch=20, _batch_size=100, netDec=None, dec_size=4096, dec_hidden_size=4096, x=None, c=None, use_tde=False, alpha=3.0):
        self.train_X =  _train_X.clone()
        train_Y = _train_Y.clone()
        train_Y = [y.item() in data_loader.unseenclasses for y in train_Y]
        # Convert training label to binary
        self.train_Y = torch.from_numpy(np.array(train_Y)).to(_train_Y.device).long()
        self.test_seen_feature = data_loader.test_seen_feature.clone()
        self.test_seen_label = torch.zeros(data_loader.test_seen_label.shape).to(data_loader.test_seen_label.device)
        if x is None:
            self.test_unseen_feature = data_loader.test_unseen_feature.clone()
            self.test_unseen_label = data_loader.test_unseen_label
            self.test_unseen_label = torch.ones(data_loader.test_unseen_label.shape).to(data_loader.test_unseen_label.device)
            self.x = None
        else:
            self.x = x.clone().unsqueeze(0)
        self.seenclasses = data_loader.seenclasses
        self.unseenclasses = data_loader.unseenclasses
        self.batch_size = _batch_size
        self.nepoch = _nepoch
        self.nclass = 2
        self.input_dim = _train_X.size(1)
        self.cuda = _cuda
        self.use_tde = use_tde
        if not use_tde:
            self.model = LINEAR_LOGSOFTMAX_CLASSIFIER(self.input_dim, self.nclass)
            self.criterion = nn.NLLLoss()
        else:
            self.model = Causal_Norm_Classifier(self.nclass, self.input_dim, use_effect=True, num_head=2, tau=16.0, alpha=alpha, gamma=0.03125)
            self.embed_mean = torch.zeros(self.input_dim).numpy()
            self.mu = 0.995
            self.criterion = nn.CrossEntropyLoss()
        self.netDec = netDec
        if self.netDec:
            self.netDec.eval()
            self.input_dim = self.input_dim + dec_size
            self.input_dim += dec_hidden_size
            if not use_tde:
                self.model = LINEAR_LOGSOFTMAX_CLASSIFIER(self.input_dim, self.nclass)
                self.criterion = nn.NLLLoss()
            else:
                self.model = Causal_Norm_Classifier(self.nclass, self.input_dim, use_effect=True, num_head=2, tau=16.0, alpha=alpha, gamma=0.03125)
                self.embed_mean = torch.zeros(self.input_dim).numpy()
                self.mu = 0.995
                self.criterion = nn.CrossEntropyLoss()
            self.train_X = self.compute_dec_out(self.train_X, self.input_dim)
            if x is None:
                self.test_unseen_feature = self.compute_dec_out(self.test_unseen_feature, self.input_dim)
                self.test_seen_feature = self.compute_dec_out(self.test_seen_feature, self.input_dim)
            else:
                self.x = self.compute_dec_out(self.x, self.input_dim)
                self.x = self.x.cuda()
        self.model.apply(util.weights_init)
        
        self.input = torch.FloatTensor(_batch_size, self.input_dim)
        self.label = torch.LongTensor(_batch_size)
        self.lr = _lr
        self.beta1 = _beta1
        self.optimizer = optim.Adam(self.model.parameters(), lr=_lr, betas=(_beta1, 0.999))
        # self.optimizer = optim.SGD(self.model.parameters(), lr=_lr, momentum=0.9)
        if self.cuda:
            self.model.cuda()
            self.criterion.cuda()
            self.input = self.input.cuda()
            self.label = self.label.cuda()
        self.index_in_epoch = 0
        self.epochs_completed = 0
        self.ntrain = self.train_X.size()[0]
        self.acc_seen, self.acc_unseen, self.H, self.epoch= self.fit()
        self.s_bacc = self.acc_seen
        self.u_bacc = self.acc_unseen
        
    def fit(self):
        best_H = 0
        best_seen = 0
        best_unseen = 0
        out = []
        best_model = copy.deepcopy(self.model.state_dict())
        # early_stopping = EarlyStopping(patience=20, verbose=True)
        self.model.train()
        for epoch in range(self.nepoch):
            for i in range(0, self.ntrain, self.batch_size):      
                self.model.zero_grad()
                batch_input, batch_label = self.next_batch(self.batch_size) 
                self.input.copy_(batch_input)
                self.label.copy_(batch_label)
                   
                inputv = Variable(self.input)
                labelv = Variable(self.label)
                if self.use_tde:
                    output, _ = self.model(inputv, label=None, embed=self.embed_mean)
                    self.embed_mean = self.mu * self.embed_mean + batch_input.detach().mean(0).view(-1).cpu().numpy()
                else:
                    output = self.model(inputv)
                loss = self.criterion(output, labelv)
                loss.backward()
                self.optimizer.step()
        self.model.eval()
        if self.x is None:
            acc_seen = 0
            acc_unseen = 0
            acc_seen = self.val_gzsl(self.test_seen_feature, self.test_seen_label)
            acc_unseen = self.val_gzsl(self.test_unseen_feature, self.test_unseen_label)
            H = 2 * acc_seen * acc_unseen / (acc_seen + acc_unseen)
            if H > best_H:
                best_seen = acc_seen
                best_unseen = acc_unseen
                best_H = H
        if self.x is not None:
            # Check if prediction is correct
            with torch.no_grad():
                if self.use_tde:
                    output, _ = self.model(self.x, label=None, embed=self.embed_mean)
                else:
                    output = self.model(self.x)
                _, pred = torch.max(output.data, 1)
            self.pred = pred
            return 0, 0, 0, 0
        return best_seen, best_unseen, best_H, epoch
                     
    def next_batch(self, batch_size):
        start = self.index_in_epoch
        # shuffle the data at the first epoch
        if self.epochs_completed == 0 and start == 0:
            perm = torch.randperm(self.ntrain)
            self.train_X = self.train_X[perm]
            self.train_Y = self.train_Y[perm]
        # the last batch
        if start + batch_size > self.ntrain:
            self.epochs_completed += 1
            rest_num_examples = self.ntrain - start
            if rest_num_examples > 0:
                X_rest_part = self.train_X[start:self.ntrain]
                Y_rest_part = self.train_Y[start:self.ntrain]
            # shuffle the data
            perm = torch.randperm(self.ntrain)
            self.train_X = self.train_X[perm]
            self.train_Y = self.train_Y[perm]
            # start next epoch
            start = 0
            self.index_in_epoch = batch_size - rest_num_examples
            end = self.index_in_epoch
            X_new_part = self.train_X[start:end]
            Y_new_part = self.train_Y[start:end]
            #print(start, end)
            if rest_num_examples > 0:
                return torch.cat((X_rest_part, X_new_part), 0) , torch.cat((Y_rest_part, Y_new_part), 0)
            else:
                return X_new_part, Y_new_part
        else:
            self.index_in_epoch += batch_size
            end = self.index_in_epoch
            #print(start, end)
            # from index start to index end-1
            return self.train_X[start:end], self.train_Y[start:end]

    def val_gzsl(self, test_X, test_label): 
        start = 0
        ntest = test_X.size()[0]
        predicted_label = torch.LongTensor(test_label.size()).int()
        self.model.eval()
        for i in range(0, ntest, self.batch_size):
            end = min(ntest, start+self.batch_size)
            if self.cuda:
                inputX = Variable(test_X[start:end].cuda())
            else:
                inputX = Variable(test_X[start:end])
            with torch.no_grad():
                if self.use_tde:
                    output, _ = self.model(inputX, label=None, embed=self.embed_mean)
                else:
                    output = self.model(inputX)
            _, predicted_label[start:end] = torch.max(output.data, 1)
            start = end

        acc = (predicted_label == test_label).float().mean()
        return acc

    def compute_dec_out(self, test_X, new_size):
        start = 0
        ntest = test_X.size()[0]
        new_test_X = torch.zeros(ntest,new_size)
        for i in range(0, ntest, self.batch_size):
            end = min(ntest, start+self.batch_size)
            if self.cuda:
                inputX = Variable(test_X[start:end].cuda())
            else:
                inputX = Variable(test_X[start:end])
            with torch.no_grad():
                feat1 = self.netDec(inputX)
                feat2 = self.netDec.getLayersOutDet()
            new_test_X[start:end] = torch.cat([inputX,feat1,feat2],dim=1).data.cpu()
            start = end
        return new_test_X


class LINEAR_LOGSOFTMAX_CLASSIFIER(nn.Module):
    def __init__(self, input_dim, nclass):
        super(LINEAR_LOGSOFTMAX_CLASSIFIER, self).__init__()
        self.fc = nn.Linear(input_dim, nclass)
        self.logic = nn.LogSoftmax(dim=1)
    def forward(self, x): 
        o = self.logic(self.fc(x))
        return o