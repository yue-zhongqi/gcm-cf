
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

class CLASSIFIER2:
    # train_Y is interger 
    def __init__(self, _train_X, _train_Y, data_loader, _nclass, _cuda, _lr=0.001, _beta1=0.5, _nepoch=20, _batch_size=100, generalized=True, netDec=None, dec_size=4096, dec_hidden_size=4096, x=None, c=None, mask_s=None, mask_u=None):
        self.train_X =  _train_X.clone() 
        self.train_Y = _train_Y.clone() 
        self.test_seen_feature = data_loader.test_seen_feature.clone()
        self.test_seen_label = data_loader.test_seen_label
        if x is None:
            self.test_unseen_feature = data_loader.test_unseen_feature.clone()
            self.test_unseen_label = data_loader.test_unseen_label
            self.x = None
        else:
            self.x = x.clone().unsqueeze(0)
            # self.c = c.unsqueeze(0)
        self.seenclasses = data_loader.seenclasses
        self.unseenclasses = data_loader.unseenclasses
        self.batch_size = _batch_size
        self.nepoch = _nepoch
        self.nclass = _nclass
        self.input_dim = _train_X.size(1)
        self.cuda = _cuda
        self.model =  LINEAR_LOGSOFTMAX_CLASSIFIER(self.input_dim, self.nclass)
        self.netDec = netDec
        if self.netDec:
            self.netDec.eval()
            self.input_dim = self.input_dim + dec_size
            self.input_dim += dec_hidden_size
            self.model = LINEAR_LOGSOFTMAX_CLASSIFIER(self.input_dim, self.nclass)
            self.train_X = self.compute_dec_out(self.train_X, self.input_dim)
            if x is None:
                self.test_unseen_feature = self.compute_dec_out(self.test_unseen_feature, self.input_dim)
                self.test_seen_feature = self.compute_dec_out(self.test_seen_feature, self.input_dim)
            else:
                self.x = self.compute_dec_out(self.x, self.input_dim)
                self.x = self.x.cuda()
        self.model.apply(util.weights_init)
        self.criterion = nn.NLLLoss()
        self.input = torch.FloatTensor(_batch_size, self.input_dim)
        self.label = torch.LongTensor(_batch_size)
        self.lr = _lr
        self.beta1 = _beta1
        self.optimizer = optim.Adam(self.model.parameters(), lr=_lr, betas=(_beta1, 0.999))
        if self.cuda:
            self.model.cuda()
            self.criterion.cuda()
            self.input = self.input.cuda()
            self.label = self.label.cuda()
        self.index_in_epoch = 0
        self.epochs_completed = 0
        self.ntrain = self.train_X.size()[0]
        self.mask_s = mask_s
        self.mask_u = mask_u
        if generalized:
            self.acc_seen, self.acc_unseen, self.H, self.epoch= self.fit()
            #print('Final: acc_seen=%.4f, acc_unseen=%.4f, h=%.4f' % (self.acc_seen, self.acc_unseen, self.H))
        else:
            self.acc,self.best_model = self.fit_zsl()
            #print('acc=%.4f' % (self.acc))

    def fit_zsl(self, mask=None):
        best_acc = 0
        mean_loss = 0
        last_loss_epoch = 1e8
        best_model = copy.deepcopy(self.model.state_dict())
        for epoch in range(self.nepoch):
            for i in range(0, self.ntrain, self.batch_size):      
                self.model.zero_grad()
                batch_input, batch_label = self.next_batch(self.batch_size) 
                self.input.copy_(batch_input)
                self.label.copy_(batch_label)
                   
                inputv = Variable(self.input)
                labelv = Variable(self.label)
                output = self.model(inputv)
                loss = self.criterion(output, labelv)
                mean_loss += loss.item()
                loss.backward()
                self.optimizer.step()
                #print('Training classifier loss= ', loss.data[0])
        if self.x is None:
            if not self.zsl_on_seen:
                acc = self.val(self.test_unseen_feature, self.test_unseen_label, self.unseenclasses, mask)
            else:
                acc = self.val(self.test_seen_feature, self.test_seen_label, self.seenclasses, mask)
            #print('acc %.4f' % (acc))
            if acc > best_acc:
                best_acc = acc
                best_model = copy.deepcopy(self.model.state_dict())
        else:
            with torch.no_grad():
                output = self.model(self.x)
                _, pred = torch.max(output.data, 1)
            self.pred = pred
            return 0, None
        return best_acc, best_model 
        
    def fit(self):
        best_H = 0
        best_seen = 0
        best_unseen = 0
        out = []
        best_model = copy.deepcopy(self.model.state_dict())
        # early_stopping = EarlyStopping(patience=20, verbose=True)
        for epoch in range(self.nepoch):
            for i in range(0, self.ntrain, self.batch_size):      
                self.model.zero_grad()
                batch_input, batch_label = self.next_batch(self.batch_size) 
                self.input.copy_(batch_input)
                self.label.copy_(batch_label)
                   
                inputv = Variable(self.input)
                labelv = Variable(self.label)
                output = self.model(inputv)
                loss = self.criterion(output, labelv)
                loss.backward()
                self.optimizer.step()
        if self.x is None:
            acc_seen = 0
            acc_unseen = 0
            acc_seen, s_bacc, pred_s = self.val_gzsl(self.test_seen_feature, self.test_seen_label, self.seenclasses, self.mask_s)
            acc_unseen, u_bacc, pred_u = self.val_gzsl(self.test_unseen_feature, self.test_unseen_label, self.unseenclasses, 1 - self.mask_u)
            self.s_bacc = s_bacc
            self.u_bacc = u_bacc
            self.pred_s = pred_s
            self.pred_u = pred_u
            H = 2 * acc_seen * acc_unseen / (acc_seen + acc_unseen)
            if H > best_H:
                best_seen = acc_seen
                best_unseen = acc_unseen
                best_H = H
        if self.x is not None:
            # Check if prediction is correct
            with torch.no_grad():
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

    def val_gzsl(self, test_X, test_label, target_classes, mask): 
        start = 0
        ntest = test_X.size()[0]
        predicted_label = torch.LongTensor(test_label.size())
        for i in range(0, ntest, self.batch_size):
            end = min(ntest, start+self.batch_size)
            if self.cuda:
                inputX = Variable(test_X[start:end].cuda())
            else:
                inputX = Variable(test_X[start:end])
            with torch.no_grad():
                output = self.model(inputX)
            for j in range(end - start):
                if mask[start + j]:
                    predicted_label[start + j] = self.seenclasses[torch.argmax(output.data[j, self.seenclasses]).item()]
                else:
                    predicted_label[start + j] = self.unseenclasses[torch.argmax(output.data[j, self.unseenclasses]).item()]
            start = end

        acc = self.compute_per_class_acc_gzsl(test_label, predicted_label, target_classes)
        bacc = self.compute_binary_acc_gzsl(test_label, predicted_label)
        return acc, bacc, predicted_label

    def compute_per_class_acc_gzsl(self, test_label, predicted_label, target_classes):
        acc_per_class = 0
        for i in target_classes:
            idx = (test_label == i)
            acc_per_class += torch.sum(test_label[idx]==predicted_label[idx]).float() / torch.sum(idx)
        acc_per_class /= target_classes.size(0)
        return acc_per_class

    def compute_binary_acc_gzsl(self, truths, preds):
        assert len(truths) == len(preds)
        correct_num = 0
        for i in range(len(truths)):
            if truths[i] in self.unseenclasses and preds[i] in self.unseenclasses:
                correct_num += 1
            if truths[i] in self.seenclasses and preds[i] in self.seenclasses:
                correct_num += 1
        return float(correct_num) / len(truths)

    # test_label is integer 
    def val(self, test_X, test_label, target_classes, mask=None):
        start = 0
        ntest = test_X.size()[0]
        predicted_label = torch.LongTensor(test_label.size())
        for i in range(0, ntest, self.batch_size):
            end = min(ntest, start+self.batch_size)
            if self.cuda:
                inputX = Variable(test_X[start:end].cuda())
            else:
                inputX = Variable(test_X[start:end])
            with torch.no_grad():
                output = self.model(inputX) 
            _, predicted_label[start:end] = torch.max(output.data, 1)
            start = end

        acc = self.compute_per_class_acc(util.map_label(test_label, target_classes), predicted_label, target_classes.size(0), mask)
        return acc

    def compute_per_class_acc(self, test_label, predicted_label, nclass, mask=None):
        acc_per_class = torch.FloatTensor(nclass).fill_(0)
        for i in range(nclass):
            idx = (test_label == i)
            if mask is None:
                acc_per_class[i] = torch.sum(test_label[idx]==predicted_label[idx]).float() / torch.sum(idx)
            else:
                acc_per_class[i] = torch.sum(torch.logical_and(test_label[idx]==predicted_label[idx], mask[idx])).float() / torch.sum(idx)
        return acc_per_class.mean() 

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