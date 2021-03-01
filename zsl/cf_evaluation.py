import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import util
import classifier
from util import cal_macc
from lib import generate_syn_feature
from classifier import CLASSIFIER
from classifier2 import CLASSIFIER2
from binary_classifier import BINARY_CLASSIFIER
from knn_classifier import KNNClassifier
import os
from datetime import datetime
import pickle


class Evaluate():
    def __init__(self, netE, netG, netDec, netF, data, opt, model_file, exp_name, clf_epoch=5, alpha=1.0, siamese=False, netS=None):
        netG.eval()
        netDec.eval()
        netF.eval()
        netE.eval()
        self.netE = netE.cuda()
        self.netG = netG.cuda()
        self.netDec = netDec.cuda()
        if opt.feedback_loop == 1:
            self.netF = None
        else:
            self.netF = netF.cuda()
        self.data = data
        self.opt = opt
        self.model_file = model_file
        self.exp_name = exp_name
        self.epoch = clf_epoch
        if opt.concat_hy:
            self.cls_netDec = self.netDec
        else:
            self.cls_netDec = None
        self.alpha = alpha
        self.siamese = siamese
        self.netS = netS
        
    def conditional_sample(self, x, y, deterministic=False):
        # x is feature vector
        # y is attribute vector
        with torch.no_grad():
            if not self.opt.survae:
                means, log_var = self.netE(x, y)
                if deterministic:
                    z = means
                else:
                    z = torch.normal(means, torch.exp(0.5 * log_var))
            else:
                z, _ = self.netE(x, y)
            zv = Variable(z)
            yv = Variable(y)
            x_gen = self.netG(zv, c=yv)
            if self.netF is not None:
                _ = self.netDec(x_gen)
                dec_hidden_feat = self.netDec.getLayersOutDet()  # no detach layers
                feedback_out = self.netF(dec_hidden_feat)
                x_gen = self.netG(zv, a1=self.opt.a2, c=yv, feedback_layers=feedback_out)
        return x_gen

    def generate_syn_feature_cf(self, x, classes, deterministic=False):
        attribute = self.data.attribute
        nclass = classes.size(0)
        opt = self.opt
        num = opt.syn_num
        syn_feature = torch.zeros(nclass * num, opt.resSize).cuda()
        syn_label = torch.zeros(nclass*num).long().cuda()
        syn_att = torch.zeros(num, opt.attSize).float().cuda()
        syn_noise = torch.zeros(num, opt.nz).float().cuda()
        with torch.no_grad():
            for i in range(nclass):
                iclass = classes[i]
                iclass_att = attribute[iclass]
                if not self.opt.survae:
                    means, log_var = self.netE(x.unsqueeze(0), iclass_att.unsqueeze(0))
                    means = means.expand(num, -1)
                    log_var = log_var.expand(num, -1)
                    syn_att.copy_(iclass_att.repeat(num, 1))
                    if deterministic:
                        syn_noise = means
                    else:
                        syn_noise = torch.normal(means, torch.exp(0.5 * log_var))
                else:
                    syn_noise, _ = self.netE(x.unsqueeze(0), iclass_att.unsqueeze(0))
                    syn_noise = syn_noise.expand(num, -1)
                syn_noisev = Variable(syn_noise)
                syn_attv = Variable(syn_att)
                fake = self.netG(syn_noisev, c=syn_attv)
                if self.netF is not None:
                    dec_out = self.netDec(fake)  # only to call the forward function of decoder
                    dec_hidden_feat = self.netDec.getLayersOutDet()  # no detach layers
                    feedback_out = self.netF(dec_hidden_feat)
                    fake = self.netG(syn_noisev, a1=opt.a2, c=syn_attv, feedback_layers=feedback_out)
                output = fake
                syn_feature.narrow(0, i*num, num).copy_(output.data)
                syn_label.narrow(0, i*num, num).fill_(iclass)
        return syn_feature, syn_label

    def zsl(self, softmax_clf, cf, deterministic=False):
        opt = self.opt
        data = self.data
        if not cf:
            with torch.no_grad():
                gen_x, gen_l = generate_syn_feature(self.netG, self.data.unseenclasses, self.data.attribute, 
                                                    opt.syn_num, netF=self.netF, netDec=self.netDec, opt=opt)
            if softmax_clf:
                zsl_cls = classifier.CLASSIFIER(gen_x, util.map_label(gen_l, data.unseenclasses), \
                                data, data.unseenclasses.size(0), opt.cuda, opt.classifier_lr, 0.5, self.epoch, opt.syn_num, \
                                generalized=False, netDec=self.cls_netDec, dec_size=opt.attSize, dec_hidden_size=4096)
                acc = zsl_cls.acc
            else:
                zsl_cls = KNNClassifier(gen_x, gen_l, data.test_unseen_feature, self.cls_netDec, dec_size=opt.attSize, dec_hidden_size=4096, batch_size=100)
                preds = zsl_cls.fit()
                truths = data.test_unseen_label.cpu().numpy()
                acc = cal_macc(truth=truths, pred=preds)
        else:
            preds = []
            truths = []
            test_x = data.test_unseen_feature
            mapped_unseen_l = util.map_label(data.test_unseen_label, data.unseenclasses)
            unseen_label_np = data.test_unseen_label.cpu().numpy()
            for i in range(test_x.shape[0]):
                gen_x, gen_l = self.generate_syn_feature_cf(test_x[i], data.unseenclasses, deterministic=deterministic)
                gen_l = util.map_label(gen_l, data.unseenclasses)
                if softmax_clf:
                    clf = classifier.CLASSIFIER(gen_x, gen_l, data, data.unseenclasses.size(0), opt.cuda, opt.classifier_lr, 0.5,
                                 self.epoch, opt.syn_num, generalized=False, netDec=self.cls_netDec, dec_size=opt.attSize, dec_hidden_size=4096, x=test_x[i])
                    pred = clf.pred
                    truths.append(mapped_unseen_l[i])
                    preds.append(pred)
                else:
                    clf = KNNClassifier(gen_x, gen_l, test_x[i].unsqueeze(0), self.cls_netDec, dec_size=opt.attSize, dec_hidden_size=4096, batch_size=100)
                    pred = clf.fit()[0]
                    preds.append(pred)
                    truths.append(unseen_label_np[i])
                if (i + 1) % 500 == 0:
                    print("%dth acc: %.3f" % (i + 1, cal_macc(truth=truths, pred=preds)))
                if self.opt.sanity:
                    break   # Sanity check
            acc = cal_macc(truth=truths, pred=preds)
        return acc

    def two_stage(self, use_mask, use_tde, seen_mask=None, unseen_mask=None, save_clf=False):
        opt = self.opt
        data = self.data
        # Unseen:
        if unseen_mask is None:
            save_file = "out/%s-unseen.pickle" % use_mask
            with open(save_file, 'rb') as handle:
                clf_results = pickle.load(handle)
                preds = clf_results["preds"]
                mask = [pred in self.data.unseenclasses for pred in preds]
                mask = torch.from_numpy(np.array(mask).astype(int))
        else:
            mask = unseen_mask
        with torch.no_grad():
            gen_x, gen_l = generate_syn_feature(self.netG, self.data.unseenclasses, self.data.attribute, 
                                                    opt.u_num, netF=self.netF, netDec=self.netDec, opt=opt)
        if not save_clf or self.zsl_cls is None:
            zsl_cls = classifier.CLASSIFIER(gen_x, util.map_label(gen_l, data.unseenclasses), \
                                data, data.unseenclasses.size(0), opt.cuda, opt.u_lr, opt.u_beta, opt.u_epoch, opt.u_batch_size, \
                                generalized=False, netDec=self.cls_netDec, dec_size=opt.attSize, dec_hidden_size=4096, mask=mask,
                                use_tde=False, alpha=self.alpha)
            u_acc = zsl_cls.acc
        else:
            zsl_cls = self.zsl_cls
            u_acc = zsl_cls.val(zsl_cls.test_unseen_feature, zsl_cls.test_unseen_label, zsl_cls.unseenclasses, mask)
        if save_clf:
            self.zsl_cls = zsl_cls
        
        # Seen:
        if seen_mask is None:
            save_file = "out/%s-seen.pickle" % use_mask
            with open(save_file, 'rb') as handle:
                clf_results = pickle.load(handle)
                preds = clf_results["preds"]
                mask = [pred in self.data.seenclasses for pred in preds]
                mask = torch.from_numpy(np.array(mask).astype(int))
        else:
            mask = seen_mask

        if not opt.adjust_s:
            zsl_cls = classifier.CLASSIFIER(data.train_feature, util.map_label(data.train_label, data.seenclasses), \
                                data, data.seenclasses.size(0), opt.cuda, opt.classifier_lr, 0.5, 15, opt.syn_num, \
                                generalized=False, netDec=self.cls_netDec, dec_size=opt.attSize, dec_hidden_size=4096, mask=mask, zsl_on_seen=True,
                                use_tde=use_tde, alpha=self.alpha)
        else:
            zsl_cls = classifier.CLASSIFIER(data.train_feature, util.map_label(data.train_label, data.seenclasses), \
                                data, data.seenclasses.size(0), opt.cuda, opt.s_lr, opt.s_beta, opt.s_epoch, opt.s_batch_size, \
                                generalized=False, netDec=self.cls_netDec, dec_size=opt.attSize, dec_hidden_size=4096, mask=mask, zsl_on_seen=True,
                                use_tde=use_tde, alpha=self.alpha)
        s_acc = zsl_cls.acc
        h_acc = 2 * u_acc * s_acc / (u_acc + s_acc)
        
        if opt.log_two_stage:
            out_dir = "results/two_stage/%s/%s" % (opt.dataset, self.exp_name)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            mask_list = use_mask.split('/')
            mask_name = "%s_%s" % (mask_list[0], mask_list[1])
            out_file = os.path.join(out_dir, "%s.txt" % mask_name)
            config_msg = "u_num:%d u_lr:%.4f u_beta:%.2f u_epoch:%d u_batch_size:%d" % (opt.u_num, opt.u_lr, opt.u_beta, opt.u_epoch, opt.u_batch_size)
            if self.opt.adjust_s:
                config_msg += " s_lr:%.4f s_beta:%.2f s_epoch:%d s_bs:%d" % (opt.s_lr, opt.s_beta, opt.s_epoch, opt.s_batch_size)
            log_msg = "%s---S:%.3f U:%.3f H:%.3f\n" % (config_msg, s_acc, u_acc, h_acc)
            with open(out_file, "a") as f:
                f.write(log_msg)
        return s_acc, u_acc, h_acc

    def gzsl(self, use_train, softmax_clf, cf, deterministic=False, additional_train=False, use_tde=False, binary=False):
        opt = self.opt
        data = self.data
        if self.siamese:
            clf = SiameseClassifier(data, opt, self.netE, self.netG, self.netF, self.cls_netDec, dec_size=opt.attSize, cf=cf, n_epochs=opt.clf_epoch, distance="l1")
            if self.netS is None:
                clf.train()
            else:
                clf.network = self.netS
                s_acc, u_acc = clf.validate(gzsl=True)
        if not cf:
            with torch.no_grad():
                gen_x, gen_l = generate_syn_feature(self.netG, self.data.unseenclasses, self.data.attribute,
                                                    opt.syn_num, netF=self.netF, netDec=self.netDec, opt=opt)
            if use_train:
                train_x = torch.cat((data.train_feature, gen_x), 0)
                train_y = torch.cat((data.train_label, gen_l), 0)
            else:
                with torch.no_grad():
                    gen_s_x, gen_s_l = generate_syn_feature(self.netG, self.data.seenclasses, self.data.attribute,
                                                    opt.syn_num, netF=self.netF, netDec=self.netDec, opt=opt)
                train_x = torch.cat((gen_s_x, gen_x), 0)
                train_y = torch.cat((gen_s_l, gen_l), 0)
            if softmax_clf:
                if not binary:
                    gzsl_cls = classifier.CLASSIFIER(train_x, train_y, \
                                data, data.allclasses.size(0), opt.cuda, opt.classifier_lr, 0.5, self.epoch, opt.syn_num,
                                generalized=True, netDec=self.cls_netDec, dec_size=opt.attSize, dec_hidden_size=4096,
                                use_tde=use_tde, alpha=self.alpha)
                    self.test_logits = gzsl_cls.all_outputs
                else:
                    gzsl_cls = BINARY_CLASSIFIER(train_x, train_y, data, 2, True, opt.classifier_lr, 0.5, self.epoch, opt.syn_num,
                                netDec=self.cls_netDec, dec_size=opt.attSize, dec_hidden_size=4096, use_tde=use_tde, alpha=self.alpha)
                s_acc = gzsl_cls.acc_seen
                u_acc = gzsl_cls.acc_unseen
                h_acc = gzsl_cls.H
                self.s_bacc = gzsl_cls.s_bacc
                self.u_bacc = gzsl_cls.u_bacc
                if not binary:
                    clf_results = {
                        "preds": gzsl_cls.pred_s.cpu().numpy()
                    }
                save_file = self.get_save_result_file("seen")
                if self.log_to_file and not binary:
                    with open(save_file, 'wb') as handle:
                        pickle.dump(clf_results, handle)
                if not binary:
                    clf_results = {
                        "preds": gzsl_cls.pred_u.cpu().numpy()
                    }
                save_file = self.get_save_result_file("unseen")
                if self.log_to_file and not binary:
                    with open(save_file, 'wb') as handle:
                        pickle.dump(clf_results, handle)
            else:
                u_cls = KNNClassifier(train_x, train_y, data.test_unseen_feature, self.cls_netDec, dec_size=opt.attSize, dec_hidden_size=4096, batch_size=100)
                preds = u_cls.fit()
                truths = data.test_unseen_label.cpu().numpy()
                u_acc = cal_macc(truth=truths, pred=preds)

                s_cls = KNNClassifier(train_x, train_y, data.test_seen_feature, self.cls_netDec, dec_size=opt.attSize, dec_hidden_size=4096, batch_size=100)
                preds = s_cls.fit()
                truths = data.test_seen_label.cpu().numpy()
                s_acc = cal_macc(truth=truths, pred=preds)
                h_acc = 2 * u_acc * s_acc / (u_acc + s_acc)
        else:
            self.test_logits = None
            def cf_gzsl(test_x, test_l, split):
                preds = []
                truths = []
                test_l_np = test_l.cpu().numpy()
                test_l_binary = np.array([y in data.unseenclasses for y in test_l])
                if additional_train:
                    gen_sx, gen_sl = generate_syn_feature(self.netG, self.data.seenclasses, self.data.attribute,
                                                100, netF=self.netF, netDec=self.netDec, opt=opt)
                    #gen_sx = self.conditional_sample(data.train_feature, data.attribute[data.train_label], deterministic=False)
                    #gen_sx2 = self.conditional_sample(data.train_feature, data.attribute[data.train_label], deterministic=False)
                    #gen_sx3 = self.conditional_sample(data.train_feature, data.attribute[data.train_label], deterministic=False)
                    #gen_sx = torch.cat((gen_sx, gen_sx2, gen_sx3), 0)
                    #gen_sl = torch.cat((data.train_label.cuda(), data.train_label.cuda(), data.train_label.cuda()), 0)
                for i in range(test_x.shape[0]):
                    gen_x, gen_l = self.generate_syn_feature_cf(test_x[i], data.unseenclasses, deterministic=deterministic)
                    if use_train:
                        #if additional_train:
                        #    train_x = torch.cat((gen_sx, gen_x), 0)
                        #    train_y = torch.cat((gen_sl, gen_l), 0)
                        #else:
                        train_x = torch.cat((data.train_feature, gen_x), 0)
                        train_y = torch.cat((data.train_label.cuda(), gen_l), 0)
                    else:
                        gen_s_x, gen_s_l = self.generate_syn_feature_cf(test_x[i], data.seenclasses, deterministic=deterministic)
                        train_x = torch.cat((gen_s_x, gen_x), 0)
                        train_y = torch.cat((gen_s_l, gen_l), 0)
                    if additional_train:
                        train_x = torch.cat((train_x, gen_sx), 0)
                        train_y = torch.cat((train_y, gen_sl.cuda()), 0)
                    if softmax_clf:
                        if not binary:
                            clf = classifier.CLASSIFIER(train_x, train_y, data, self.opt.nclass_all, opt.cuda, opt.classifier_lr, opt.beta1,\
                                self.epoch, opt.syn_num, generalized=True, netDec=self.cls_netDec, dec_size=opt.attSize,
                                dec_hidden_size=4096, x=test_x[i], use_tde=use_tde, alpha=self.alpha)
                            if self.test_logits is None:
                                self.test_logits = clf.logits
                            else:
                                self.test_logits = np.concatenate((self.test_logits, clf.logits), axis=0)
                        else:
                            clf = BINARY_CLASSIFIER(train_x, train_y, data, 2, True, opt.classifier_lr, 0.5, self.epoch, opt.syn_num,
                                netDec=self.cls_netDec, dec_size=opt.attSize, dec_hidden_size=4096, use_tde=use_tde, alpha=self.alpha,
                                x=test_x[i])
                        pred = clf.pred
                        truths.append(test_l_np[i])
                        preds.append(pred.item())
                    else:
                        clf = KNNClassifier(train_x, train_y, test_x[i].unsqueeze(0), self.cls_netDec, dec_size=opt.attSize, dec_hidden_size=4096, batch_size=100)
                        pred = clf.fit()[0]
                        preds.append(pred)
                        truths.append(test_l_np[i])
                    if (i + 1) % 500 == 0:
                        if not binary:
                            binary_acc = self.get_binary_acc(truths, preds)
                            print("%s-%dth acc: %.3f, binary acc: %.3f" % (split, i + 1, cal_macc(truth=truths, pred=preds), binary_acc))
                        else:
                            test_l_binary_t = test_l_binary[:len(preds)].astype(int)
                            preds_np = np.array(preds)
                            acc = (preds_np == test_l_binary_t).mean()
                            print("%s-%dth binary acc: %.3f" % (split, i + 1, acc))
                    if self.opt.sanity:
                        break   # Sanity check
                if not binary:
                    acc = cal_macc(truth=truths, pred=preds)
                    binary_acc = self.get_binary_acc(truths, preds)
                    clf_results = {
                        "truths": truths,
                        "preds": preds
                    }
                else:
                    acc = (np.array(preds) == test_l_binary.astype(int)).mean()
                    binary_acc = acc
                    clf_results = {
                        "truths": test_l_binary,
                        "preds": preds
                    }
                save_file = self.get_save_result_file(split)
                if self.log_to_file:
                    with open(save_file, 'wb') as handle:
                        pickle.dump(clf_results, handle)
                return acc, binary_acc
            s_acc, s_bacc = cf_gzsl(data.test_seen_feature, data.test_seen_label, "seen")
            u_acc, u_bacc = cf_gzsl(data.test_unseen_feature, data.test_unseen_label, "unseen")
            
            # s_acc = 0.3
            if u_acc + s_acc == 0:
                h_acc = 0
            else:
                h_acc = 2 * u_acc * s_acc / (u_acc + s_acc)
            self.s_bacc = s_bacc
            self.u_bacc = u_bacc
        return s_acc, u_acc, h_acc

    def get_binary_acc(self, truths, preds):
        assert len(truths) == len(preds)
        correct_num = 0
        for i in range(len(truths)):
            if truths[i] in self.data.unseenclasses and preds[i] in self.data.unseenclasses:
                correct_num += 1
            if truths[i] in self.data.seenclasses and preds[i] in self.data.seenclasses:
                correct_num += 1
        return float(correct_num) / len(truths)

    def eval(self, gzsl=True, use_train=True, softmax_clf=True, cf=False, log_to_file=False, log_to_console=True, deterministic=False, additional_train=False, use_mask=None, use_tde=False, binary=False):
        test_config = {
            "gzsl": gzsl,
            "use_train": use_train,
            "softmax_clf": softmax_clf,
            "cf": cf,
            "deterministic": deterministic,
            "n_epoch": self.epoch,
            "concat_hy": self.opt.concat_hy,
            "feedback": (self.opt.feedback_loop == 2),
            "num": self.opt.syn_num,
            "additional_train": additional_train,
            "use_tde": use_tde,
            "alpha": self.alpha,
            "binary": binary,
            "beta": self.opt.beta1,
            "lr": self.opt.classifier_lr
        }
        if use_mask:
            assert not cf
        print_message = "%s-%s %s:" % (self.exp_name, self.model_file, str(test_config))
        self.test_config = test_config
        self.log_to_file = log_to_file
        if log_to_console:
            print(print_message)
        if not gzsl:
            acc = self.zsl(softmax_clf, cf, deterministic=deterministic)
            result_str = "ZSL Acc: %.3f" % acc
            self.acc = acc
        else:
            if use_mask is None:
                s_acc, u_acc, h_acc = self.gzsl(use_train, softmax_clf, cf, deterministic=deterministic, additional_train=additional_train,
                                                use_tde=use_tde, binary=binary)
                result_str = "GZSL S: %.3f, U: %.3f, H: %.3f, Sbacc: %.3f, Ubacc: %.3f" % (s_acc, u_acc, h_acc, self.s_bacc, self.u_bacc)
            else:
                s_acc, u_acc, h_acc = self.two_stage(use_mask, use_tde)
                result_str = "GZSL S: %.3f, U: %.3f, H: %.3f" % (s_acc, u_acc, h_acc)
            self.s_acc = s_acc
            self.u_acc = u_acc
            self.h_acc = h_acc
        if self.opt.sanity:
            log_to_file = False
        self.log(test_config, result_str, log_to_file)

    def save_auroc(self, cf=False):
        self.log_to_file = False
        self.test_config = {"task": "auroc"}
        outdir = "out/auroc/%s/%s" % (self.opt.dataset, self.exp_name)
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        s, u, h = self.gzsl(True, True, cf, False, False, False, False)
        print("%s save auroc: S%.3f U%.3f H%.3f" % (self.exp_name, s, u, h))
        save_file = os.path.join(outdir, "cf%s.pickle" % (cf))
        with open(save_file, 'wb') as handle:
            pickle.dump(self.test_logits, handle)

    def analyze_auroc(self, cf=False, exp_name=None):
        self.zsl_cls = None
        intervals = 20
        if exp_name is None:
            outdir = "out/auroc/%s/%s" % (self.opt.dataset, self.exp_name)
        else:
            outdir = "out/auroc/%s/%s" % (self.opt.dataset, exp_name)
        save_file = os.path.join(outdir, "cf%s.pickle" % (cf))
        with open(save_file, "rb") as handle:
            test_logits = pickle.load(handle)
        plot_save_dir = "auroc/%s" % (self.opt.dataset)
        if not os.path.exists(plot_save_dir):
            os.makedirs(plot_save_dir)
        test_seen_label = self.data.test_seen_label.cpu().numpy()
        test_unseen_label = self.data.test_unseen_label.cpu().numpy()
        seen_classes = self.data.seenclasses
        unseen_classes = self.data.unseenclasses
        ns = len(self.data.test_seen_feature)
        max_seen_logits = test_logits[:, seen_classes].max(axis=-1)
        max_unseen_logits = test_logits[:, unseen_classes].max(axis=-1)
        u_minus_s = max_unseen_logits - max_seen_logits
        max_diff = u_minus_s.max()
        min_diff = u_minus_s.min()
        interval_length = (max_diff - min_diff) / intervals
        current_diff = max_diff
        su_scores = []
        if not cf:
            for _ in range(intervals):
                current_logits = test_logits.copy()
                current_logits[:, seen_classes] += current_diff
                pred = np.argmax(current_logits, axis=-1)
                pred_s = pred[:ns]
                pred_u = pred[ns:]
                s = cal_macc(truth=test_seen_label, pred=pred_s)
                u = cal_macc(truth=test_unseen_label, pred=pred_u)
                su_scores.append((u, s))
                current_diff -= interval_length
        else:
            save_u_dir = "out/auroc/CUB"
            for _ in range(intervals):
                current_logits = test_logits.copy()
                current_logits[:, seen_classes] += current_diff
                pred = np.argmax(current_logits, axis=-1)
                pred_s = pred[:ns]
                pred_u = pred[ns:]
                if self.opt.save_auroc_mask:
                    save_file = "%s/%d.pickle" % (save_u_dir, _)
                    with open(save_file, 'wb') as handle:
                        pickle.dump({"preds": pred_u}, handle)
                s_mask = [pred in seen_classes for pred in pred_s]
                s_mask = torch.from_numpy(np.array(s_mask).astype(int))
                u_mask = [pred in unseen_classes for pred in pred_u]
                u_mask = torch.from_numpy(np.array(u_mask).astype(int))
                # print(u_mask.float().mean())
                if u_mask.float().mean() > 0.9994:
                    a = 1
                s, u, _ = self.two_stage("", False, s_mask, u_mask, save_clf=True)
                #if _ > 0 and u < su_scores[-1][0]:
                    #print(u, su_scores[-1][0])
                su_scores.append((u, s))
                current_diff -= interval_length
        su_scores = np.array(su_scores)
        for line in su_scores:
            print(line[0], line[1])
        area = 0.0
        for i in range(intervals - 1):
            area += (su_scores[i, 1] + su_scores[i + 1, 1]) * (su_scores[i + 1, 0] - su_scores[i, 0]) / 2.0
        print(area)

        import matplotlib.pyplot as plt
        plt.xlim(right=0.9)
        plt.ylim(top=1.0)
        plt.plot(su_scores[:, 0], su_scores[:, 1], '-', color='blue', label="ResNet Baseline")
        plt.savefig("auroc/%s/cf%s.png" % (self.opt.dataset, cf))
        

    def eval_dist(self, deterministic=False):
        test_s_x = self.data.test_seen_feature
        test_s_l = self.data.test_seen_label
        train_x = self.data.train_feature
        train_l = self.data.train_label
        accumulate_gt_dist = 0.0
        accumulate_unseen_min_dist = 0.0
        accumulate_unseen_mean_dist = 0.0
        for i in range(test_s_x.shape[0]):
            current_l = test_s_l[i]
            current_x = test_s_x[i]
            gen_x, gen_l = self.generate_syn_feature_cf(test_s_x[i], self.data.unseenclasses, deterministic=deterministic)
            gt_center = train_x[train_l == current_l].mean(dim=0)
            unseen_centers = [gen_x[gen_l == unseen_l].mean(dim=0) for unseen_l in self.data.unseenclasses]
            gt_dist = ((current_x - gt_center) ** 2).mean().item()
            unseen_dist = np.array([((current_x - uc) ** 2).mean().item() for uc in unseen_centers])
            accumulate_gt_dist += gt_dist
            accumulate_unseen_min_dist += unseen_dist.min()
            accumulate_unseen_mean_dist += unseen_dist.mean()
        print("GT mean: %.3f, Unseen mean: %.3f, Unseen min: %.3f" % (accumulate_gt_dist, accumulate_unseen_mean_dist, accumulate_unseen_min_dist))

    def log(self, test_config, result_str, log_to_file):
        print_message = "%s-%s %s: %s" % (self.exp_name, self.model_file, str(test_config), result_str)
        print(print_message)
        if log_to_file:
            out_dir = 'results/%s/%s' % (self.opt.dataset, self.exp_name)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            out_file = os.path.join(out_dir, "%s.txt" % self.model_file)
            timestamp = datetime.today().strftime('%m-%d')
            log_msg = "%s--%s: %s\n" % (timestamp, str(test_config), result_str)
            with open(out_file, "a") as f:
                f.write(log_msg)

    def get_config_str(self):
        test_config = self.test_config
        key_array = []
        for key in test_config.keys():
            key_array.append("%s_%s" % (key, test_config[key]))
        return "-".join(key_array)

    def get_save_result_file(self, split):
        result_dir = "out/%s/%s" % (self.exp_name, self.model_file)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        return "out/%s/%s/%s-%s.pickle" % (self.exp_name, self.model_file, self.get_config_str(), split)
