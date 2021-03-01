#import h5py
import numpy as np
import scipy.io as sio
import torch
from sklearn import preprocessing
from sklearn.decomposition import PCA
import sys
import pdb
import h5py


def get_exp_name(opt):
    cf_option_str = [opt.dataset]

    # Encoder use Y as input
    if opt.encoder_use_y:
        additional_str = "encyx"
    else:
        additional_str = "encx"
    cf_option_str.append(additional_str)

    # Train deterministic
    if opt.train_deterministic:
        additional_str = "deter"
        cf_option_str.append(additional_str)
        
    # Use zy disentangle
    if opt.zy_disentangle:
        additional_str = "zydscale%.3f" % (opt.zy_lambda)
        cf_option_str.append(additional_str)

    # Use yz disentangle
    if opt.yz_disentangle:
        additional_str = "yzdscale%.3f" % (opt.yz_lambda)
        if opt.yz_celoss:
            additional_str += "cel"
        cf_option_str.append(additional_str)

    # Use yx disentangle
    if opt.yx_disentangle:
        additional_str = "yxdscale%.3f" % (opt.yx_lambda)
        cf_option_str.append(additional_str)

    # Use zx disentangle
    if opt.zx_disentangle:
        additional_str = "zxdscale%.3f" % (opt.zx_lambda)
        cf_option_str.append(additional_str)

    # Use z disentangle
    if opt.z_disentangle:
        additional_str = "zdbeta%.1ftcvae%danneal%d" % (opt.zd_beta, opt.zd_tcvae, opt.zd_beta_annealing)
        cf_option_str.append(additional_str)

    # Use contrastive loss
    if opt.contrastive_loss:
        additional_str = "contrav%dscale%.1ft%.1f" % (opt.contra_v, opt.contra_lambda, opt.temperature)
        if opt.K != 30:
            additional_str += "K%d" % opt.K
        cf_option_str.append(additional_str)

    # No feedback loop
    if opt.feedback_loop == 1:
        additional_str = "nofeedback"
        cf_option_str.append(additional_str)

    # Encoded noise
    if not opt.encoded_noise:
        additional_str = "noise"
        cf_option_str.append(additional_str)

    # Siamese loss
    if opt.siamese_loss:
        additional_str = "siamese%.1fsoftmax%ddist%s" % (opt.siamese_lambda, opt.siamese_use_softmax, opt.siamese_distance)
        cf_option_str.append(additional_str)

    # Latent size
    if opt.latent_size != 312:
        additional_str = "latent%d" % (opt.latent_size)
        cf_option_str.append(additional_str)

    # Attr PCA
    if opt.pca_attribute != 0:
        additional_str = "pca%d" % (opt.pca_attribute)
        cf_option_str.append(additional_str)

    # SurVAE
    if opt.survae:
        additional_str = "survae%.1f" % opt.m_lambda
        cf_option_str.append(additional_str)

    # Add noise
    if opt.add_noise != 0.0:
        additional_str = "noise%.2f" % opt.add_noise
        cf_option_str.append(additional_str)
    
    # VAE Reconstruction loss
    if opt.recon != "bce":
        additional_str = "recon%s" % (opt.recon)
        cf_option_str.append(additional_str)
    
    # Att Dec use z
    if opt.attdec_use_z:
        additional_str = "attdecz"
        cf_option_str.append(additional_str)
    
    # Att Dec use MSE loss
    if opt.attdec_use_mse:
        additional_str = "attdecmse"
        cf_option_str.append(additional_str)

    # Unconstrained Z loss
    if opt.z_loss:
        additional_str = "zloss%.1f" % opt.z_loss_lambda
        cf_option_str.append(additional_str)

    # Prototype loss
    if opt.p_loss:
        additional_str = "ploss%.3f" % opt.p_loss_lambda
        cf_option_str.append(additional_str)

    # Additional str
    if opt.additional != "":
        cf_option_str.append(opt.additional)
    return "-".join(cf_option_str)

def loss_to_float(loss):
    if isinstance(loss, torch.Tensor):
        return loss.item()
    else:
        return float(loss)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def map_label(label, classes):
    mapped_label = torch.LongTensor(label.size())
    for i in range(classes.size(0)):
        mapped_label[label==classes[i]] = i    
    return mapped_label

class DATA_LOADER(object):
    def __init__(self, opt):
        self.read_matdataset(opt)
        self.index_in_epoch = 0
        self.epochs_completed = 0

    def read_matdataset(self, opt):
        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".mat")
        feature = matcontent['features'].T
        label = matcontent['labels'].astype(int).squeeze() - 1
        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.class_embedding + "_splits.mat")
        trainval_loc = matcontent['trainval_loc'].squeeze() - 1
        train_loc = matcontent['train_loc'].squeeze() - 1
        val_unseen_loc = matcontent['val_loc'].squeeze() - 1
        test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
        test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1

        self.attribute = torch.from_numpy(matcontent['att'].T).float()
        self.attribute /= self.attribute.pow(2).sum(1).sqrt().unsqueeze(1).expand(self.attribute.size(0),self.attribute.size(1))
        
        if opt.pca_attribute != 0:
            attribute = self.attribute.numpy()
            pca = PCA(n_components=opt.pca_attribute)
            attribute = pca.fit_transform(attribute)
            self.attribute = torch.from_numpy(attribute)
        self.attribute = self.attribute.cuda()

        if not opt.validation:
            if opt.preprocessing:
                if opt.standardization:
                    print('standardization...')
                    scaler = preprocessing.StandardScaler()
                else:
                    scaler = preprocessing.MinMaxScaler()
                
                _train_feature = scaler.fit_transform(feature[trainval_loc])
                _test_seen_feature = scaler.transform(feature[test_seen_loc])
                _test_unseen_feature = scaler.transform(feature[test_unseen_loc])
                self.train_feature = torch.from_numpy(_train_feature).float().cuda()
                mx = self.train_feature.max()
                self.train_feature.mul_(1/mx)
                self.train_label = torch.from_numpy(label[trainval_loc]).long()
                self.test_unseen_feature = torch.from_numpy(_test_unseen_feature).float().cuda()
                self.test_unseen_feature.mul_(1/mx)
                self.test_unseen_label = torch.from_numpy(label[test_unseen_loc]).long()
                self.test_seen_feature = torch.from_numpy(_test_seen_feature).float().cuda()
                self.test_seen_feature.mul_(1/mx)
                self.test_seen_label = torch.from_numpy(label[test_seen_loc]).long()
            else:
                self.train_feature = torch.from_numpy(feature[trainval_loc]).float().cuda()
                self.train_label = torch.from_numpy(label[trainval_loc]).long()
                self.test_unseen_feature = torch.from_numpy(feature[test_unseen_loc]).float().cuda()
                self.test_unseen_label = torch.from_numpy(label[test_unseen_loc]).long()
                self.test_seen_feature = torch.from_numpy(feature[test_seen_loc]).float().cuda()
                self.test_seen_label = torch.from_numpy(label[test_seen_loc]).long()
        else:
            self.train_feature = torch.from_numpy(feature[train_loc]).float().cuda()
            self.train_label = torch.from_numpy(label[train_loc]).long()
            self.test_unseen_feature = torch.from_numpy(feature[val_unseen_loc]).float().cuda()
            self.test_unseen_label = torch.from_numpy(label[val_unseen_loc]).long()
    
        self.seenclasses = torch.from_numpy(np.unique(self.train_label.numpy()))
        self.unseenclasses = torch.from_numpy(np.unique(self.test_unseen_label.numpy()))
        self.ntrain = self.train_feature.size()[0]
        self.ntest_seen = self.test_seen_feature.size()[0]
        self.ntest_unseen = self.test_unseen_feature.size()[0]
        self.ntrain_class = self.seenclasses.size(0)
        self.ntest_class = self.unseenclasses.size(0)
        self.train_class = self.seenclasses.clone()
        self.allclasses = torch.arange(0, self.ntrain_class + self.ntest_class).long()
        self.train_mapped_label = map_label(self.train_label, self.seenclasses)

    def next_seen_batch(self, seen_batch, return_label=False):
        idx = torch.randperm(self.ntrain)[0:seen_batch]
        batch_feature = self.train_feature[idx]
        batch_label = self.train_label[idx]
        batch_att = self.attribute[batch_label]
        if return_label:
            return batch_feature, batch_att, batch_label
        else:
            return batch_feature, batch_att

def cal_macc(*, truth, pred):
    assert len(truth) == len(pred)
    count = {}
    total = {}
    labels = list(set(truth))
    for label in labels:
        count[label] = 0
        total[label] = 0

    for y in truth:
        total[y] += 1

    correct = np.nonzero(np.asarray(truth) == np.asarray(pred))[0]

    for c in correct:
        idx = truth[c]
        count[idx] += 1

    macc = 0
    num_class = len(labels)
    for key in count.keys():
        if total[key] == 0:
            num_class -= 1
        else:
            macc += count[key] / total[key]
    macc /= num_class
    return macc