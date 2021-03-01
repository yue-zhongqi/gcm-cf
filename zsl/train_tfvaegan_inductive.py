
#tf-vaegan inductive
from __future__ import print_function
import os
import random
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import numpy as np
import math
import sys
from sklearn import preprocessing
from torch.nn.utils import clip_grad_norm_
import csv
#import functions
import model
import util
import classifier as classifier
from config import opt
from cf_evaluation import Evaluate
from lib import loss_fn, sample, WeightedL1, generate_syn_feature, calc_gradient_penalty, get_pretrain_classifier, contrastive_loss, zy_disentangle_loss, siamese_loss, mse_loss
from lib import yz_disentangle_loss, zx_disentangle_loss, yx_disentangle_loss, unconstrained_z_loss, get_p_loss
from tensorboardX import SummaryWriter
from tcvae.tcvae import anneal_kl
from survae.distributions import ConditionalNormal

if opt.debug:
    # architecture
    opt.encoder_use_y = True
    opt.feedback_loop = 2
    opt.encoded_noise = True

    # model
    opt.continue_from = 0

    # loss
    opt.zy_disentangle = False
    opt.zy_lambda = 0.1

    opt.yz_disentangle = False
    opt.yz_lambda = 1.0
    opt.yz_celoss = True

    opt.zx_disentangle = False
    opt.zx_lambda = 0.1

    opt.yx_disentangle = False
    opt.yx_lambda = 0.1

    opt.z_disentangle = False
    opt.zd_beta = 6.0
    opt.zd_tcvae = False
    opt.zd_beta_annealing = True
    
    opt.contrastive_loss = False
    opt.temperature = 500.0
    opt.contra_lambda = 1.0

    # test
    opt.syn_num = 300
    opt.concat_hy = 1
    opt.clf_epoch = 15

    opt.sanity = False
    # additional train mask
    #opt.use_mask = "CUB-encx-zdbeta6.0tcvae0anneal1-contrav3scale1.0t500.0-best-gzsl_True-use_train_True-softmax_clf_True-cf_True-deterministic_False-n_epoch_5-concat_hy_1-feedback_True-num_300-additional_train_True"

    # cf mask
    #opt.use_mask = "CUB-encx-zdbeta6.0tcvae0anneal1-contrav3scale1.0t500.0-best-gzsl_True-use_train_True-softmax_clf_True-cf_True-deterministic_False-n_epoch_5-concat_hy_1-feedback_True-num_300-additional_train_False"
    
    # baseline mask
    # opt.use_mask = "CUB-encx-best-gzsl_True-use_train_True-softmax_clf_True-cf_False-deterministic_False-n_epoch_15-concat_hy_1-feedback_True-num_300-additional_train_False"
    
    # Baseline mask
    #opt.use_mask = "CUB-encx/200/gzsl_True-use_train_1-softmax_clf_True-cf_False-deterministic_False-n_epoch_5-concat_hy_1-feedback_True-num_300-additional_train_False-use_tde_False-alpha_0.5-binary_False"

    # Cf mask 2
    #opt.use_mask = "CUB-encx-zdbeta6.0tcvae0anneal0/300/gzsl_True-use_train_1-softmax_clf_True-cf_True-deterministic_False-n_epoch_5-concat_hy_1-feedback_True-num_300-additional_train_False-use_tde_False-alpha_0.5-binary_False"

    opt.use_mask = "AWA2-encx-zdbeta6.0tcvae0anneal1-contrav3scale1.0t500.0-latent85-noise0.20/best34/gzsl_True-use_train_True-softmax_clf_True-cf_True-deterministic_False-n_epoch_2-concat_hy_1-feedback_True-num_1800-additional_train_False-use_tde_False-alpha_1.0-binary_False"
    
    opt.use_tde = False
    opt.tde_alpha = 0.5
    opt.binary = False
    opt.use_train = True

    opt.siamese = False          # Test using siamese
    opt.siamese_loss = False     # Train using siamese loss
    opt.siamese_lambda = 1.0
    opt.siamese_use_softmax = True
    opt.siamese_distance = "l1"
    # opt.latent_size = 100

    opt.pca_attribute = 0
    opt.survae = False

    opt.add_noise = 0

    opt.attdec_use_mse = False

    opt.yx_disentangle = False
    opt.yx_lambda = 0.01

    opt.p_loss = False
    opt.p_loss_lambda = 1.0

    opt.save_auroc = False
    opt.save_auroc_cf = False

    opt.analyze_auroc = True
    opt.analyze_auroc_cf = False
    #opt.analyze_auroc_expname = "CUB-encx-zdbeta6.0tcvae0anneal1-contrav3scale1.0t500.0-noise0.20"

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)
cudnn.benchmark = True
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# Dimension reduction on attributes
if opt.pca_attribute != 0:
    opt.attSize = opt.pca_attribute

# load data
data = util.DATA_LOADER(opt)
print("# of training samples: ", data.ntrain)
opt.ntrain = data.ntrain

exp_name = util.get_exp_name(opt)
print("Experiment: %s" % exp_name)

# Testing config
cf_eval_epochs = []
if opt.cf_eval != "":
    assert not opt.baseline
    split_str_list = opt.cf_eval.split(",")
    cf_eval_epochs = [int(e) for e in split_str_list]

# Model loading
save_path = "/data2/xxx/Model/tfvaegan/%s" % exp_name
if not os.path.exists(save_path):
    os.makedirs(save_path)

if opt.survae:
    survae = SurVAEAlpha6(opt.resSize, opt.attSize, opt.decoder_layer_sizes[0], a1=opt.a1)
    netE = SurVAE_Encoder(survae)
    netG = SurVAE_Generator(survae)
else:
    netE = model.Encoder(opt)
    netG = model.Generator(opt)
netD = model.Discriminator_D1(opt)
netF = model.Feedback(opt)
netDec = model.AttDec(opt, opt.attSize)
netS = None

# Load saved model
if opt.eval:
    if opt.continue_from > 0:
        best_model_file = os.path.join(save_path, "%d.pkl" % opt.continue_from)
        model_file_name = "%d" % opt.continue_from
    else:
        if not opt.load_best_acc:
            best_model_file = os.path.join(save_path, "best.pkl")
            model_file_name = "best"
        else:
            best_model_file = os.path.join(save_path, "best_acc.pkl")
            model_file_name = "best_acc"
    states = torch.load(best_model_file)
    if opt.survae:
        survae.load_state_dict(states["survae"])
        netE = SurVAE_Encoder(survae)
        netG = SurVAE_Generator(survae)
    else:
        netE.load_state_dict(states["netE"])
        netG.load_state_dict(states["netG"])
    netD.load_state_dict(states["netD"])
    netF.load_state_dict(states["netF"])
    netDec.load_state_dict(states["netDec"])
elif opt.continue_from > 0:
    model_file = os.path.join(save_path, "%d.pkl" % opt.continue_from)
    states = torch.load(model_file)
    if opt.survae:
        survae.load_state_dict(states["survae"])
        netE = SurVAE_Encoder(survae)
        netG = SurVAE_Generator(survae)
    else:
        netE.load_state_dict(states["netE"])
        netG.load_state_dict(states["netG"])
    netD.load_state_dict(states["netD"])
    netF.load_state_dict(states["netF"])
    netDec.load_state_dict(states["netDec"])

###########
# Init Tensors
input_res = torch.FloatTensor(opt.batch_size, opt.resSize)
input_att = torch.FloatTensor(opt.batch_size, opt.attSize)  # attSize class-embedding size
noise = torch.FloatTensor(opt.batch_size, opt.nz)
# one = torch.FloatTensor([1])
one = torch.tensor(1, dtype=torch.float)
mone = one * -1
##########
# Cuda
if opt.cuda:
    netD.cuda()
    netE.cuda()
    netF.cuda()
    netG.cuda()
    netDec.cuda()
    input_res = input_res.cuda()
    noise, input_att = noise.cuda(), input_att.cuda()
    one = one.cuda()
    mone = mone.cuda()

# Optimizer
if opt.survae:
    optimizer = optim.Adam(survae.parameters(), lr=1e-5, betas=(opt.beta1, 0.999))
    clip_grad_norm_(survae.parameters(), 20)
else:
    optimizer = optim.Adam(netE.parameters(), lr=opt.lr)
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerF = optim.Adam(netF.parameters(), lr=opt.feed_lr, betas=(opt.beta1, 0.999))
optimizerDec = optim.Adam(netDec.parameters(), lr=opt.dec_lr, betas=(opt.beta1, 0.999))

if opt.eval:
    # Test
    log_to_file = True and (not opt.debug)
    # log_to_file = True
    evaluate = Evaluate(netE, netG, netDec, netF, data, opt, model_file_name, exp_name, opt.clf_epoch, alpha=opt.tde_alpha, siamese=False, netS=netS)
    #evaluate.eval(gzsl=True, use_train=True, softmax_clf=True, cf=False, log_to_file=log_to_file, deterministic=opt.test_deterministic)
    #evaluate.eval(gzsl=False, use_train=True, softmax_clf=True, cf=False, log_to_file=log_to_file, deterministic=opt.test_deterministic)
    #evaluate.eval_dist()
    if opt.save_auroc:
        evaluate.save_auroc(opt.save_auroc_cf)
    elif opt.analyze_auroc:
        evaluate.analyze_auroc(opt.analyze_auroc_cf, opt.analyze_auroc_expname)
    else:
        if not opt.two_stage:
            evaluate.eval(gzsl=True, use_train=opt.use_train, softmax_clf=True, cf=False, log_to_file=log_to_file, deterministic=opt.test_deterministic, additional_train=False, use_tde=opt.use_tde, binary=opt.binary)
            
            evaluate.eval(gzsl=False, use_train=opt.use_train, softmax_clf=True, cf=False, log_to_file=log_to_file, deterministic=opt.test_deterministic, additional_train=False, use_tde=opt.use_tde, binary=opt.binary)
            if not opt.baseline:
                evaluate.eval(gzsl=True, use_train=opt.use_train, softmax_clf=True, cf=True, log_to_file=log_to_file, deterministic=opt.test_deterministic, additional_train=False, use_tde=opt.use_tde, binary=opt.binary)
            # evaluate.eval(gzsl=False, use_train=opt.use_train, softmax_clf=True, cf=True, log_to_file=log_to_file, deterministic=opt.test_deterministic, additional_train=False, use_tde=opt.use_tde, binary=opt.binary)
        
            #evaluate.eval(gzsl=False, use_train=True, softmax_clf=True, cf=True, log_to_file=log_to_file, deterministic=opt.test_deterministic)
            #evaluate.eval(gzsl=True, use_train=False, softmax_clf=False, cf=True, log_to_file=True)
        else:
            # Two stage
            evaluate.eval(gzsl=True, use_train=True, softmax_clf=True, cf=False, log_to_file=log_to_file, deterministic=opt.test_deterministic, use_mask=opt.use_mask,use_tde=opt.use_tde)
else:
    # Train loop
    # Additional modules
    if opt.p_loss:
        prototypes = torch.zeros(len(data.allclasses), opt.resSize).cuda()
        for label in data.seenclasses:
            classwise_samples = data.train_feature[data.train_label == label, :]
            assert classwise_samples.shape[0] > 0
            prototypes[label] = classwise_samples.mean(dim=0)
        for label in data.unseenclasses:
            classwise_samples = data.test_unseen_feature[data.test_unseen_label == label, :]
            assert classwise_samples.shape[0] > 0
            prototypes[label] = classwise_samples.mean(dim=0)

    if opt.zy_disentangle:
        net1 = nn.Sequential(nn.Linear(opt.attSize, 1000), nn.ReLU(), nn.Linear(1000, opt.latent_size * 2))
        z_encoder = ConditionalNormal(net1).cuda()
        optimizerZ = optim.Adam(z_encoder.parameters(), lr=0.01, betas=(opt.beta1, 0.999))
    if opt.yz_disentangle:
        if not opt.yz_celoss:
            net2 = nn.Sequential(nn.Linear(opt.latent_size, 1000), nn.ReLU(), nn.Linear(1000, opt.attSize * 2))
            yz_encoder = ConditionalNormal(net2).cuda()
            optimizerYZ = optim.Adam(yz_encoder.parameters(), lr=0.0001, betas=(opt.beta1, 0.999))
        else:
            yz_encoder = nn.Linear(opt.latent_size, data.ntrain_class).cuda()
            optimizerYZ = optim.Adam(yz_encoder.parameters(), lr=0.001, betas=(opt.beta1, 0.999))
    if opt.yx_disentangle:
        net3 = nn.Sequential(nn.Linear(opt.resSize, 2000), nn.ReLU(), nn.Linear(2000, opt.attSize * 2))
        yx_encoder = ConditionalNormal(net3).cuda()
        optimizerYX = optim.Adam(yx_encoder.parameters(), lr=0.00001, betas=(opt.beta1, 0.999))
    if opt.zx_disentangle:
        net4 = nn.Sequential(nn.Linear(opt.resSize, 2000), nn.ReLU(), nn.Linear(2000, opt.latent_size * 2))
        zx_encoder = ConditionalNormal(net4).cuda()
        optimizerZX = optim.Adam(zx_encoder.parameters(), lr=0.001, betas=(opt.beta1, 0.999))

    # Prepare summary writer
    if not opt.debug:
        log_dir = "runs/%s" % (exp_name)
        writer = SummaryWriter(log_dir)

    # Prepare loss specific modules
    if (opt.contrastive_loss and opt.contra_v != 3) or opt.z_loss:
        seen_clf = get_pretrain_classifier(data, opt)
    else:
        seen_clf = None

    best_gzsl_acc = 0
    best_zsl_acc = 0
    best_epoch = 0
    for epoch in range(opt.continue_from, opt.nepoch):
        # reset G to training mode
        netG.train()
        netDec.train()
        netF.train()
        netE.train()
        netD.train()
        D_cost_array = []
        G_cost_array = []
        WD_array = []
        vae_loss_array = []
        contrastive_loss_array = []
        zy_disentangle_loss_array = []
        yz_disentangle_loss_array = []
        zx_disentangle_loss_array = []
        yx_disentangle_loss_array = []
        unconstrained_z_loss_array = []
        p_loss_array = []
        siamese_loss_array = []
        assert netD.training
        assert netE.training
        assert netF.training
        assert netG.training
        assert netDec.training
        for loop in range(0, opt.feedback_loop):
            for i in range(0, data.ntrain, opt.batch_size):
                ######### Discriminator training ##############
                for p in netD.parameters():  # unfreeze discrimator
                    p.requires_grad = True
                for p in netDec.parameters():  # unfreeze deocder
                    p.requires_grad = True

                # Train D1 and Decoder (and Decoder Discriminator)
                gp_sum = 0  # lAMBDA VARIABLE
                for iter_d in range(opt.critic_iter):
                    input_res, input_att, input_l = sample(data, opt, return_label=True)
                    netD.zero_grad()
                    input_resv = Variable(input_res)
                    input_attv = Variable(input_att)

                    netDec.zero_grad()
                    recons = netDec(input_resv)
                    if opt.attdec_use_mse:
                        R_cost = nn.MSELoss()(recons, input_attv)
                    else:
                        R_cost = WeightedL1(recons, input_attv)
                    R_cost.backward()
                    optimizerDec.step()
                    criticD_real = netD(input_resv, input_attv)
                    criticD_real = opt.gammaD * criticD_real.mean()
                    criticD_real.backward(mone)
                    if opt.encoded_noise:
                        if not opt.survae:
                            means, log_var = netE(input_resv, input_attv)
                            std = torch.exp(0.5 * log_var)
                            eps = torch.randn([opt.batch_size, opt.latent_size]).cpu()
                            eps = Variable(eps.cuda())
                            z = eps * std + means  # torch.Size([64, 312])
                        else:
                            z, _ = netE(input_resv, input_attv)
                    else:
                        noise.normal_(0, 1)
                        z = Variable(noise)

                    if loop == 1:
                        fake = netG(z, c=input_attv)
                        dec_out = netDec(fake)
                        dec_hidden_feat = netDec.getLayersOutDet()
                        feedback_out = netF(dec_hidden_feat)
                        fake = netG(z, a1=opt.a1, c=input_attv, feedback_layers=feedback_out)
                    else:
                        fake = netG(z, c=input_attv)

                    criticD_fake = netD(fake.detach(), input_attv)
                    criticD_fake = opt.gammaD * criticD_fake.mean()
                    criticD_fake.backward(one)
                    # gradient penalty
                    gradient_penalty = opt.gammaD * calc_gradient_penalty(netD, input_res, fake.data, input_att, opt)
                    # if opt.lambda_mult == 1.1:
                    gp_sum += gradient_penalty.data
                    gradient_penalty.backward()
                    Wasserstein_D = criticD_real - criticD_fake
                    WD_array.append(util.loss_to_float(Wasserstein_D))
                    D_cost = criticD_fake - criticD_real + gradient_penalty  # add Y here and #add vae reconstruction loss
                    D_cost_array.append(util.loss_to_float(D_cost))
                    optimizerD.step()
                gp_sum /= (opt.gammaD * opt.lambda1 * opt.critic_iter)
                if (gp_sum > 1.05).sum() > 0:
                    opt.lambda1 *= 1.1
                elif (gp_sum < 1.001).sum() > 0:
                    opt.lambda1 /= 1.1

                ############# Generator training ##############
                # Train Generator and Decoder
                for p in netD.parameters():  # freeze discrimator
                    p.requires_grad = False
                if opt.recons_weight > 0 and opt.freeze_dec:
                    for p in netDec.parameters():  # freeze decoder
                        p.requires_grad = False

                netE.zero_grad()
                netG.zero_grad()
                netF.zero_grad()
                input_resv = Variable(input_res)
                input_attv = Variable(input_att)
                if not opt.survae:
                    means, log_var = netE(input_resv, input_attv)
                    std = torch.exp(0.5 * log_var)
                    eps = torch.randn([opt.batch_size, opt.latent_size]).cpu()
                    eps = Variable(eps.cuda())
                    z = eps * std + means  # torch.Size([64, 312])
                else:
                    z, _ = netE(input_resv, input_attv)
                if loop == 1:
                    recon_x = netG(z, c=input_attv)
                    dec_out = netDec(recon_x)
                    dec_hidden_feat = netDec.getLayersOutDet()
                    feedback_out = netF(dec_hidden_feat)
                    recon_x = netG(z, a1=opt.a1, c=input_attv, feedback_layers=feedback_out)
                else:
                    recon_x = netG(z, c=input_attv)
                    feedback_out = None

                beta = anneal_kl(opt, epoch)
                if not opt.survae:
                    input_eps = torch.randn([opt.batch_size, opt.resSize]).cpu().cuda()
                    hard_input = input_resv + opt.add_noise * input_eps
                    vae_loss_seen = loss_fn(recon_x, hard_input, means, log_var, opt, z, beta=beta)  # minimize E 3 with this setting feedback will update the loss as well
                else:
                    vae_loss_seen = netE.survae.vae_loss(input_resv, input_attv, feedback=feedback_out)
                    m_loss = mse_loss(input_resv, recon_x) * opt.m_lambda
                    if m_loss == m_loss:
                        vae_loss_seen += m_loss
                errG = vae_loss_seen
                vae_loss_array.append(util.loss_to_float(vae_loss_seen))

                # Contrastive loss
                if opt.contrastive_loss:
                    if loop == 1:
                        input_netF = netF
                    elif loop == 0:
                        input_netF = None
                    else:
                        assert False
                    contra_loss = contrastive_loss(netE, input_netF, netG, netDec, seen_clf, input_resv, input_l, data, opt,
                                                   deterministic=opt.train_deterministic, temperature=opt.temperature, K=opt.K)
                    contrastive_loss_array.append(util.loss_to_float(contra_loss))
                    contra_loss *= opt.contra_lambda
                    contra_loss.backward(retain_graph=True)

                # p_loss
                if opt.p_loss:
                    p_loss = get_p_loss(netE, netF, netG, netDec, recon_x, input_attv, prototypes[input_l], opt)
                    p_loss_array.append(util.loss_to_float(p_loss))
                    p_loss *= opt.p_loss_lambda
                    p_loss.backward(retain_graph=True)

                # ZY disentangle loss
                if opt.zy_disentangle:
                    disentangle_loss = zy_disentangle_loss(z_encoder, optimizerZ, z, input_attv)
                    zy_disentangle_loss_array.append(util.loss_to_float(disentangle_loss))
                    disentangle_loss *= opt.zy_lambda
                    disentangle_loss.backward(retain_graph=True)
                # YZ disentangle loss
                if opt.yz_disentangle:
                    disentangle_loss = yz_disentangle_loss(yz_encoder, optimizerYZ, input_attv, z, epoch, input_l, data, opt)
                    if disentangle_loss != disentangle_loss:
                        disentangle_loss = 0
                    yz_disentangle_loss_array.append(util.loss_to_float(disentangle_loss))
                    if disentangle_loss != 0:
                        disentangle_loss *= opt.yz_lambda
                        disentangle_loss.backward(retain_graph=True)
                # YX disentangle loss
                if opt.yx_disentangle:
                    disentangle_loss = yx_disentangle_loss(yx_encoder, optimizerYX, input_attv, recon_x)
                    if disentangle_loss != disentangle_loss:
                        disentangle_loss = 0
                    yx_disentangle_loss_array.append(util.loss_to_float(disentangle_loss))
                    if disentangle_loss != 0:
                        disentangle_loss *= opt.yx_lambda
                        disentangle_loss.backward(retain_graph=True)
                # ZX disentangle loss
                if opt.zx_disentangle:
                    disentangle_loss = zx_disentangle_loss(zx_encoder, optimizerZX, z, recon_x)
                    if disentangle_loss != disentangle_loss:
                        disentangle_loss = 0
                    zx_disentangle_loss_array.append(util.loss_to_float(disentangle_loss))
                    if disentangle_loss != 0:
                        disentangle_loss *= opt.zx_lambda
                        disentangle_loss.backward(retain_graph=True)

                # Unconstrained Z loss
                if opt.z_loss:
                    z_loss = unconstrained_z_loss(data, seen_clf, netG, netF, netDec, opt)
                    unconstrained_z_loss_array.append(util.loss_to_float(z_loss))
                    z_loss *= opt.z_loss_lambda
                    z_loss.backward(retain_graph=True)

                if opt.encoded_noise:
                    criticG_fake = netD(recon_x, input_attv).mean()
                    fake = recon_x
                else:
                    noise.normal_(0, 1)
                    noisev = Variable(noise)
                    if loop == 1:
                        fake = netG(noisev, c=input_attv)
                        dec_out = netDec(recon_x)  # Feedback from Decoder encoded output
                        dec_hidden_feat = netDec.getLayersOutDet()
                        feedback_out = netF(dec_hidden_feat)
                        fake = netG(noisev, a1=opt.a1, c=input_attv, feedback_layers=feedback_out)
                    else:
                        fake = netG(noisev, c=input_attv)
                    criticG_fake = netD(fake, input_attv).mean()

                G_cost = -criticG_fake
                errG += opt.gammaG * G_cost
                G_cost_array.append(util.loss_to_float(G_cost))
                netDec.zero_grad()
                recons_fake = netDec(fake)
                if opt.attdec_use_mse:
                    R_cost = nn.MSELoss()(recons_fake, input_attv)
                else:
                    R_cost = WeightedL1(recons_fake, input_attv)
                errG += opt.recons_weight * R_cost
                errG.backward()
                # write a condition here
                if opt.survae:
                    optimizer.step()
                else:
                    optimizer.step()
                    optimizerG.step()
                if loop == 1:
                    optimizerF.step()
                if opt.recons_weight > 0 and not opt.freeze_dec:  # not train decoder at feedback time
                    optimizerDec.step()
        print('[%d/%d]  Loss_D: %.4f Loss_G: %.4f, Wasserstein_dist:%.4f, vae_loss_seen:%.4f' %
               (epoch, opt.nepoch, D_cost.item(), G_cost.item(), Wasserstein_D.item(), vae_loss_seen.item()))

        # Log
        D_cost_mean = np.array(D_cost_array).mean()
        G_cost_mean = np.array(G_cost_array).mean()
        WD_mean = np.array(WD_array).mean()
        vae_loss_mean = np.array(vae_loss_array).mean()
        if not opt.debug:
            writer.add_scalar("d_cost", D_cost_mean, epoch)
            writer.add_scalar("g_cost", D_cost_mean, epoch)
            writer.add_scalar("wasserstein_dist", WD_mean, epoch)
            writer.add_scalar("vae_loss", vae_loss_mean, epoch)
            if opt.contrastive_loss:
                contrastive_loss_mean = np.array(contrastive_loss_array).mean()
                writer.add_scalar("contrastive_loss", contrastive_loss_mean, epoch)
            if opt.zy_disentangle:
                disentangle_loss_mean = np.array(zy_disentangle_loss_array).mean()
                writer.add_scalar("zy_disentangle_loss", disentangle_loss_mean, epoch)
            if opt.yz_disentangle:
                disentangle_loss_mean = np.array(yz_disentangle_loss_array).mean()
                writer.add_scalar("yz_disentangle_loss", disentangle_loss_mean, epoch)
                print("YZ: %.3f" % disentangle_loss_mean)
            if opt.zx_disentangle:
                disentangle_loss_mean = np.array(zx_disentangle_loss_array).mean()
                writer.add_scalar("zx_disentangle_loss", disentangle_loss_mean, epoch)
                print("ZX: %.3f" % disentangle_loss_mean)
            if opt.yx_disentangle:
                disentangle_loss_mean = np.array(yx_disentangle_loss_array).mean()
                writer.add_scalar("yx_disentangle_loss", disentangle_loss_mean, epoch)
            if opt.z_loss:
                z_loss_mean = np.array(unconstrained_z_loss_array).mean()
                writer.add_scalar("z_loss", z_loss_mean)
            if opt.p_loss:
                p_loss_mean = np.array(p_loss_array).mean()
                writer.add_scalar("p_loss", p_loss_mean)

        # Validation
        if (epoch + 1) % opt.val_interval == 0:
            netG.eval()
            netDec.eval()
            netF.eval()
            netE.eval()
            netD.eval()
            assert not netD.training
            assert not netE.training
            assert not netF.training
            assert not netG.training
            assert not netDec.training
            
            evaluate = Evaluate(netE, netG, netDec, netF, data, opt, "training", exp_name, clf_epoch=opt.clf_epoch)
            if opt.two_stage and opt.use_mask is not None:
                evaluate.eval(gzsl=True, use_train=True, softmax_clf=True, cf=False, log_to_console=False, deterministic=False, use_mask=opt.use_mask,use_tde=False)
            else:
                evaluate.eval(gzsl=True, use_train=True, softmax_clf=True, cf=False, log_to_console=False)
            evaluate.eval(gzsl=False, use_train=True, softmax_clf=True, cf=False, log_to_console=False)
            print('GZSL: seen=%.4f, unseen=%.4f, h=%.4f' % (evaluate.s_acc, evaluate.u_acc, evaluate.h_acc), end=" ")
            print('ZSL: unseen accuracy=%.4f' % (evaluate.acc))

            # Save best acc model
            if best_zsl_acc < evaluate.acc and (not opt.debug):
                best_zsl_acc = evaluate.acc
                # Save model
                states = {}
                states["netD"] = netD.state_dict()
                states["netF"] = netF.state_dict()
                states["netDec"] = netDec.state_dict()
                if opt.survae:
                    states["survae"] = survae.state_dict()
                else:
                    states["netE"] = netE.state_dict()
                    states["netG"] = netG.state_dict()
                torch.save(states, os.path.join(save_path, "best_acc.pkl"))
                
            # Save best H model
            if best_gzsl_acc < evaluate.h_acc and (not opt.debug):
                best_acc_seen, best_acc_unseen, best_gzsl_acc = evaluate.s_acc, evaluate.u_acc, evaluate.h_acc
                # Save model
                states = {}
                states["netD"] = netD.state_dict()
                states["netF"] = netF.state_dict()
                states["netDec"] = netDec.state_dict()
                if opt.survae:
                    states["survae"] = survae.state_dict()
                else:
                    states["netE"] = netE.state_dict()
                    states["netG"] = netG.state_dict()
                torch.save(states, os.path.join(save_path, "best.pkl"))
                best_epoch = epoch
            # Log
            if not opt.debug:
                writer.add_scalar("s_acc", evaluate.s_acc, epoch)
                writer.add_scalar("u_acc", evaluate.u_acc, epoch)
                writer.add_scalar("h_acc", evaluate.h_acc, epoch)
                writer.add_scalar("acc", evaluate.acc, epoch)

        if (epoch + 1) % opt.save_interval == 0 and (not opt.debug):
            # Save model
            states = {}
            states["netD"] = netD.state_dict()
            states["netF"] = netF.state_dict()
            states["netDec"] = netDec.state_dict()
            if opt.survae:
                states["survae"] = survae.state_dict()
            else:
                states["netE"] = netE.state_dict()
                states["netG"] = netG.state_dict()
            torch.save(states, os.path.join(save_path, "%d.pkl" % (epoch + 1)))
        
        if (epoch + 1) in cf_eval_epochs and (not opt.debug):
            # Cf evaluation
            evaluate = Evaluate(netE, netG, netDec, netF, data, opt, "%d" % (epoch + 1), exp_name, clf_epoch=opt.clf_epoch, siamese=False, netS=netS)
            evaluate.eval(gzsl=True, use_train=True, softmax_clf=True, cf=True, log_to_file=True, deterministic=opt.test_deterministic)

    print('Dataset', opt.dataset)
    print('the best ZSL unseen accuracy is', best_zsl_acc)
    print('Dataset', opt.dataset)
    print('the best GZSL seen accuracy is', best_acc_seen)
    print('the best GZSL unseen accuracy is', best_acc_unseen)
    print('the best GZSL H is', best_gzsl_acc)

    # Final evaluation
    best_model_file = os.path.join(save_path, "best.pkl")
    states = torch.load(best_model_file)
    if opt.survae:
        survae.load_state_dict(states["survae"])
        netE = SurVAE_Encoder(survae)
        netG = SurVAE_Generator(survae)
    else:
        netE.load_state_dict(states["netE"])
        netG.load_state_dict(states["netG"])
    netD.load_state_dict(states["netD"])
    netF.load_state_dict(states["netF"])
    netDec.load_state_dict(states["netDec"])
    
    # Note that here we will use softmax testing, not using two stage or siamese, for example
    evaluate = Evaluate(netE, netG, netDec, netF, data, opt, "best%d" % (best_epoch), exp_name, clf_epoch=opt.clf_epoch, siamese=False, netS=netS)
    # Baseline (has not deterministic sampling)
    evaluate.eval(gzsl=True, use_train=True, softmax_clf=True, cf=False, log_to_file=True)
    evaluate.eval(gzsl=False, use_train=True, softmax_clf=True, cf=False, log_to_file=True)
    # Counterfactual
    if not opt.baseline:
        evaluate.eval(gzsl=False, use_train=True, softmax_clf=True, cf=True, log_to_file=True, deterministic=opt.test_deterministic)
        evaluate.eval(gzsl=True, use_train=True, softmax_clf=True, cf=True, log_to_file=True, deterministic=opt.test_deterministic)
    #evaluate.eval(gzsl=True, use_train=False, softmax_clf=True, cf=True, log_to_file=True)
