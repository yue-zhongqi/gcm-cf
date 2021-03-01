
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='FLO', help='FLO')
parser.add_argument('--dataroot', default='data', help='path to dataset')
parser.add_argument('--image_embedding', default='res101')
parser.add_argument('--class_embedding', default='att')
parser.add_argument('--syn_num', type=int, default=100, help='number features to generate per class')
parser.add_argument('--gzsl', action='store_true', default=False, help='enable generalized zero-shot learning')
parser.add_argument('--preprocessing', action='store_true', default=False, help='enbale MinMaxScaler on visual features')
parser.add_argument('--standardization', action='store_true', default=False)
parser.add_argument('--workers', type=int, help='number of data loading workers', default=8)
parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
parser.add_argument('--resSize', type=int, default=2048, help='size of visual features')
parser.add_argument('--attSize', type=int, default=1024, help='size of semantic features')
parser.add_argument('--nz', type=int, default=312, help='size of the latent z vector')
parser.add_argument('--ngh', type=int, default=4096, help='size of the hidden units in generator')
parser.add_argument('--ndh', type=int, default=1024, help='size of the hidden units in discriminator')
parser.add_argument('--nepoch', type=int, default=2000, help='number of epochs to train for')
parser.add_argument('--critic_iter', type=int, default=5, help='critic iteration, following WGAN-GP')
parser.add_argument('--lambda1', type=float, default=10, help='gradient penalty regularizer, following WGAN-GP')
parser.add_argument('--lambda2', type=float, default=10, help='gradient penalty regularizer, following WGAN-GP')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate to train GANs ')
parser.add_argument('--feed_lr', type=float, default=0.0001, help='learning rate to train GANs ')
parser.add_argument('--dec_lr', type=float, default=0.0001, help='learning rate to train GANs ')
parser.add_argument('--classifier_lr', type=float, default=0.001, help='learning rate to train softmax classifier')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', default=True, help='enables cuda')
parser.add_argument('--encoded_noise', action='store_true', default=False, help='enables validation mode')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--nclass_all', type=int, default=200, help='number of all classes')
parser.add_argument('--validation', action='store_true', default=False, help='enables validation mode')
parser.add_argument("--encoder_layer_sizes", type=list, default=[8192, 4096])
parser.add_argument("--decoder_layer_sizes", type=list, default=[4096, 8192])
parser.add_argument('--gammaD', type=int, default=1000, help='weight on the W-GAN loss')
parser.add_argument('--gammaG', type=int, default=1000, help='weight on the W-GAN loss')
parser.add_argument('--gammaG_D2', type=int, default=1000, help='weight on the W-GAN loss')
parser.add_argument('--gammaD2', type=int, default=1000, help='weight on the W-GAN loss')
parser.add_argument("--latent_size", type=int, default=312)
parser.add_argument("--conditional", action='store_true', default=True)
### 
parser.add_argument('--a1', type=float, default=1.0)
parser.add_argument('--a2', type=float, default=1.0)
parser.add_argument('--recons_weight', type=float, default=1.0, help='recons_weight for decoder')
parser.add_argument('--feedback_loop', type=int, default=2)
parser.add_argument('--freeze_dec', action='store_true', default=False, help='Freeze Decoder for fake samples')

# CF ZSL general
parser.add_argument('--val_interval', type=int, default=10, help='validation interval')
parser.add_argument('--save_interval', type=int, default=50, help='save interval')
parser.add_argument('--continue_from', type=int, default=0, help='save interval')
parser.add_argument("--debug", action="store_true", default=False, help="Turn on debug mode")
parser.add_argument('--additional', type=str, default="", help='additional str to add to exp name')

# CF ZSL Training
parser.add_argument("--encoder_use_y", action="store_true", default=False, help="Encoder use y as input")
parser.add_argument("--train_deterministic", action="store_true", default=False, help="Deterministic sampling during training")

parser.add_argument("--z_disentangle", action="store_true", default=False, help="Use z disentangle loss")
parser.add_argument("--zd_beta", type=float, default=1.0, help="beta for scaling KL loss")
parser.add_argument("--zd_tcvae", action="store_true", default=False, help="Use TCVAE")
parser.add_argument("--zd_beta_annealing", action="store_true", default=False, help="Slowly increase beta")

parser.add_argument("--zy_disentangle", action="store_true", default=False, help="Use disentangle loss for y->z")
parser.add_argument("--zy_lambda", type=float, default=0.01, help="Scaling factor for zy disentangling loss")

parser.add_argument("--yz_disentangle", action="store_true", default=False, help="Use disentangle loss for z->y")
parser.add_argument("--yz_lambda", type=float, default=0.01, help="Scaling factor for yz disentangling loss")
parser.add_argument("--yz_celoss", action="store_true", default=False, help="Use cross entropy loss for z->l")

parser.add_argument("--yx_disentangle", action="store_true", default=False, help="Use disentangle loss for x->y")
parser.add_argument("--yx_lambda", type=float, default=0.01, help="Scaling factor for yx disentangling loss")

parser.add_argument("--zx_disentangle", action="store_true", default=False, help="Use disentangle loss for x->z")
parser.add_argument("--zx_lambda", type=float, default=0.01, help="Scaling factor for zx disentangling loss")

parser.add_argument("--contrastive_loss", action="store_true", default=False, help="Use contrastive loss")
parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for contrastive loss")
parser.add_argument("--contra_lambda", type=float, default=1.0, help="Scaling factor of contrastive loss")
parser.add_argument("--contra_v", type=int, default=3, help="Version of contra loss to be used")
parser.add_argument("--K", type=int, default=30, help="Number of negative samples")

parser.add_argument("--siamese_loss", action="store_true", default=False, help="Train a Siamese network")
parser.add_argument("--siamese_lambda", type=float, default=1.0, help="Scaling factor for siamese network")
parser.add_argument("--siamese_use_softmax", action="store_true", default=False, help="Train a Siamese network")
parser.add_argument("--siamese_distance", type=str, default="l1", help="Distance metric for Siamese Net")

parser.add_argument("--pca_attribute", type=int, default=0, help="dimensionality reduction for attribute")

parser.add_argument('--survae', action='store_true', default=False, help='Use SurVAE model for encoder and generator')
parser.add_argument("--m_lambda", type=float, default=100.0, help="Strength for m_loss")

parser.add_argument("--add_noise", type=float, default=0.0, help="Add noise to reconstruction while training")

parser.add_argument("--recon", type=str, default="bce", help="VAE reconstruction loss: bce or l2 or l1")

parser.add_argument("--attdec_use_z", action="store_true", default=False, help="Use Z as additional input to attdec network")
parser.add_argument("--attdec_use_mse", action="store_true", default=False, help="Use MSE to calculate loss for attdec network")

parser.add_argument("--z_loss", action="store_true", default=False, help="Unconstrained z loss")
parser.add_argument("--z_loss_lambda", type=float, default=1.0, help="Scaling factor for unconstrained z loss")

parser.add_argument("--p_loss", action="store_true", default=False, help="Prototype loss")
parser.add_argument("--p_loss_lambda", type=float, default=1.0, help="Scaling factor for prototype loss")
#parser.add_argument()

# CF ZSL Testing
parser.add_argument('--eval', action='store_true', default=False, help='Evaluation mode')
parser.add_argument('--two_stage', action='store_true', default=False, help='Evaluation mode: Two stage')
parser.add_argument('--clf_epoch', type=int, default=5, help='Train epoch for softmax classifier')
parser.add_argument('--concat_hy', type=int, default=1, help='Concat h and y during classification')
parser.add_argument('--sanity', action='store_true', default=False, help='Test only 1 instance for sanity check')
parser.add_argument('--baseline', action='store_true', default=False, help='Only evaluate baseline')
parser.add_argument("--cf_eval", type=str, default="", help="Epochs to do counterfactual evaluation")
parser.add_argument("--report_softmax", action="store_true", default=False, help="Report softmax acc")
parser.add_argument("--report_knn", action="store_true", default=False, help="Report softmax acc")
parser.add_argument("--report_gzsl", action="store_true", default=False, help="Report gzsl acc")
parser.add_argument("--report_zsl", action="store_true", default=False, help="Report zsl acc")
parser.add_argument("--test_deterministic", action="store_true", default=False, help="Deterministic sampling during testing")
parser.add_argument("--use_mask", type=str, default=None, help="Mask name for two stage classifier")
parser.add_argument("--use_train", type=int, default=1, help="Use training data when testing")
parser.add_argument("--binary", action="store_true", default=False, help="Use binary classification")
parser.add_argument("--siamese", action="store_true", default=False, help="Use Siamese classifier")
parser.add_argument("--load_best_acc", action="store_true", default=False, help="Load the model with best zsl acc")

# AUROC/AUSUC
parser.add_argument("--save_auroc", action="store_true", default=False, help="Save auroc")
parser.add_argument("--save_auroc_cf", action="store_true", default=False, help="Use cf when saving auroc")
parser.add_argument("--analyze_auroc", action="store_true", default=False, help="Analyze auroc")
parser.add_argument("--analyze_auroc_cf", action="store_true", default=False, help="Use cf when analyzing auroc")
parser.add_argument("--analyze_auroc_expname", type=str, default=None, help="Exp name to load pre-saved auroc logits")
parser.add_argument("--save_auroc_mask", action="store_true", default=False, help="Save two stage auroc mask to file")

# TDE Classifier
parser.add_argument("--use_tde", action="store_true", default=False, help="Use TDE classifier")
parser.add_argument("--tde_alpha", type=float, default=0.5, help="Scaling factor of contrastive loss")

# Two stage testing parameters
parser.add_argument("--log_two_stage", action="store_true", default=False, help="Save two stage results to file")
parser.add_argument('--u_num', type=int, default=400, help='Number of generated unseen')
parser.add_argument('--u_lr', type=float, default=0.001, help='Classifier lr for unseen')
parser.add_argument('--u_beta', type=float, default=0.5, help='Classifier beta for unseen')
parser.add_argument('--u_epoch', type=int, default=2, help='Number of epochs for unseen')
parser.add_argument('--u_batch_size', type=int, default=400, help='Batch size for unseen classifier')

parser.add_argument("--adjust_s", action="store_true", default=False, help="Adjust two stage seen hyper")
parser.add_argument('--s_lr', type=float, default=0.001, help='Classifier lr for unseen')
parser.add_argument('--s_beta', type=float, default=0.5, help='Classifier beta for unseen')
parser.add_argument('--s_epoch', type=int, default=2, help='Number of epochs for unseen')
parser.add_argument('--s_batch_size', type=int, default=400, help='Batch size for unseen classifier')

opt = parser.parse_args()
opt.lambda2 = opt.lambda1
opt.encoder_layer_sizes[0] = opt.resSize
opt.decoder_layer_sizes[-1] = opt.resSize
# opt.latent_size = opt.attSize
