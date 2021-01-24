import logging
from copy import deepcopy

import sklearn
import torch
import itertools

from losses import CosineSimilarityLoss
from losses import MSELoss
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import numpy as np
import time

class CycleGANModel(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.add_argument('--task', type=str, default='reconstruction',
                            help='specify the task of the CycleGAN [translation|reconstruction]')
        parser.add_argument('--encoder', type=str, default='bert-base',
                            help='specify generator architecture and language [marianMT|bert-base-german-cased]')
        parser.add_argument('--decoder', type=str, default='marianMT',
                            help='specify generator architecture and language [marianMT|bert-base-german-cased]')

        parser.add_argument('--language', type=str, default='it', help='specify destination language')

        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_G', type=float, default=10.0, help='scaling factor for generator loss')
            parser.add_argument('--lambda_D', type=float, default=10.0, help='scaling factor for discriminator loss')
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_C_1', type=float, default=5.0,
                                help='weight for embedding loss (fakeA -> fakeB)')  # aligment loss
            parser.add_argument('--lambda_C_2', type=float, default=5.0,
                                help='weight for embedding loss (fakeA -> recB, fakeB -> recA)')  # mixed loss
            parser.add_argument('--lambda_C_3', type=float, default=2.5,
                                help='weight for embedding loss (recA -> recB)')  # mixed loss (dubbio translation)
            parser.add_argument('--lambda_identity', type=float, default=0,
                                help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
            # 0.5
        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_AB', 'G_AB', 'cycle_ABA', 'D_BA', 'G_BA', 'cycle_BAB']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        #if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
        #    visual_names_A.append('idt_B')
        #    visual_names_B.append('idt_A')

        # self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_AB', 'G_BA', 'D_AB', 'D_BA']
        else:  # during test time, only load Gs
            self.model_names = ['G_AB', 'G_BA']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)

        self.netG_AB, self.netG_BA = networks.define_Gs(opt.task, opt.encoder, opt.decoder, opt.language, 'en', opt.norm,
                                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, opt.freeze_GB_encoder)

        if self.isTrain:  # define discriminators
            in_dim = self.netG_AB.module.get_word_embedding_dimension()

            netDAB_name = networks.define_name(opt.netD, 'en')
            netDBA_name = networks.define_name(opt.netD, opt.language)

            self.netD_AB = networks.define_D(in_dim, netDAB_name,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_BA = networks.define_D(in_dim, netDBA_name,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = CosineSimilarityLoss().to(self.device)
            self.criterionIdt = torch.nn.CosineEmbeddingLoss()  # CosineSimilarityLoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_AB.parameters(), self.netG_BA.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_AB.parameters(), self.netD_BA.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
        self.tempo_medio = 0
        self.n_iter = 0

        self.loss_G_AB = 0
        self.loss_G_BA = 0
        self.loss_D_AB = 0
        self.loss_D_BA = 0
        self.loss_cycle_ABA = 0
        self.loss_cycle_BAB = 0
        self.loss_G = 0

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        # torch.cuda.empty_cache()
        AtoB = self.opt.direction == 'AtoB'
        if self.opt.dataset_mode == "ParallelSentences":
            self.real_A = input['A' if AtoB else 'B']
            self.real_B = input['B' if AtoB else 'A']
        else:
            self.real_A = {'input_ids': input['A' if AtoB else 'B'].to(self.device),
                           'attention_mask': (input['A' if AtoB else 'B'] > 0).to(self.device)}
            self.real_B = {'input_ids': input['B' if AtoB else 'A'].to(self.device),
                           'attention_mask': (input['B' if AtoB else 'A'] > 0).to(self.device)}

        self.sentence_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):

        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B, self.fake_B_embeddings, self.loss_G_AB_1 = self.netG_AB(self.real_A, self.real_B, True)  # G_A(A)
        self.rec_A, self.rec_A_embeddings, self.loss_G_BA_1 = self.netG_BA(self.fake_B, self.real_A, True)  # G_B(G_A(A))
        self.fake_A, self.fake_A_embeddings, self.loss_G_BA_2 = self.netG_BA(self.real_B, self.real_A, True)  # G_B(B)
        self.rec_B, self.rec_B_embeddings, self.loss_G_AB_2 = self.netG_AB(self.fake_A, self.real_B, True)  # G_A(G_B(B))


    def backward_D_basic(self, netD, real_sent, fake_sent):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        #real = netD.module.batch_encode_plus(real_sent, verbose=False).to(self.device)

        #fake = netD.module.batch_encode_plus(fake_sent, verbose=False).to(self.device)

        # Real
        #pred_real = netD(real)
        loss_D_real = netD(real_sent, 1).loss
        # Fake
        loss_D_fake = netD(fake_sent, 0).loss
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5 * self.opt.lambda_D
        #print(loss_D)
        loss_D.backward()


        return loss_D#.item()

    def backward_D_AB(self):
        """Calculate GAN loss for discriminator D_A"""
        # fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_AB = self.backward_D_basic(self.netD_AB, self.real_B, self.fake_B)

    def backward_D_BA(self):
        """Calculate GAN loss for discriminator D_B"""
        # fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_BA = self.backward_D_basic(self.netD_BA, self.real_A, self.fake_A)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""

        lambda_G = self.opt.lambda_G
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        lambda_C_1 = self.opt.lambda_C_1
        #lambda_C_2 = self.opt.lambda_C_2
        #lambda_C_3 = self.opt.lambda_C_3


        self.loss_G_AB = self.netD_AB(self.fake_B, 1).loss

        self.loss_G_AB = (self.loss_G_AB + ((self.loss_G_AB_1 + self.loss_G_AB_2) * 0.5)) * lambda_G

        self.loss_G_BA = self.netD_BA(self.fake_A, 1).loss

        self.loss_G_BA = (self.loss_G_BA + ((self.loss_G_BA_1 + self.loss_G_BA_2) * 0.5)) * lambda_G

        # Forward cycle loss || G_B(G_A(A)) - A||
        size_vector = torch.ones(
            self.netG_AB.module.batch_encode_plus(self.real_A, verbose=False)["input_ids"].size()).to(self.device)
        real_A_tokens = self.netG_AB.module.batch_encode_plus(self.real_A, verbose=False)["input_ids"].to(self.device,
                                                                                                         dtype=torch.float32)
        rec_A_tokens = self.netG_AB.module.batch_encode_plus(self.rec_A, verbose=False)["input_ids"].to(self.device,
                                                                                                       dtype=torch.float32)
        self.loss_cycle_ABA = self.criterionCycle(real_A_tokens,
                                                rec_A_tokens,
                                                size_vector) * lambda_A

        # Backward cycle loss || G_A(G_B(B)) - B||
        real_B_tokens = self.netG_BA.module.batch_encode_plus(self.real_B, verbose=False)["input_ids"].to(self.device,
                                                                                                         dtype=torch.float32)
        rec_B_tokens = self.netG_BA.module.batch_encode_plus(self.rec_B, verbose=False)["input_ids"].to(self.device,
                                                                                                       dtype=torch.float32)

        self.loss_cycle_BAB = self.criterionCycle(real_B_tokens,
                                                rec_B_tokens,
                                                size_vector) * lambda_B

        size_vector = torch.ones(self.fake_A_embeddings.size()).to(self.device)

        # Backward cycle loss || G_B(B) - G_A(A)||

        loss_cycle_C_1 = self.criterionCycle(self.fake_A_embeddings,
                                             self.fake_B_embeddings,
                                             size_vector) * lambda_C_1

        '''
        # Backward cycle loss || G_B(B) - G_A(A)||
        loss_cycle_C_2_1 = self.criterionCycle(self.fake_A_embeddings,
                                               self.rec_B_embeddings,
                                               size_vector) * lambda_C_2

        # Backward cycle loss || G_B(B) - G_A(A)||
        loss_cycle_C_2_2 = self.criterionCycle(self.fake_B_embeddings,
                                               self.rec_A_embeddings,
                                               size_vector) * lambda_C_2

        # Backward cycle loss || G_B(B) - G_A(A)||
        loss_cycle_C_3 = self.criterionCycle(self.rec_A_embeddings,
                                             self.rec_B_embeddings,
                                             size_vector) * lambda_C_3
        '''
        # 'weight for embedding loss (fakeA -> recB, fakeB -> recA)')  # mixed loss
        # 'weight for embedding loss (recA -> recB)')  # mixed loss (dubbio translation)

        # combined loss and calculate gradients
        self.loss_cycle_ABA = self.loss_cycle_ABA + loss_cycle_C_1  # + loss_cycle_C_2_2 + loss_cycle_C_3
        self.loss_cycle_BAB = self.loss_cycle_BAB + loss_cycle_C_1  # + loss_cycle_C_2_1 + loss_cycle_C_3
        self.loss_G = self.loss_G_AB + self.loss_G_BA + self.loss_cycle_ABA + self.loss_cycle_BAB  # + self.loss_idt_A.item() + self.loss_idt_B.item()

        self.loss_G.backward()

        #self.loss_G_AB = self.loss_G_AB.item()
        #self.loss_G_BA = self.loss_G_BA.item()
        #self.loss_cycle_ABA = self.loss_cycle_ABA.item() + loss_cycle_C_1.item()
        #self.loss_cycle_BAB = self.loss_cycle_BAB.item() + loss_cycle_C_1.item()

        #del real_A_tokens
        #del rec_A_tokens
        #del real_B_tokens
        #del rec_B_tokens
        #del size_vector
        #del self.loss_G
        #torch.cuda.empty_cache()



    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward

        self.netG_AB.train()
        self.netG_BA.train()
        self.netD_AB.train()
        self.netD_BA.train()

        self.forward()  # compute fake images and reconstruction images.


        self.set_requires_grad([self.netD_AB, self.netD_BA], False)
        torch.enable_grad()

        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()  # calculate gradients for G_A and G_B
        self.optimizer_G.step()  # update G_A and G_B's weights

        # D_A and D_B
        self.set_requires_grad([self.netD_AB, self.netD_BA], True)

        self.optimizer_D.zero_grad()  # set D_A and D_B's gradients to zero
        self.backward_D_AB()  # calculate gradients for D_A
        self.backward_D_BA()  # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights

        del self.fake_A_embeddings
        del self.fake_B_embeddings
        del self.rec_A_embeddings
        del self.rec_B_embeddings
        del self.fake_A
        del self.fake_B
        del self.rec_A
        del self.rec_B
        torch.no_grad()
        torch.cuda.empty_cache()



    def evaluate(self, sentences_file="eval_sentences.txt", distance_file="distances.txt", top_k_file="top_k.txt"):
        logging.info("\n\nEvaluating...")

        self.netG_AB.module.eval()
        self.netG_BA.module.eval()
        self.netD_AB.module.eval()
        self.netD_BA.module.eval()
        self.forward()  # calculate loss functions, get gradients, update network weights
        with open(sentences_file, "a") as sentences_file:
            for j in range(len(self.real_A)):
                str1 = " A->B->A : " + self.real_A[j] + " -> " + self.fake_B[j] + " -> " + self.rec_A[j]
                str2 = " B->A->B : " + self.real_B[j] + " -> " + self.fake_A[j] + " -> " + self.rec_B[j]
                logging.info(str1)
                logging.info(str2)
                sentences_file.write('%s\n' % str1)  # save the message
                sentences_file.write('%s\n\n' % str2)  # save the message

        distances = sklearn.metrics.pairwise_distances(self.fake_A_embeddings.cpu().detach().numpy(),
                                                       self.fake_B_embeddings.cpu().detach().numpy(),
                                                       metric='cosine',
                                                       n_jobs=-1)

        with open(distance_file, "a") as distances_file:
            for i in range(len(distances)):
                distances_file.write(str(distances[i][i]) + '\n')

        #with open(distance_file, "a") as distances_file:
        #   distances_file.write(distances+"\n")

        dim = len(distances)
        top_k = np.zeros(dim, dtype=np.float)
        for i in range(dim):

            lower = 0
            for j in range(len(distances[i])):
                if i != j and distances[i][i] > distances[i][j]:
                    lower += 1
            top_k[lower] += 1

        with open(top_k_file, "a") as top_file:
            tot = 0
            for i in range(dim):
                top_k[i] = top_k[i] / dim * 100
                tot += top_k[i]
                top_file.write('Top ' + str(i + 1) + ': ' + str(tot) + '%\n')
        # mi salvo in un dict per ogni frase quanto lontano è l'embedding reale (quanti ce ne sono più vicini) e faccio una classifica
        # per vedere quanti hanno l'embedding reale nella top 1, top 2 e cosi via (cumulativo)
        # salvo info in un file, per ogni epoca

        self.netG_AB.module.train()
        self.netG_BA.module.train()
        self.netD_AB.module.train()
        self.netD_BA.module.train()
