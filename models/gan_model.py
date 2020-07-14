import torch
import itertools

from losses import CosineSimilarityLoss
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from errors import NotEqualDimensionsError


class GANModel(BaseModel):
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
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--net_ref', type=str, default='bert-base-cased', help='specify generator architecture [bert-base-cased]')
            parser.add_argument('--type', type=str, default='A', help='specify the GAN model that should be used [A,B]') #A->concatenati e mandati insieme, B-> singolarmente
            parser.add_argument('--netG', type=str, default='bert-base-german-cased', help='specify generator architecture and language [bert-base-german-cased]')


            #parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            #parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            #parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D', 'G']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['fake_B']
        visual_names_B = ['real_B']
        if self.isTrain : # and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            visual_names_A.append('idt_B')

        #self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G', 'ref', 'D']
        else:  # during test time, only load Gs
            self.model_names = ['G', 'ref']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG = networks.define_G("encoder", opt.netG, '', '', opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        self.netref = networks.define_G("encoder", opt.net_ref, '', '', opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define discriminators
            if opt.type == 'A':
                in_dim = self.netG._modules['module'].get_word_embedding_dimension()
                in_dim += self.netref._modules['module'].get_word_embedding_dimension()
            elif opt.type == 'B':
                in_dim_G = self.netG._modules['module'].get_word_embedding_dimension()
                in_dim_ref = self.netref._modules['module'].get_word_embedding_dimension() + 1
                if in_dim_G != in_dim_ref :
                    raise NotEqualDimensionsError('The reference model and the generator have different output dimensions')
                in_dim=in_dim_G

            self.netD = networks.define_D(in_dim, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        torch.cuda.empty_cache()
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = {'input_ids': input['A' if AtoB else 'B'].to(self.device), 'attention_mask': (input['A' if AtoB else 'B'] > 0).to(self.device)}
        self.real_B = {'input_ids': input['B' if AtoB else 'A'].to(self.device), 'attention_mask': (input['B' if AtoB else 'A'] > 0).to(self.device)}

        self.sentence_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.generated = self.netG(self.real_B)  # G(A)
        self.landmark = self.netref(self.real_A)  # G_A(A)



    def backward_D(self):
        """Calculate GAN loss for discriminator D_A"""
        # Real
        if self.opt.type == "A":
            pred_real = self.netD(torch.cat([self.landmark['all_layer_embeddings'].detach(), self.generated['all_layer_embeddings'].detach()]))
            generated = torch.tensor([]).to(self.device)
            for i in range(len(self.generated['all_layer_embeddings'])):
                j = (i+1) % len(self.generated['all_layer_embeddings'])
                generated = torch.cat([generated, self.generated['all_layer_embeddings'][j]], dim=0)
            pred_fake = self.netD(generated.detach())
        elif self.opt.type == "B":
            pred_real = self.netD(self.landmark['all_layer_embeddings'].detach())
            pred_fake = self.netD(self.generated['all_layer_embeddings'].detach())

        loss_D_real = self.criterionGAN(pred_real, True)
        loss_D_fake = self.criterionGAN(pred_fake, False)

        # Combined loss and calculate gradients
        self.loss_D = (loss_D_real + loss_D_fake) * 0.5
        self.loss_D.backward()


    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        '''
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A['all_layer_embeddings'], self.real_B['all_layer_embeddings']) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0
        '''
        # GAN loss D(G(A))
        self.loss_G = self.criterionGAN(self.netD(self.generated['all_layer_embeddings']), True)
        # combined loss and calculate gradients
        #self.loss_G = self.loss_G
        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.netG._modules['module'].train()
        self.netD._modules['module'].train()

        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad(self.netD, False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D()      # calculate gradients for D_A
        #self.backward_D_B()      # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights


        self.netG._modules['module'].eval()
        self.netD._modules['module'].eval()
