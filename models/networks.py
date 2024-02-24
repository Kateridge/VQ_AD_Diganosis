import torch
from munch import Munch
from torch import nn, einsum
from einops import rearrange
from einops.layers.torch import Rearrange
from torch.nn import AdaptiveAvgPool1d
from models.VQ_customed import Codebook
from models.VQ import VectorQuantize
from models.FGAN import Classifier, Generator
from models.minGPT import GPT, GPT_withoutTE
import torch.nn.functional as F

from models.pix2pixGAN import UnetGenerator


def build_models(args, model):
    if args.task == 'ADCN' or args.task == 'pMCIsMCI':
        class_num = 2
    else:
        class_num = 3
    if model == 'CNN_Single':
        # networks
        CNN = sNet(args.dim)
        CLS = nn.Linear(args.dim, class_num)
        nets = Munch(CNN=CNN, CLS=CLS)
    elif model == 'VQCNN_Single':
        # networks
        CNN = sNet(args.dim)
        # CODEBOOK = VectorQuantize(dim=args.dim, codebook_size=args.code_num, decay=args.codebook_ema,
        #                           commitment_weight=1, threshold_ema_dead_code=0, accept_image_fmap=True)
        CODEBOOK = Codebook(args.code_num, args.dim, args.beta)
        CLS = nn.Linear(args.dim, class_num)
        nets = Munch(CNN=CNN, CLS=CLS, CODEBOOK=CODEBOOK)
    elif model == 'CNN':
        # networks
        MRI_CNN = sNet(args.dim)
        PET_CNN = sNet(args.dim)
        CLS_Head = nn.Linear(args.dim * 2, class_num)
        nets = Munch(MRI=MRI_CNN, PET=PET_CNN, CLS=CLS_Head)
    elif model == 'VQCNN':
        # networks
        MRI_CNN = sNet(args.dim)
        MRICODEBOOK = Codebook(args.code_num, args.dim, args.beta)
        PET_CNN = sNet(args.dim)
        PETCODEBOOK = Codebook(args.code_num, args.dim, args.beta)
        CLS_Head = nn.Linear(args.dim * 2, class_num)
        nets = Munch(MRI=MRI_CNN, MRICODEBOOK=MRICODEBOOK, PET=PET_CNN, PETCODEBOOK=PETCODEBOOK, CLS=CLS_Head)
    elif model == 'MULTIMODEL_Transformer':
        # MRI networks
        MRI_CNN = sNet(args.dim)
        MRICODEBOOK = Codebook(args.code_num, args.dim, args.beta)
        MRI_CLS_Head = nn.Linear(args.dim, class_num)
        # PET networks
        PET_CNN = sNet(args.dim)
        PETCODEBOOK = Codebook(args.code_num, args.dim, args.beta)
        PET_CLS_Head = nn.Linear(args.dim, class_num)
        #
        transformer = GPT(args.code_num, 64, n_layer=6, n_head=4, n_embd=64,
                          embd_pdrop=0.1, resid_pdrop=0.1, attn_pdrop=0.1, n_unmasked=0)
        CLS_Head = nn.Linear(args.dim * 2, class_num)

        nets = Munch(MRICNN=MRI_CNN, MRICODEBOOK=MRICODEBOOK, MRICLS=MRI_CLS_Head,
                     PETCNN=PET_CNN, PETCODEBOOK=PETCODEBOOK, PETCLS=PET_CLS_Head,
                     Trans=transformer, CLS=CLS_Head)
    elif model == 'MULTIMODEL_Transformer_withoutVQ':
        # MRI networks
        MRI_CNN = sNet(args.dim)
        MRI_CLS_Head = nn.Linear(args.dim, class_num)
        # PET networks
        PET_CNN = sNet(args.dim)
        PET_CLS_Head = nn.Linear(args.dim, class_num)
        #
        transformer = GPT_withoutTE(args.code_num, 64, n_layer=6, n_head=4, n_embd=64,
                          embd_pdrop=0.1, resid_pdrop=0.1, attn_pdrop=0.1, n_unmasked=0)
        CLS_Head = nn.Linear(args.dim * 2, class_num)

        nets = Munch(MRICNN=MRI_CNN, MRICLS=MRI_CLS_Head, PETCNN=PET_CNN, PETCLS=PET_CLS_Head,
                     Trans=transformer, CLS=CLS_Head)
    elif model == 'Transformer':
        # networks
        MRI_CNN = sNet(args.dim)
        PET_CNN = sNet(args.dim)
        Cross_Enc = CrossTransformer(dim=args.dim, depth=args.trans_enc_depth, heads=args.heads_num,
                                     dim_head=args.dim // args.heads_num, mlp_dim=args.dim * 4, dropout=args.dropout)
        CLS_Head = nn.Linear(args.dim * 4, class_num)
        nets = Munch(MRI=MRI_CNN, PET=PET_CNN, Trans=Cross_Enc, CLS=CLS_Head)
    elif model == 'VQTransformer':
        # networks
        MRI_CNN = sNet(args.dim)
        PET_CNN = sNet(args.dim)
        MRICODEBOOK = Codebook(args.code_num, args.dim, args.beta)
        PETCODEBOOK = Codebook(args.code_num, args.dim, args.beta)
        Cross_Enc = CrossTransformer(dim=args.dim, depth=args.trans_enc_depth, heads=args.heads_num,
                                     dim_head=args.dim // args.heads_num, mlp_dim=args.dim * 4, dropout=args.dropout)
        CLS_Head = nn.Linear(args.dim * 4, class_num)
        nets = Munch(MRI=MRI_CNN, PET=PET_CNN, MRICODEBOOK=MRICODEBOOK, PETCODEBOOK=PETCODEBOOK,
                     Trans=Cross_Enc, CLS=CLS_Head)
    elif model == 'DeepGuidance':
        MRI_CNN = sNet(args.dim)
        MRI_CLS_Head = nn.Linear(args.dim, class_num)
        PET_CNN = sNet(args.dim)
        PET_CLS_Head = nn.Linear(args.dim, class_num)
        CLS_Head = nn.Linear(args.dim * 2, class_num)
        GuidanceModel = DeepGuidance()
        nets = Munch(MRICNN=MRI_CNN, PETCNN=PET_CNN, GUIDE=GuidanceModel,
                     MRICLS=MRI_CLS_Head, PETCLS=PET_CLS_Head, CLS=CLS_Head)
    elif model == 'pix2pixGAN':
        MRI_CNN = sNet(args.dim)
        PET_CNN = sNet(args.dim)
        CLS_Head = nn.Linear(args.dim * 2, class_num)
        GAN = UnetGenerator()
        nets = Munch(MRICNN=MRI_CNN, PETCNN=PET_CNN, GAN=GAN, CLS=CLS_Head)
    elif model == 'pix2pixGAN_Single':
        PET_CNN = sNet(args.dim)
        CLS_Head = nn.Linear(args.dim, class_num)
        GAN = UnetGenerator()
        nets = Munch(PETCNN=PET_CNN, GAN=GAN, CLS=CLS_Head)
    elif model == 'FGAN':
        MRI_CNN = Classifier()
        PET_CNN = Classifier()
        GAN = Generator()
        CLS = nn.Linear(5120*2, 2)
        nets = Munch(MRICNN=MRI_CNN, PETCNN=PET_CNN, GAN=GAN, CLS=CLS)
    elif model == 'FGAN_Single':
        PET_CNN = Classifier()
        CLS_Head = nn.Linear(5120, class_num)
        GAN = Generator()
        nets = Munch(PETCNN=PET_CNN, GAN=GAN, CLS=CLS_Head)
    else:
        nets = None
    return nets


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


class Swish(nn.Module):
    def __init__(self, inplace=True):
        super(Swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            x.mul_(torch.sigmoid(x))
            return x
        else:
            return x * torch.sigmoid(x)


class sNet(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, dim // 4, kernel_size=(3, 3, 3)),
            nn.BatchNorm3d(dim // 4),
            nn.ReLU(),
            nn.Conv3d(dim // 4, dim // 4, kernel_size=(3, 3, 3)),
            nn.BatchNorm3d(dim // 4),
            nn.ReLU(),
            nn.MaxPool3d(2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(dim // 4, dim // 2, kernel_size=(3, 3, 3)),
            nn.BatchNorm3d(dim // 2),
            nn.ReLU(),
            nn.Conv3d(dim // 2, dim // 2, kernel_size=(3, 3, 3)),
            nn.BatchNorm3d(dim // 2),
            nn.ReLU(),
            nn.MaxPool3d(2, stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(dim // 2, dim // 1, kernel_size=(3, 3, 3)),
            nn.BatchNorm3d(dim // 1),
            nn.ReLU(),
            nn.Conv3d(dim // 1, dim // 1, kernel_size=(3, 3, 3)),
            nn.BatchNorm3d(dim // 1),
            nn.ReLU(),
            nn.MaxPool3d(2, stride=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv3d(dim // 1, dim * 2, kernel_size=(3, 3, 3)),
            nn.BatchNorm3d(dim * 2),
            nn.ReLU(),
            nn.Conv3d(dim * 2, dim * 2, kernel_size=(3, 3, 3)),
            nn.BatchNorm3d(dim * 2),
            nn.ReLU(),
            nn.MaxPool3d(2, stride=2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv3d(dim * 2, dim, kernel_size=(1, 1, 1)),
            nn.BatchNorm3d(dim),
            nn.ReLU(),
        )

    def forward(self, mri):
        conv1_out = self.conv1(mri)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        conv4_out = self.conv4(conv3_out)
        conv5_out = self.conv5(conv4_out)
        return conv5_out


class sNet_NEW(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, dim // 4, kernel_size=(7, 7, 7), padding=1),
            nn.GroupNorm(8, dim // 4),
            Swish(),
            nn.MaxPool3d(2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(dim // 4, dim // 4, kernel_size=(3, 3, 3), padding=1),
            nn.GroupNorm(8, dim // 4),
            Swish(),
            nn.Conv3d(dim // 4, dim // 2, kernel_size=(3, 3, 3), padding=1),
            nn.GroupNorm(8, dim // 2),
            Swish(),
            nn.MaxPool3d(2, stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(dim // 2, dim // 2, kernel_size=(3, 3, 3), padding=1),
            nn.GroupNorm(8, dim // 2),
            Swish(),
            nn.Conv3d(dim // 2, dim // 1, kernel_size=(3, 3, 3), padding=1),
            nn.GroupNorm(8, dim // 1),
            Swish(),
            nn.MaxPool3d(2, stride=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv3d(dim // 1, dim * 2, kernel_size=(3, 3, 3), padding=1),
            nn.GroupNorm(8, dim * 2),
            Swish(),
            nn.Conv3d(dim * 2, dim * 2, kernel_size=(3, 3, 3), padding=1),
            nn.GroupNorm(8, dim * 2),
            Swish(),
            nn.MaxPool3d(2, stride=2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv3d(dim * 2, dim, kernel_size=(3, 3, 3), padding=1),
            nn.GroupNorm(8, dim),
            Swish(),
        )
        self.avg = nn.AvgPool3d(2, stride=2)

    def forward(self, mri):
        conv1_out = self.conv1(mri)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        conv4_out = self.conv4(conv3_out)
        conv5_out = self.conv5(conv4_out)
        return self.avg(conv5_out)


# pre-layernorm
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


# feedforward
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


# tabular-assisted feedforward
class T_Assisted_FeedForward(nn.Module):
    def __init__(self, img_dim, tabular_dim, hidden_dim):
        super().__init__()
        self.aux = nn.Sequential(
            nn.Linear(img_dim + tabular_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * img_dim)
        )
        self.scale_activation = nn.Tanh()
        self.global_pool = nn.Sequential(Rearrange('b n d -> b d n'), AdaptiveAvgPool1d(1),
                                         Rearrange('b d n -> b (d n)'))
        self.img_dim = img_dim

    def forward(self, img, context=None):
        squeeze = self.global_pool(img)  # b, d
        squeeze = torch.cat((squeeze, context), dim=1)  # b, d + d_t
        attention = self.aux(squeeze)  # b, 2d
        v_scale, v_shift = torch.split(attention, self.img_dim, dim=1)
        v_scale = v_scale.view(v_scale.size()[0], 1, v_scale.size()[1]).expand_as(img)
        # # activate to [-1,1]
        v_scale = self.scale_activation(v_scale)
        v_shift = v_shift.view(v_shift.size()[0], 1, v_shift.size()[1]).expand_as(img)
        return (v_scale * img) + v_shift


# attention
class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, kv_include_self=False):
        b, n, _ = x.shape
        h = self.heads
        context = default(context, x)

        if kv_include_self:
            # cross attention requires CLS token includes itself as key / value
            context = torch.cat((x, context), dim=1)

        qkv = (self.to_q(x), *self.to_kv(context).chunk(2, dim=-1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


# transformer encoder
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x, context=None):
        for attn, ff in self.layers:
            x = attn(x, context=context) + x
            x = ff(x) + x
        return self.norm(x)


class Transformer_T_Assited(nn.Module):
    def __init__(self, dim, tabular_dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(1):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, T_Assisted_FeedForward(dim, tabular_dim, mlp_dim))
            ]))

    def forward(self, x, tabular, context=None):
        for attn, ff in self.layers:
            x = attn(x, context=context) + x
            x = ff(x, context=tabular) + x
        return self.norm(x)


class CrossTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Transformer(dim, 1, heads, dim_head, mlp_dim, dropout=dropout),
                Transformer(dim, 1, heads, dim_head, mlp_dim, dropout=dropout)
            ]))

    def forward(self, mri_tokens, pet_tokens):
        for mri_enc, pet_enc in self.layers:
            mri_tokens = mri_enc(mri_tokens, context=torch.cat([mri_tokens, pet_tokens], dim=1))
            pet_tokens = pet_enc(pet_tokens, context=torch.cat([mri_tokens, pet_tokens], dim=1))
        return mri_tokens, pet_tokens


class DeepGuidance(nn.Module):
    def __init__(self):
        super(DeepGuidance, self).__init__()

        self.e1 = nn.Linear(in_features=2304, out_features=512)
        self.e2 = nn.Linear(in_features=512, out_features=256)

        self.d1 = nn.Linear(in_features=256, out_features=512)
        self.d2 = nn.Linear(in_features=512, out_features=2304)

        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = F.relu(self.e1(x))
        x = self.dropout(x)
        x = F.relu(self.e2(x))
        x = self.dropout(x)

        x = F.relu(self.d1(x))
        x = self.dropout(x)
        x = F.relu(self.d2(x))
        return x
