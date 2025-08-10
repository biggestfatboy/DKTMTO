import math
import copy
import torch
from torch import nn, einsum
import torch.nn.functional as F
from inspect import isfunction
from functools import partial
from torch.utils import data
from pathlib import Path
from torch.optim import Adam
from torchvision import transforms, utils
from einops import rearrange
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class linear_time_emb(nn.Module):

    def __init__(self, dim, dim_out, time_emb_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.GELU(),
            nn.Linear(time_emb_dim, dim)
        )

        self.fc = nn.Sequential(
            nn.GELU(),
            nn.Linear(dim, dim_out)
        )
    def forward(self, x, time_emb):
        h = x
        condition = self.mlp(time_emb)
        h = h + condition
        h = self.fc(h)
        return h


class Unet_fc(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim*4),
            nn.GELU(),
            nn.Linear(dim*4, dim)
        )

        self.fc1 = linear_time_emb(dim, dim//2, dim)
        self.fc2 = linear_time_emb(dim//2,dim//4,dim)
        self.fc3 = linear_time_emb(dim//4,dim//2,dim)
        self.fc4 = linear_time_emb(dim , dim, dim)
    def forward(self,x , t):

        t = self.time_mlp(t)
        x1 = self.fc1(x,t)
        x2 = self.fc2(x1,t)
        x3 = self.fc3(x2,t)
        x4 = torch.cat((x1,x3),dim=1)
        x5 = self.fc4(x4,t)

        return x5


def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, steps, steps)
    alphas_cumprod = torch.cos(((x / steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

class pop_diffusion(nn.Module):
    def __init__(self, denoise_fn, pop_dim, time_steps = 10, loss_type = 'l1'):
        super().__init__()
        self.pop_dim = pop_dim
        self.denoise_fn = denoise_fn
        self.num_timesteps = int(time_steps)
        self.loss_type = loss_type
        self.distance = nn.MSELoss()
        betas = cosine_beta_schedule(time_steps)
        alphas = 1- betas

        alphas_cumprod = torch.cumprod(alphas,axis = 0)

        self.register_buffer('alphs_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))


    @torch.no_grad()
    def sample(self, batch_size=16, pop = None, t=None):

        self.denoise_fn.eval()
        if t == None:
            t = self.num_timesteps
        xt = pop

        while(t):
            step = torch.full((batch_size,), t-1, dtype=torch.long).cuda()
            x1_bar = self.denoise_fn(pop, step)
            x2_bar = self.get_x2_bar_from_xt(x1_bar, pop, step)

            xt_bar = x1_bar
            if t != 0:
                xt_bar = self.q_sample(xt_bar, x2_bar, step)

            xt_sub1_bar = x1_bar
            if t-1 != 0:
                step2 = torch.full((batch_size,), t-2, dtype=torch.long).cuda()
                xt_sub1_bar = self.q_sample(xt_sub1_bar,x2_bar, step2)

            x = pop - xt_bar + xt_sub1_bar
            pop = x
            t = t-1
        self.denoise_fn.train()
        return pop



    def get_x2_bar_from_xt(self, x1_bar, xt, t):
        return (
                (xt - extract(self.sqrt_alphas_cumprod, t, x1_bar.shape) * x1_bar) /
                extract(self.sqrt_one_minus_alphas_cumprod, t, x1_bar.shape)
        )


    def q_sample(self, x_start, x_end, t):
        return(
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_end
        )

    def p_losses(self, x_start, x_end, t):
        b, dim = x_start.shape

        x_mix = self.q_sample(x_start, x_end, t)
        x_recon = self.denoise_fn(x_mix,t)

        if self.loss_type == 'l1':
            loss = (x_start-x_recon).abs().mean()
        elif self.loss_type == 'l2':
            loss = self.distance(x_start, x_recon)

        return loss

    def forward(self, x1, x2, *args, **kwargs):
        b, dim, device = *x1.shape, x1.device

        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        return self.p_losses(x1, x2, t, *args, **kwargs)

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        pop_dim = 128,
        train_batch_size = 32,
        train_lr = 2e-5,
        weight_decay = 1e-4,
        max_epochs = 100,
        fp16 = False,
        pop1_pop2_dataloader = None,
        device = None,
    ):
        super().__init__()
        self.model = diffusion_model
        self.pop_dim  = pop_dim
        self.train_batch_size = train_batch_size
        self.pop1_pop2_dataloader = pop1_pop2_dataloader
        self.opt = Adam(self.model.parameters(), lr=train_lr,weight_decay=weight_decay)
        self.epoch = 0
        self.max_epoch = max_epochs
        self.weight_decay = weight_decay
        self.fp16 = fp16
        self.device = device

    def train(self):
        train_loss = []

        for epoch in range(self.max_epoch):
            running_loss = 0
            for pop1, pop2 in self.pop1_pop2_dataloader:
                pop1 = pop1.to(self.device).float()
                pop2 = pop2.to(self.device).float()
                loss = self.model(pop1,pop2)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                running_loss += loss.item()

            loss = running_loss / len(self.pop1_pop2_dataloader)
            train_loss.append(loss)
        #    print('Epoch [{}/{}], Loss: {:.3f}'.format(epoch + 1, self.max_epoch, loss))
        """        plt.figure()
        plt.plot(train_loss)
        plt.title('Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.savefig('./DDPM_MTO_loss.png')"""
    def test_from_otherpop(self,pop2):
        pop2 = torch.from_numpy(pop2)
        pop2 = pop2.to(self.device).float()
        transfer_pop = self.model.sample(batch_size=pop2.shape[0],pop = pop2)
        transfer_pop = transfer_pop.cpu().data.numpy()
        return transfer_pop