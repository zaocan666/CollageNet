from torch.distributions import kl_divergence, Normal
from torch.nn import functional as F
from torch.optim.lr_scheduler import ExponentialLR
import torch

class MinExponentialLR(ExponentialLR):
    def __init__(self, optimizer, gamma, minimum, last_epoch=-1):
        self.min = minimum
        super(MinExponentialLR, self).__init__(optimizer, gamma, last_epoch=last_epoch)

    def get_lr(self):
        return [
            max(base_lr * self.gamma**self.last_epoch, self.min)
            for base_lr in self.base_lrs
        ]

def std_normal(shape):
    N = Normal(torch.zeros(shape), torch.ones(shape))
    if torch.cuda.is_available():
        N.loc = N.loc.cuda()
        N.scale = N.scale.cuda()
    return N

def loss_function(recon,
                  recon_rhythm,
                  target_tensor,
                  rhythm_target,
                  std_1,
                  mean_1,
                  std_2,
                  mean_2,
                  step,
                  beta=.1):
    
    CE1 = F.nll_loss(
        recon.view(-1, recon.size(-1)),
        target_tensor.view(-1),
        reduction='sum')
    CE2 = F.nll_loss(
        recon_rhythm.view(-1, recon_rhythm.size(-1)),
        rhythm_target,
        reduction='sum')

    KLD1 = -0.5 * torch.sum(1 + 2*torch.log(std_1) - mean_1.pow(2) - std_1.pow(2))/recon.size(1)
    KLD2 = -0.5 * torch.sum(1 + 2*torch.log(std_2) - mean_2.pow(2) - std_2.pow(2))/recon.size(1)

    CE = (CE1+CE2) / recon.size(0)
    KL = (KLD1+KLD2) / recon.size(0)
    return CE + beta * KL, CE, KL


def loss_function_pianoTree(recon_pitch, pitch, recon_dur, dur,
                  dist, pitch_criterion, dur_criterion, normal,
                  weights=(1, .5, .1)):
    pitch_loss = pitch_criterion(recon_pitch, pitch)
    recon_dur = recon_dur.view(-1, 5, 2)
    dur = dur.view(-1, 5)
    dur0 = dur_criterion(recon_dur[:, 0, :], dur[:, 0])
    dur1 = dur_criterion(recon_dur[:, 1, :], dur[:, 1])
    dur2 = dur_criterion(recon_dur[:, 2, :], dur[:, 2])
    dur3 = dur_criterion(recon_dur[:, 3, :], dur[:, 3])
    dur4 = dur_criterion(recon_dur[:, 4, :], dur[:, 4])

    w = torch.tensor([1, 0.6, 0.4, 0.3, 0.3],
                      dtype=float,
                      device=recon_dur.device)

    dur_loss = w[0] * dur0 + w[1] * dur1 + w[2] * dur2 + w[3] * dur3 + \
               w[4] * dur4
    kl_div = kl_divergence(dist, normal).mean()
    loss = weights[0] * pitch_loss + weights[1] * dur_loss + \
        weights[2] * kl_div
    return loss, pitch_loss, dur_loss, kl_div