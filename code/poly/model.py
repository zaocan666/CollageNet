import torch
import torch
from torch import nn
from torch.distributions import Normal
from torch.distributions import Normal, kl_divergence
import numpy as np
from torch.cuda.amp import autocast as autocast

def get_zs_from_dists(dists, sample=False):
    return [dist.rsample() if sample else dist.mean for dist in dists]

def standard_normal(shape):
    N = Normal(torch.zeros(shape), torch.ones(shape))
    if torch.cuda.is_available():
        N.loc = N.loc.cuda()
        N.scale = N.scale.cuda()
    return N

def kl_with_normal(dist):
    shape = dist.mean.size(-1)
    normal = standard_normal(shape)
    kl = kl_divergence(dist, normal).mean()
    return kl

class PytorchModel(nn.Module):

    def __init__(self, name, device, amp):
        self.name = name
        super(PytorchModel, self).__init__()
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available()
                                  else 'cpu')
        self.device = device
        self.amp = amp

    def run(self, *input):
        """A general way to run the model.
        Usually tensor input -> tensor output"""
        raise NotImplementedError

    def loss(self, *input, **kwargs):
        """Call it during training. The output is loss and possibly others to
        display on tensorboard."""
        raise NotImplementedError

    def inference(self, *input):
        """Call it during inference.
        The output is usually numpy after argmax."""
        raise NotImplementedError

    def loss_function(self, *input):
        raise NotImplementedError
    
    def forward_(self, mode, *input, **kwargs):
        if mode in ["run", 0]:
            return self.run(*input, **kwargs)
        elif mode in ['loss', 'train', 1]:
            return self.loss(*input, **kwargs)
        elif mode in ['inference', 'eval', 'val', 2]:
            return self.inference(*input, **kwargs)
        else:
            raise NotImplementedError

    def forward(self, mode, *input, **kwargs):
        if self.amp:
            with autocast():
                return self.forward_(mode, *input, **kwargs)
        else:
            return self.forward_(mode, *input, **kwargs)

    def load_model(self, model_path, map_location=None):
        if map_location is None:
            map_location = self.device
        dic = torch.load(model_path, map_location=map_location)
        for name in list(dic.keys()):
            dic[name.replace('module.', '')] = dic.pop(name)
        self.load_state_dict(dic)
        self.to(self.device)

    @staticmethod
    def init_model(*inputs):
        raise NotImplementedError

class DisentangleVAE(PytorchModel):

    def __init__(self, name, device, chd_encoder, rhy_encoder, decoder,
                 chd_decoder, amp):
        super(DisentangleVAE, self).__init__(name, device, amp)
        self.chd_encoder = chd_encoder
        self.rhy_encoder = rhy_encoder
        self.decoder = decoder
        self.num_step = self.decoder.num_step
        self.chd_decoder = chd_decoder

    def confuse_prmat(self, pr_mat):
        non_zero_ent = torch.nonzero(pr_mat.long())
        eps = torch.randint(0, 2, (non_zero_ent.size(0),))
        eps = ((2 * eps) - 1).long()
        confuse_ent = torch.clamp(non_zero_ent[:, 2] + eps, min=0, max=127)
        pr_mat[non_zero_ent[:, 0], non_zero_ent[:, 1], confuse_ent] = \
            pr_mat[non_zero_ent[:, 0], non_zero_ent[:, 1], non_zero_ent[:, 2]]
        return pr_mat

    def get_chroma(self, pr_mat):
        bs = pr_mat.size(0)
        pad = torch.zeros(bs, 32, 4).to(self.device)
        pr_mat = torch.cat([pr_mat, pad], dim=-1)
        c = pr_mat.view(bs, 32, -1, 12).contiguous()
        c = c.sum(dim=-2)  # (bs, 32, 12)
        c = c.view(bs, 8, 4, 12)
        c = c.sum(dim=-2).float()
        c = torch.log(c + 1)
        return c.to(self.device)

    def run(self, x, c, pr_mat, tfr1, tfr2, tfr3, inference=False, confuse=True, sample=True):
        # x: p_grids (32, max_note_count, 6), In the last dim, the 0th column is for pitch, 1: 6 is for duration in binary repr.
        # c: chord (8, 36)
        # pr_mat: (32, 128), duration matrix

        embedded_x, lengths = self.decoder.emb_x(x)
        dist_chd = self.chd_encoder(c)
        dist_rhy = self.rhy_encoder(pr_mat)
        z_chd, z_rhy = get_zs_from_dists([dist_chd, dist_rhy], sample=sample)
        dec_z = torch.cat([z_chd, z_rhy], dim=-1)
        if not inference:
            pitch_outs, dur_outs = self.decoder(dec_z, inference, embedded_x,
                                                lengths, tfr1, tfr2)
        else:
            pitch_outs, dur_outs = self.decoder(dec_z, inference, None,
                                                None, tfr1, tfr2)
        recon_root, recon_chroma, recon_bass = self.chd_decoder(z_chd, inference,
                                                                tfr3, c)
        return pitch_outs, dur_outs, dist_chd, dist_rhy, recon_root, \
            recon_chroma, recon_bass

    def loss_function(self, x, c, recon_pitch, recon_dur, dist_chd,
                      dist_rhy, recon_root, recon_chroma, recon_bass,
                      beta, weights, weighted_dur=False):
        # x: p_grids (32, max_note_count, 6), In the last dim, the 0th column is for pitch, 1: 6 is for duration in binary repr.
        # c: chord (8, 36)
        # recon_pitch (B, num_step, max_simu_note-1, pitch_range)
        # recon_dur (B, num_step, max_simu_note-1, dur_width, 2)
        recon_loss, pl, dl = self.decoder.recon_loss(x, recon_pitch, recon_dur,
                                                     weights, weighted_dur)
        kl_loss, kl_chd, kl_rhy = self.kl_loss(dist_chd, dist_rhy)
        chord_loss, root, chroma, bass = self.chord_loss(c, recon_root,
                                                         recon_chroma,
                                                         recon_bass)
        loss = recon_loss + beta * kl_loss + chord_loss
        return loss, recon_loss, pl, dl, kl_loss, kl_chd, kl_rhy, chord_loss, \
               root, chroma, bass

    def chord_loss(self, c, recon_root, recon_chroma, recon_bass):
        # recon_root (bs, 8, 12)
        # recon_chroma (bs, 8, 12, 2)
        # recon_bass (bs, 8, 12)

        loss_fun = nn.CrossEntropyLoss()
        root = c[:, :, 0: 12].max(-1)[-1].view(-1).contiguous()
        bass = c[:, :, 12:24].max(-1)[-1].view(-1).contiguous()
        chroma = c[:, :, 24: ].long().view(-1).contiguous()

        recon_root = recon_root.view(-1, 12).contiguous()
        recon_chroma = recon_chroma.view(-1, 2).contiguous()
        recon_bass = recon_bass.view(-1, 12).contiguous()
        root_loss = loss_fun(recon_root, root)
        chroma_loss = loss_fun(recon_chroma, chroma)
        bass_loss = loss_fun(recon_bass, bass)
        chord_loss = root_loss + chroma_loss + bass_loss
        return chord_loss, root_loss, chroma_loss, bass_loss

    def kl_loss(self, *dists):
        # kl = kl_with_normal(dists[0])
        kl_chd = kl_with_normal(dists[0])
        kl_rhy = kl_with_normal(dists[1])
        kl_loss = kl_chd + kl_rhy
        return kl_loss, kl_chd, kl_rhy

    def loss(self, x, c, pr_mat, tfr1=0., tfr2=0., tfr3=0.,
             beta=0.1, weights=(1, 0.5), inference=False, sample=True):
        # x: p_grids (32, max_note_count, 6), In the last dim, the 0th column is for pitch, 1: 6 is for duration in binary repr.
        # c: chord (8, 36)
        # pr_mat: (32, 128), duration matrix

        outputs = self.run(x, c, pr_mat, tfr1, tfr2, tfr3, inference=inference, sample=sample)
        loss = self.loss_function(x, c, *outputs, beta, weights)
        pitch_outs, dur_outs, dist_chd, dist_rhy, recon_root, recon_chroma, recon_bass = outputs
        return loss + (pitch_outs, dur_outs, dist_chd.mean, dist_chd.stddev, dist_rhy.mean, dist_rhy.stddev, recon_root, recon_chroma, recon_bass)

    def inference_encode(self, pr_mat, c):
        self.eval()
        with torch.no_grad():
            dist_chd = self.chd_encoder(c)
            dist_rhy = self.rhy_encoder(pr_mat)
        return dist_chd, dist_rhy

    def inference_decode(self, z_chd, z_rhy):
        self.eval()
        with torch.no_grad():
            dec_z = torch.cat([z_chd, z_rhy], dim=-1)
            pitch_outs, dur_outs = self.decoder(dec_z, True, None,
                                                None, 0., 0.)
            est_x, _, _ = self.decoder.output_to_numpy(pitch_outs, dur_outs)
        return est_x

    def inference(self, pr_mat, c, sample):
        self.eval()
        with torch.no_grad():
            dist_chd = self.chd_encoder(c)
            dist_rhy = self.rhy_encoder(pr_mat)
            z_chd, z_rhy = get_zs_from_dists([dist_chd, dist_rhy], sample)
            dec_z = torch.cat([z_chd, z_rhy], dim=-1)
            pitch_outs, dur_outs = self.decoder(dec_z, True, None,
                                                None, 0., 0.)
            est_x, _, _ = self.decoder.output_to_numpy(pitch_outs, dur_outs)
        return est_x

    def swap(self, pr_mat1, pr_mat2, c1, c2, fix_rhy, fix_chd):
        pr_mat = pr_mat1 if fix_rhy else pr_mat2
        c = c1 if fix_chd else c2
        est_x = self.inference(pr_mat, c, sample=False)
        return est_x

    def posterior_sample(self, pr_mat, c, scale=None, sample_chd=True,
                         sample_txt=True):
        if scale is None and sample_chd and sample_txt:
            est_x = self.inference(pr_mat, c, sample=True)
        else:
            dist_chd, dist_rhy = self.inference_encode(pr_mat, c)
            if scale is not None:
                mean_chd = dist_chd.mean
                mean_rhy = dist_rhy.mean
                std_chd = dist_chd.scale * scale
                std_rhy = dist_rhy.scale * scale
                dist_rhy = Normal(mean_rhy, std_rhy)
                dist_chd = Normal(mean_chd, std_chd)
            z_chd, z_rhy = get_zs_from_dists([dist_chd, dist_rhy], True)
            if not sample_chd:
                z_chd = dist_chd.mean
            if not sample_txt:
                z_rhy = dist_rhy.mean
            est_x = self.inference_decode(z_chd, z_rhy)
        return est_x

    def prior_sample(self, x, c, sample_chd=False, sample_rhy=False,
                     scale=1.):
        dist_chd, dist_rhy = self.inference_encode(x, c)
        mean = torch.zeros_like(dist_rhy.mean)
        loc = torch.ones_like(dist_rhy.mean) * scale
        if sample_chd:
            dist_chd = Normal(mean, loc)
        if sample_rhy:
            dist_rhy = Normal(mean, loc)
        z_chd, z_rhy = get_zs_from_dists([dist_chd, dist_rhy], True)
        return self.inference_decode(z_chd, z_rhy)

    def gt_sample(self, x):
        out = x[:, :, 1:].numpy()
        return out

    def interp(self, pr_mat1, c1, pr_mat2, c2, interp_chd=False,
               interp_rhy=False, int_count=10):
        dist_chd1, dist_rhy1 = self.inference_encode(pr_mat1, c1)
        dist_chd2, dist_rhy2 = self.inference_encode(pr_mat2, c2)
        [z_chd1, z_rhy1, z_chd2, z_rhy2] = \
            get_zs_from_dists([dist_chd1, dist_rhy1, dist_chd2, dist_rhy2],
                              False)
        if interp_chd:
            z_chds = self.interp_z(z_chd1, z_chd2, int_count)
        else:
            z_chds = z_chd1.unsqueeze(1).repeat(1, int_count, 1)
        if interp_rhy:
            z_rhys = self.interp_z(z_rhy1, z_rhy2, int_count)
        else:
            z_rhys = z_rhy1.unsqueeze(1).repeat(1, int_count, 1)
        bs = z_chds.size(0)
        z_chds = z_chds.view(bs * int_count, -1).contiguous()
        z_rhys = z_rhys.view(bs * int_count, -1).contiguous()
        estxs = self.inference_decode(z_chds, z_rhys)
        return estxs.reshape((bs, int_count, 32, 15, -1))

    def interp_z(self, z1, z2, int_count=10):
        z1 = z1.numpy()
        z2 = z2.numpy()
        zs = torch.stack([self.interp_path(zz1, zz2, int_count)
                          for zz1, zz2 in zip(z1, z2)], dim=0)
        return zs

    def interp_path(self, z1, z2, interpolation_count=10):
        result_shape = z1.shape
        z1 = z1.reshape(-1)
        z2 = z2.reshape(-1)

        def slerp2(p0, p1, t):
            omega = np.arccos(
                np.dot(p0 / np.linalg.norm(p0), p1 / np.linalg.norm(p1)))
            so = np.sin(omega)
            return np.sin((1.0 - t) * omega)[:, None] / so * p0[
                None] + np.sin(
                t * omega)[:, None] / so * p1[None]

        percentages = np.linspace(0.0, 1.0, interpolation_count)

        normalized_z1 = z1 / np.linalg.norm(z1)
        normalized_z2 = z2 / np.linalg.norm(z2)
        dirs = slerp2(normalized_z1, normalized_z2, percentages)
        length = np.linspace(np.log(np.linalg.norm(z1)),
                             np.log(np.linalg.norm(z2)),
                             interpolation_count)
        out = (dirs * np.exp(length[:, None])).reshape(
            [interpolation_count] + list(result_shape))
        return torch.from_numpy(out).to(self.device).float()
