import os
import sys
import json
import time
import torch
from torch import nn
from torch.distributions import Normal
from torch.optim.lr_scheduler import ExponentialLR
from torch import optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch.cuda.amp import GradScaler

p = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
if p not in sys.path:
    sys.path.append(p)
from loss import loss_function_pianoTree
from utils import scheduled_sampling, epoch_time, AverageMeter, pitch_dur_accuracy, setup_seed, mean_parallel_loss,\
                chord_accuracy, kl_anneal_sigmoid
from poly.ptvae import ChordEncoder, ChordDecoder, TextureEncoder, PtvaeDecoder
from poly.model import DisentangleVAE
from data_loader import PolyphonicDataset
from BalancedDataParallel import BalancedDataParallel

class MinExponentialLR(ExponentialLR):
    def __init__(self, optimizer, gamma, minimum, last_epoch=-1):
        self.min = minimum
        super(MinExponentialLR, self).__init__(optimizer, gamma, last_epoch=-1)

    def get_lr(self):
        return [
            max(base_lr * self.gamma ** self.last_epoch, self.min)
            for base_lr in self.base_lrs
        ]

###############################################################################
# Load config
###############################################################################
config_fn = './poly/model_config.json'
with open(config_fn) as f:
    args = json.load(f)

dataset_path = args['data']

###############################################################################
# Initialize project
###############################################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if args['balanced_data']:
    print('gpus_bsz:', args['gpus_bsz'])
    args['batch_size'] = sum(args['gpus_bsz'])
else:
    assert args['batch_size']%torch.cuda.device_count()==0

setup_seed(args['random_seed'])

PARALLEL = True
PITCH_PAD = 130
MAX_PITCH = 127
MIN_PITCH = 0
DUR_PAD=2

MODEL_PATH = args['model_path']
LOG_PATH = args['log_path']
START_EPOCH = args['start_epoch']
DECAY = args['decay']
N_EPOCH = args['n_epochs']
TFR1 = args['tf_rate1']
TFR2 = args['tf_rate2']
TFR3 = args['tf_rate3']
WEIGHTS = args['weights']
BETA = args['beta']
CLIP = args['clip']

os.makedirs(MODEL_PATH, exist_ok=True)

writer = SummaryWriter(LOG_PATH)
print('Project initialized.', flush=True)

###############################################################################
# load data
###############################################################################
train_path = os.path.join(dataset_path, 'train_2bar.npy')
val_path = os.path.join(dataset_path, 'val_2bar.npy')

pitch_range = MAX_PITCH-MIN_PITCH+1+2
train_dataset = PolyphonicDataset(train_path, pitch_range = pitch_range, pitch_pad=PITCH_PAD, 
        shift_low=args['down_aug'], shift_high=args['up_aug'], contain_chord=True)
val_dataset = PolyphonicDataset(val_path, pitch_range = pitch_range, pitch_pad=PITCH_PAD, 
        shift_low=0, shift_high=0, contain_chord=True)

train_loader = DataLoader(train_dataset, args['batch_size'], shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, args['batch_size'], shuffle=False, drop_last=False)

print(len(train_dataset), len(val_dataset), flush=True)
print('Dataset loaded!', flush=True)

###############################################################################
# model parameter
###############################################################################

chd_encoder = ChordEncoder(input_dim=args['chordEncoder_in_dim'], hidden_dim=args['chordEncoder_hidden_dim'], z_dim=args['chordEncoder_z_dim'])
tex_encoder = TextureEncoder(emb_size=args['texEncoder_cnn_dim'], hidden_dim=args['texEncoder_gru_hidden_dim'], z_dim=args['texEncoder_z_dim'],
                             num_channel=args['texEncoder_cnn_channel'])

chd_decoder = ChordDecoder(z_dim=args['chordEncoder_z_dim'], z_input_dim=args['chordDecoder_inpuy_dim'], hidden_dim=args['chordDecoder_hidden_dim'])
pt_decoder = PtvaeDecoder(note_embedding=None, note_emb_size=args['ptdec_note_emb_size'], z_size=args['chordEncoder_z_dim']+args['texEncoder_z_dim'],
                            dec_emb_hid_size=args['ptdec_emb_hid_size'], dec_time_hid_size=args['ptdec_time_hid_size'], dec_notes_hid_size=args['ptdec_notes_hid_size'],
                            dec_z_in_size=args['ptdec_z_in_size'], dec_dur_hid_size=args['ptdec_dur_hid_size'])

model = DisentangleVAE('disvae-nozoth', device, chd_encoder, tex_encoder, pt_decoder, chd_decoder, amp=args['amp'])

print(f'The model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters', flush=True)

if START_EPOCH>0:
    model_path = os.path.join(MODEL_PATH, os.listdir(MODEL_PATH)[0])
    checkpoint = torch.load(model_path)
    assert START_EPOCH == checkpoint['epoch']+1
    model.load_state_dict({k.replace('module.',''):v for k,v in checkpoint['state_dict'].items()})

    print('loaded model from:'+model_path, flush=True)
else:
    print('fresh start, no model loaded', flush=True)

if PARALLEL:
    if args['balanced_data']:
        model = BalancedDataParallel(args['gpus_bsz'], model)
    else:
        model = torch.nn.DataParallel(model)
    model = model.to(device)
else:
    model = model.to(device)
print('Model loaded!', flush=True)


###############################################################################
# Optimizer and Criterion
###############################################################################
optimizer = optim.Adam(model.parameters(), lr=args['lr'])

if DECAY:
    scheduler = MinExponentialLR(optimizer, gamma=0.9999, minimum=1e-5)

if args['amp']:
    scaler = GradScaler()
###############################################################################
# Main
###############################################################################
def train(model, train_loader, optimizer, epoch):
    model.train()
    
    num_batch = len(train_loader)

    batch_start_time = time.time()

    for i, (piano_grid, pr_matrix, chords_multihot, _) in enumerate(train_loader):
        # piano_grid (B, 32, max_note_count, 6) np.int64
        # pr_matrix (B, 32, 128), duration of each note of each step. duration is from 0~32. np.float32
        # chords_multihot (B, 8, 36) np.float32
        if torch.cuda.is_available():
            piano_grid = piano_grid.cuda()
            pr_matrix = pr_matrix.cuda()
            chords_multihot = chords_multihot.cuda()
        
        optimizer.zero_grad()
        tfr1 = scheduled_sampling(((epoch + i / num_batch) / N_EPOCH),
                                  TFR1[0], TFR1[1])
        tfr2 = scheduled_sampling(((epoch + i / num_batch) / N_EPOCH),
                                  TFR2[0], TFR2[1])
        tfr3 = scheduled_sampling(((epoch + i / num_batch) / N_EPOCH),
                                  TFR3[0], TFR3[1])
        
        if args['beta_mid_epoch']>0:
            beta = BETA*kl_anneal_sigmoid(step=epoch * num_batch + i, mid_step=args['beta_mid_epoch']*num_batch, start_beta = 1e-5)
        else:
            beta = BETA

        outputs = model('train', piano_grid, chords_multihot, pr_matrix, tfr1=tfr1, tfr2=tfr2, tfr3=tfr3, beta=beta, weights=WEIGHTS, inference=False)
        loss, recon_loss, _, _, kl_loss, _, _, chord_recon_loss, _, _, _, \
            pitch_outs, dur_outs, _, _, _, _, recon_root, recon_chroma, recon_bass = outputs
        # pitch_outs (B, num_step, max_simu_note-1, pitch_range); dur_outs (B, num_step, max_simu_note-1, dur_width, 2)
        # recon_root (B, 8, 12); recon_chroma (B, 8, 12, 2); recon_bass (B, 8, 12)
        
        if args['balanced_data']:
            gpus_bsz = model.chunk_sizes
        else:
            gpus_bsz = [1]*torch.cuda.device_count()
        loss, recon_loss, kl_loss, chord_recon_loss=mean_parallel_loss((loss, recon_loss, kl_loss, chord_recon_loss), gpus_bsz)
        
        if args['amp']:
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            optimizer.step()

        if DECAY:
            scheduler.step()

        batch_mins, batch_secs = epoch_time(time.time() - batch_start_time)
        batch_start_time = time.time()

        if i % (num_batch//100) == 0:
            recon_pitch_acc, recon_dur_acc, _, _ = pitch_dur_accuracy(piano_grid, pitch_outs, dur_outs, pitch_pad=PITCH_PAD, dur_pad=DUR_PAD)
            root_acc, bass_acc, chroma_acc, _, _, _ = chord_accuracy(chords_multihot, recon_root=recon_root, recon_chroma=recon_chroma, recon_bass=recon_bass)

            print('Epoch: {:d}/{:d}, Time: {:d}m {:d}s, step: {:d}/{:d}, pitch_acc: {:.2f}, dur_acc: {:.2f}, root_acc: {:.2f}, bass_acc: {:.2f}, chroma_acc: {:.2f}, batch loss: {:.5f}, CE_loss: {:.5f}, KL_loss: {:.5f}, Chord_recon_loss: {:.5f}'.format(
                                epoch, N_EPOCH, batch_mins, batch_secs, i, num_batch, recon_pitch_acc*100, recon_dur_acc*100, root_acc*100, bass_acc*100, chroma_acc*100,
                                loss.item(), recon_loss.item(), BETA * kl_loss.item(), chord_recon_loss.item()), flush=True)

            writer.add_scalar('train_loss/loss', loss.item(), epoch * num_batch + i)
            writer.add_scalar('train_loss/CE_loss', recon_loss.item(), epoch * num_batch + i)
            writer.add_scalar('train_loss/kl_loss', BETA * kl_loss.item(), epoch * num_batch + i)
            writer.add_scalar('train_loss/Chord_recon_loss', chord_recon_loss.item(), epoch * num_batch + i)

            writer.add_scalar('train_acc/pitch_acc', recon_pitch_acc*100, epoch * num_batch + i)
            writer.add_scalar('train_acc/dur_acc', recon_dur_acc*100, epoch * num_batch + i)
            writer.add_scalar('train_acc/root_acc', root_acc*100, epoch * num_batch + i)
            writer.add_scalar('train_acc/bass_acc', bass_acc*100, epoch * num_batch + i)
            writer.add_scalar('train_acc/chroma_acc', chroma_acc*100, epoch * num_batch + i)

            writer.add_scalar('train_others/lr', optimizer.param_groups[0]['lr'], epoch * num_batch + i)
            writer.add_scalar('train_others/beta', beta, epoch * num_batch + i)
    
    return model, optimizer


def evaluate(model, val_loader, epoch):
    model.eval()

    loss_avg = AverageMeter()
    CE_loss_avg = AverageMeter()
    kl_loss_avg = AverageMeter()
    Chord_recon_loss_avg = AverageMeter()

    pitch_acc_avg = AverageMeter()
    dur_acc_avg = AverageMeter()
    root_acc_avg = AverageMeter()
    bass_acc_avg = AverageMeter()
    chroma_acc_avg = AverageMeter()

    for i, (piano_grid, pr_matrix, chords_multihot, _) in enumerate(val_loader):
        # piano_grid (B, 32, max_note_count, 6) np.int64
        # pr_matrix (B, 32, 128), duration of each note of each step. duration is from 0~32. np.float32
        # chords_multihot (B, 8, 36) np.float32
        if torch.cuda.is_available():
            piano_grid = piano_grid.cuda()
            pr_matrix = pr_matrix.cuda()
            chords_multihot = chords_multihot.cuda()

        outputs = model('train', piano_grid, chords_multihot, pr_matrix, tfr1=0, tfr2=0, tfr3=0, beta = BETA, weights = WEIGHTS, inference=True)
        loss, recon_loss, _, _, kl_loss, _, _, chord_recon_loss, _, _, _, \
            pitch_outs, dur_outs, _, _, _, _, recon_root, recon_chroma, recon_bass = outputs
        # pitch_outs (B, num_step, max_simu_note-1, pitch_range); dur_outs (B, num_step, max_simu_note-1, dur_width, 2)
        # recon_root (B, 8, 12); recon_chroma (B, 8, 12, 2); recon_bass (B, 8, 12)

        if args['balanced_data']:
            gpus_bsz = model.chunk_sizes
        else:
            gpus_bsz = [1]*torch.cuda.device_count()
        loss, recon_loss, kl_loss, chord_recon_loss=mean_parallel_loss((loss, recon_loss, kl_loss, chord_recon_loss), gpus_bsz)

        recon_pitch_acc, recon_dur_acc, _, _ = pitch_dur_accuracy(piano_grid, pitch_outs, dur_outs, pitch_pad=PITCH_PAD, dur_pad=DUR_PAD)
        root_acc, bass_acc, chroma_acc, _, _, _ = chord_accuracy(chords_multihot, recon_root=recon_root, recon_chroma=recon_chroma, recon_bass=recon_bass)

        batch_num = piano_grid.shape[0]
        loss_avg.update(loss.item()*batch_num, batch_num)
        CE_loss_avg.update(recon_loss.item()*batch_num, batch_num)
        kl_loss_avg.update(kl_loss.item()*batch_num, batch_num)
        Chord_recon_loss_avg.update(chord_recon_loss.item()*batch_num, batch_num)

        pitch_acc_avg.update(recon_pitch_acc*batch_num, batch_num)
        dur_acc_avg.update(recon_dur_acc*batch_num, batch_num)
        root_acc_avg.update(root_acc*batch_num, batch_num)
        bass_acc_avg.update(bass_acc*batch_num, batch_num)
        chroma_acc_avg.update(chroma_acc*batch_num, batch_num)

    print('Test Epoch: {:d}/{:d}, pitch_acc: {:.2f}, dur_acc: {:.2f}, root_acc: {:.2f}, bass_acc: {:.2f}, chroma_acc: {:.2f}, batch loss: {:.5f}, CE_loss: {:.5f}, KL_loss: {:.5f}, Chord_recon_loss: {:.5f}'.format(
                                epoch, N_EPOCH, pitch_acc_avg.avg*100, dur_acc_avg.avg*100, root_acc_avg.avg*100, bass_acc_avg.avg*100, chroma_acc_avg.avg*100,
                                loss_avg.avg, CE_loss_avg.avg, BETA * kl_loss_avg.avg, Chord_recon_loss_avg.avg), flush=True)

    writer.add_scalar('val_loss/loss', loss_avg.avg, epoch)
    writer.add_scalar('val_loss/CE_loss', CE_loss_avg.avg, epoch)
    writer.add_scalar('val_loss/kl_loss', kl_loss_avg.avg, epoch)
    writer.add_scalar('val_loss/Chord_recon_loss', Chord_recon_loss_avg.avg, epoch)

    writer.add_scalar('val_acc/pitch_acc', pitch_acc_avg.avg*100, epoch)
    writer.add_scalar('val_acc/dur_acc', dur_acc_avg.avg*100, epoch)
    writer.add_scalar('val_acc/root_acc', root_acc_avg.avg*100, epoch)
    writer.add_scalar('val_acc/bass_acc', bass_acc_avg.avg*100, epoch)
    writer.add_scalar('val_acc/chroma_acc', chroma_acc_avg.avg*100, epoch)

with torch.no_grad():
        evaluate(model, val_loader, -1)
for epoch in range(START_EPOCH, N_EPOCH):
    print(f'Start Epoch: {epoch + 1:02}', flush=True)

    start_time = time.time()

    model, optimizer = train(model, train_loader, optimizer, epoch)
    with torch.no_grad():
        evaluate(model, val_loader, epoch)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(end_time - start_time)
    
    checkpoint = {'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),}
    torch.save(checkpoint, os.path.join(MODEL_PATH,
                                                'pgrid-epoch-model.pt'))

    print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s', flush=True)


if __name__ == '__main__':
    pass
