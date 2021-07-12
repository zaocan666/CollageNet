import json
import torch
import os
from torch import optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import sys
import time

p = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
if p not in sys.path:
    sys.path.append(p)
from utils import AverageMeter, kl_anneal_sigmoid
from EC2_VAE.model import VAE
from data_loader import Melody_dataset
from loss import loss_function, MinExponentialLR
from utils import get_rhythm_target

MELODY_DIMS = 130
CHORDS_DIMS = 12
RHYTHM_DIMS = 3


# some initialization
with open('EC2_VAE/model_config.json') as f:
    args = json.load(f)

print(args, flush=True)

os.makedirs('run_EC2', exist_ok=True)
os.makedirs('params_EC2', exist_ok=True)
save_path = 'params_EC2/{}.pt'.format(args['name'])

writer = SummaryWriter('run_EC2/{}'.format(args['name']))

condition_dims =  0
model = VAE(roll_dims = MELODY_DIMS, hidden_dims = args['hidden_dim'], rhythm_dims = RHYTHM_DIMS, 
            condition_dims = condition_dims, z1_dims = args['pitch_dim'],
            z2_dims = args['rhythm_dim'], n_step = args['time_step'], eps=args['eps'])
if args['if_parallel']:
    model = torch.nn.DataParallel(model)
    
optimizer = optim.Adam(model.parameters(), lr=args['lr'])
if args['decay'] > 0:
    scheduler = MinExponentialLR(optimizer, gamma=args['decay'], minimum=1e-5)

print('Using: ', torch.cuda.get_device_name(torch.cuda.current_device()), flush=True)
model.cuda()

step, pre_epoch = 0, 0

data_root = args["data"]
dataset = Melody_dataset(data_path = os.path.join(data_root, 'train_2bar.npy'), notes_dim = MELODY_DIMS, shift_low=args['down_aug'], shift_high=args['up_aug'])
dl = DataLoader(dataset, batch_size= args['batch_size'], shuffle=True, num_workers= 0)
dataset_test = Melody_dataset(data_path = os.path.join(data_root, 'val_2bar.npy'), notes_dim = MELODY_DIMS, shift_low=0, shift_high=0)
dl_test = DataLoader(dataset_test, batch_size= args['batch_size'], shuffle=False, num_workers= 0)

# end of initialization

def train_epoch(epoch, step, beta):
    writer.add_scalar('epoch', epoch, step)

    model.train()
    start_time = time.time()
    for melody_one_hot, melody_target, _ in dl:
        melody_one_hot = melody_one_hot.float()
        
        melody_one_hot = melody_one_hot.cuda()
        melody_target = melody_target.cuda()

        recon, recon_rhythm, dis1m, dis1s, dis2m, dis2s = model(melody_one_hot, None)

        rhythm_target = get_rhythm_target(melody_one_hot)

        loss, CE_loss, KL_loss = loss_function(
            recon,
            recon_rhythm,
            melody_target,
            rhythm_target,
            dis1s,
            dis1m,
            dis2s,
            dis2m,
            step,
            beta=beta)
            
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

        with torch.no_grad():
            recon_melody = recon.max(-1)[1]
            recon_acc = float((recon_melody==melody_target).sum())/melody_target.shape[1]/melody_target.shape[0]

        step += 1

        batch_time = time.time()-start_time
        start_time = time.time()
        if step % (len(dl)//10) == 0:
            print('Epoch: {:d}/{:d}, batch time: {:.1f} s, step: {:d} {:d}/{:d}, recon_acc: {:.4f}, batch loss: {:.5f}, CE_loss: {:.5f}, KL_loss: {:.5f}'.format(
                                epoch, args['n_epochs'], batch_time, step, step%len(dl), len(dl), recon_acc, loss.item(), CE_loss.item(), KL_loss.item()), flush=True)

            writer.add_scalar('train/batch_loss', loss.item(), step)
            writer.add_scalar('train/CE_loss', CE_loss.item(), step)
            writer.add_scalar('train/KL_loss', KL_loss.item(), step)
            writer.add_scalar('train/recon_acc', recon_acc, step)

            writer.add_scalar('train/KL_beta', beta, step)
            writer.add_scalar('train/eps', model.eps, step)
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], step)

        if args['decay'] > 0:
            scheduler.step()

        if args['beta_start_epoch']>0:
            beta = kl_anneal_sigmoid(step=step, mid_step=args['beta_start_epoch']*len(dl), start_beta = args['beta_start'])
        else:
            beta = args['beta_start']
        
        if args['eps_start_epoch']>0:
            model.eps = (1 - kl_anneal_sigmoid(step=step, mid_step=args['eps_start_epoch']*len(dl), start_beta = 1e-5))*args['eps']
        else:
            model.eps = args['eps']

    return step, loss.item(), beta

def test_epoch(epoch, step):
    loss_avg = AverageMeter()
    CE_loss_avg = AverageMeter()
    KL_loss_avg = AverageMeter()
    recon_acc_avg = AverageMeter()

    model.eval()
    for melody_one_hot, melody_target, _ in dl_test:
        melody_one_hot = melody_one_hot.float()
        melody_one_hot = melody_one_hot.cuda()
        melody_target = melody_target.cuda()
        
        recon, recon_rhythm, dis1m, dis1s, dis2m, dis2s = model(melody_one_hot, None)

        rhythm_target = get_rhythm_target(melody_one_hot)

        loss, CE_loss, KL_loss = loss_function(
            recon,
            recon_rhythm,
            melody_target,
            rhythm_target,
            dis1s,
            dis1m,
            dis2s,
            dis2m,
            step,
            beta=1)

        recon_melody = recon.max(-1)[1]
        recon_acc = float((recon_melody==melody_target).sum())/melody_target.shape[1]

        batch_num = melody_target.shape[0]
        loss_avg.update(loss.item()*batch_num, batch_num)
        CE_loss_avg.update(CE_loss.item()*batch_num, batch_num)
        KL_loss_avg.update(KL_loss.item()*batch_num, batch_num)
        recon_acc_avg.update(recon_acc, batch_num)
    
    print('Test Epoch: {:d}/{:d}, recon_acc: {:.5f}, loss: {:.5f}, CE_loss: {:.5f}, KL_loss: {:.5f}'.format(
                                    epoch, args['n_epochs'], recon_acc_avg.avg, loss_avg.avg, CE_loss_avg.avg, KL_loss_avg.avg), flush=True)

    writer.add_scalar('test/batch_loss', loss_avg.avg, step)
    writer.add_scalar('test/CE_loss', CE_loss_avg.avg, step)
    writer.add_scalar('test/KL_loss', KL_loss_avg.avg, step)
    writer.add_scalar('test/recon_acc', recon_acc_avg.avg, step)
    
    return loss_avg.avg

beta = 0
for epoch in range(args['n_epochs']):
    with torch.no_grad():
        test_loss = test_epoch(epoch, step)
    step, train_loss, beta = train_epoch(epoch, step, beta)    

    checkpoint = {'epoch': epoch,
                'state_dict': model.state_dict(),
                'train_loss': train_loss,
                'test_loss': test_loss,
                'optimizer': optimizer.state_dict(),}
    
    torch.save(checkpoint, save_path)
