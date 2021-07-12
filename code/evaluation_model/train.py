import os
import sys
import json
import socket
import getpass
import glog as log
import time
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR
from tensorboardX import SummaryWriter
import torch.nn.functional as F

p = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
if p not in sys.path:
    sys.path.append(p)
from utils import set_log_file, setup_seed, critic_accuracy, AverageMeter
from data_loader import Eval_dataset
from evaluation_model.model import PianoTree_evaluation


class MinExponentialLR(ExponentialLR):
    def __init__(self, optimizer, gamma, minimum, last_epoch=-1):
        self.min = minimum
        super(MinExponentialLR, self).__init__(optimizer, gamma, last_epoch=-1)

    def get_lr(self):
        return [
            max(base_lr * self.gamma ** self.last_epoch, self.min)
            for base_lr in self.base_lrs
        ]
    
def train_epoch(args, model, train_loader, optimizer, scheduler, step, writer):
    model.train()

    batch_start_time = time.time()
    for batch_idx, (batch) in enumerate(train_loader):
        match_melody_one_hot, match_accompany_piano_grid, nmatch_melody_one_hot, nmatch_accompany_piano_grid = batch
        match_melody_one_hot = match_melody_one_hot.cuda()
        match_accompany_piano_grid = match_accompany_piano_grid.cuda()
        nmatch_melody_one_hot = nmatch_melody_one_hot.cuda()
        nmatch_accompany_piano_grid = nmatch_accompany_piano_grid.cuda()
        data_loading_time = time.time() - batch_start_time

        optimizer.zero_grad()
        melody_inputs = torch.cat([match_melody_one_hot, nmatch_melody_one_hot], dim=0)
        accompany_inputs = torch.cat([match_accompany_piano_grid, nmatch_accompany_piano_grid], dim=0)
        outputs = model(x_accompany=accompany_inputs, x_melody=melody_inputs)

        labels = torch.zeros([outputs.shape[0], 1], device="cuda")
        labels[:match_melody_one_hot.shape[0], :]=1

        loss_all = F.binary_cross_entropy(outputs, labels, reduction='none')
        loss_match = loss_all[:match_melody_one_hot.shape[0]].mean()
        loss_nmatch = loss_all[match_melody_one_hot.shape[0]:].mean()
        loss = (loss_match*args['loss_match_k'] + loss_nmatch)*0.5
        loss.backward()
        optimizer.step()

        if args['decay']:
            scheduler.step()

        batch_secs = time.time() - batch_start_time
        batch_start_time = time.time()

        if (step % args['print_every']==0) or (batch_idx==0) or (batch_idx==len(train_loader)-1):
            E_accuracy, E_batch_equal = critic_accuracy(D_output=outputs, label=labels)
            E_match_accuracy = E_batch_equal[:match_melody_one_hot.shape[0]].mean()
            E_nmatch_accuray = E_batch_equal[match_melody_one_hot.shape[0]:].mean()

            writer.add_scalar('train/loss', loss.item(), step)
            writer.add_scalar('train/loss_match', loss_match.item(), step)
            writer.add_scalar('train/loss_nmatch', loss_nmatch.item(), step)
            writer.add_scalar('train/E_accuracy', E_accuracy*100, step)
            writer.add_scalar('train/E_match_accuracy', E_match_accuracy*100, step)
            writer.add_scalar('train/E_nmatch_accuray', E_nmatch_accuray*100, step)
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], step)

            log.info("Train: {:d}/{:d}, epoch i: {:d}/{:d}, Time: {:.2f}s, DataTime: {:.2f}s, acc: {:.2f}, m_acc: {:.2f}, nm_acc: {:.2f}, loss: {:.5f}, loss_match: {:.5f}, loss_nmatch: {:.5f}".format(
                step, args['iteration_num'], batch_idx, len(train_loader), batch_secs, data_loading_time, E_accuracy*100, E_match_accuracy*100, E_nmatch_accuray*100,
                 loss.item(), loss_match.item(), loss_nmatch.item()))
        
        step +=1 
    return step

def eval_epoch(args, model, val_loader, step, writer, epoch):
    model.eval()

    loss_avg = AverageMeter()
    loss_match_avg = AverageMeter()
    loss_nmatch_avg = AverageMeter()
    E_accuracy_avg = AverageMeter()
    E_match_accuracy_avg = AverageMeter()
    E_nmatch_accuracy_avg = AverageMeter()

    start_time = time.time()
    for batch_idx, (batch) in enumerate(val_loader):
        match_melody_one_hot, match_accompany_piano_grid, nmatch_melody_one_hot, nmatch_accompany_piano_grid = batch
        match_melody_one_hot = match_melody_one_hot.cuda()
        match_accompany_piano_grid = match_accompany_piano_grid.cuda()
        nmatch_melody_one_hot = nmatch_melody_one_hot.cuda()
        nmatch_accompany_piano_grid = nmatch_accompany_piano_grid.cuda()

        melody_inputs = torch.cat([match_melody_one_hot, nmatch_melody_one_hot], dim=0)
        accompany_inputs = torch.cat([match_accompany_piano_grid, nmatch_accompany_piano_grid], dim=0)
        outputs = model(x_accompany=accompany_inputs, x_melody=melody_inputs)

        labels = torch.zeros([outputs.shape[0], 1], device="cuda")
        labels[:match_melody_one_hot.shape[0], :]=1

        loss_all = F.binary_cross_entropy(outputs, labels, reduction='none')
        loss = loss_all.sum()
        
        E_accuracy, E_batch_equal = critic_accuracy(D_output=outputs, label=labels)
        E_match_accuracy = E_batch_equal[:match_melody_one_hot.shape[0]].sum()
        E_nmatch_accuray = E_batch_equal[match_melody_one_hot.shape[0]:].sum()
        loss_match = loss_all[:match_melody_one_hot.shape[0]].sum()
        loss_nmatch = loss_all[match_melody_one_hot.shape[0]:].sum()

        loss_avg.update(loss.item(), outputs.shape[0])
        loss_match_avg.update(loss_match.item(), match_melody_one_hot.shape[0])
        loss_nmatch_avg.update(loss_nmatch.item(), nmatch_melody_one_hot.shape[0])
        E_accuracy_avg.update(E_accuracy*100*outputs.shape[0], outputs.shape[0])
        E_match_accuracy_avg.update(E_match_accuracy*100, match_melody_one_hot.shape[0])
        E_nmatch_accuracy_avg.update(E_nmatch_accuray*100, nmatch_melody_one_hot.shape[0])

    writer.add_scalar('val/loss', loss_avg.avg, step)
    writer.add_scalar('val/loss_match', loss_match_avg.avg, step)
    writer.add_scalar('val/loss_nmatch', loss_nmatch_avg.avg, step)
    writer.add_scalar('val/E_accuracy', E_accuracy_avg.avg, step)
    writer.add_scalar('val/E_match_accuracy', E_match_accuracy_avg.avg, step)
    writer.add_scalar('val/E_nmatch_accuray', E_nmatch_accuracy_avg.avg, step)

    log.info("Val: {:d}/{:d}, epoch: {:d}, Time: {:.2f}s, acc: {:.2f}, m_acc: {:.2f}, nm_acc: {:.2f}, loss: {:.5f}, loss_match: {:.5f}, loss_nmatch: {:.5f}".format(
                step, args['iteration_num'], epoch, time.time()-start_time, E_accuracy_avg.avg, E_match_accuracy_avg.avg, E_nmatch_accuracy_avg.avg,
                 loss_avg.avg, loss_match_avg.avg, loss_nmatch_avg.avg))

    return E_accuracy_avg.avg

def main(args):
    setup_seed(args['random_seed'])

    PITCH_PAD = 130
    MAX_PITCH = 127
    MIN_PITCH = 0
    pitch_range = MAX_PITCH-MIN_PITCH+1+2

    args['note_emb_size'] *= args["_model_size_k"]
    args['enc_notes_hid_size'] *= args["_model_size_k"]
    args['enc_time_hid_size'] *= args["_model_size_k"]
    args['melody_hid_size'] *= args["_model_size_k"]
    args['before_cat_dim'] *= args["_model_size_k"]

    model = PianoTree_evaluation(max_simu_note=args['max_simu_note'], max_pitch=MAX_PITCH, min_pitch=MIN_PITCH, note_emb_size=args['note_emb_size'], melody_hidden_dims=args['melody_hid_size'],
            enc_notes_hid_size=args['enc_notes_hid_size'], enc_time_hid_size=args['enc_time_hid_size'], before_cat_dim=args['before_cat_dim'], mid_sizes=args['mid_sizes'], drop_out_p=args['dropout_rate'])
    if args['parallel']:
        model = torch.nn.DataParallel(model)
    model.cuda()

    log.info("model loaded")
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
    if args['decay']:
        scheduler = MinExponentialLR(optimizer, gamma=0.9999, minimum=1e-5)
    else:
        scheduler = None

    writer = SummaryWriter('run_evaluation')
    MODEL_PATH = 'params_evaluation'
    os.makedirs(MODEL_PATH, exist_ok=True)

    dummy_input = (torch.ones([1, model.num_step, model.max_simu_note, model.dur_width+1], device="cuda").long(), torch.ones([1, model.num_step, 130], device="cuda"))
    writer.add_graph(model, dummy_input)

    train_path = os.path.join(args['data_path'], 'train_2bar.npy')
    val_path = os.path.join(args['data_path'], 'val_2bar.npy')
    train_dataset = Eval_dataset(filepath=train_path, pitch_range=pitch_range, pitch_pad=PITCH_PAD, shift_low=args['down_aug'], shift_high=args['up_aug'],
                                max_simu_note=args['max_simu_note'], nmatch_fix=args['nmatch_fix'])
    val_dataset = Eval_dataset(filepath=val_path, pitch_range=pitch_range, pitch_pad=PITCH_PAD, shift_low=0, shift_high=0,
                                max_simu_note=args['max_simu_note'], nmatch_fix=args['nmatch_fix'])
    train_loader = DataLoader(train_dataset, args['batch_size'], shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, args['batch_size'], shuffle=False, drop_last=False)

    step = 0
    epoch_num = args['iteration_num']//len(train_loader) + 1
    best_step=0
    with torch.no_grad():
        best_mean_acc = eval_epoch(args, model, val_loader, step, writer, -1)
    for epoch in range(epoch_num):
        step = train_epoch(args, model, train_loader, optimizer, scheduler, step, writer)
        with torch.no_grad():
            mean_acc = eval_epoch(args, model, val_loader, step, writer, epoch)

        checkpoint = {'step': step,
                    'mean_acc':mean_acc,
                    'state_dict': model.state_dict()}
        
        if mean_acc > best_mean_acc:
            name = 'best-model.pt'
            best_mean_acc = mean_acc
            best_step = step
        else:
            name = 'epoch-model.pt'
        torch.save(checkpoint, os.path.join(MODEL_PATH, name))
    
    log.info("best step: %d; mean_acc: %.2f"%(best_step, best_mean_acc))

if __name__ == '__main__':
    config_fn = './evaluation_model/model_config.json'
    with open(config_fn) as f:
        args = json.load(f)

    set_log_file('train_log_evaluation.txt', file_only=args['ssh'])

    log.info('Host: {}, user: {}, CUDA_VISIBLE_DEVICES: {}, cwd: {}'.format(
        socket.gethostname(), getpass.getuser(), os.environ.get('CUDA_VISIBLE_DEVICES', ''), os.getcwd()))
    
    main(args)