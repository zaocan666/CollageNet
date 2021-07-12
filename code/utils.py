import math
import torch
import pretty_midi as pyd
import numpy as np
import music21
from music21 import harmony
import re
import random
import sys
import os
from torch._C import device
import torch.nn.functional as F
import glog as log
import json

p = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
if p not in sys.path:
    sys.path.append(p)
from EC2_VAE.model import VAE as EC2_VAE
from poly.ptvae import ChordEncoder, ChordDecoder, TextureEncoder, PtvaeDecoder
from poly.model import DisentangleVAE as Poly_model

hold_pitch = 128
rest_pitch = 129
notes_dim = 130

def epoch_time(elapsed_time):
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
    
def scheduled_sampling(i, high=0.7, low=0.05):
    x = 10 * (i - 0.5)
    z = 1 / (1 + np.exp(x))
    y = (high - low) * z + low
    return y

def kl_anneal(epoch, start_epoch, delta_epoch=5):
    if epoch <= start_epoch:
        return 0
    elif epoch <= start_epoch + delta_epoch:
        return 1.0*(epoch-start_epoch)/delta_epoch
    else:
        return 1.0

def kl_anneal_sigmoid(step, mid_step, start_beta=1e-5):
    half_distance = math.log((1/start_beta)-1)
    x = (step-mid_step)*half_distance/(mid_step-0)
    return 1.0/(1+math.exp(-x))

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, max_count=100):
        self.reset(max_count)

    def reset(self, max_count=100):
        self.val = 0
        self.avg = 0
        self.len = 0
        self.max_count = max_count
    
    def update(self, val, num=1):
        self.val = val/float(num)
        self.avg = (self.avg*self.len+val)/(self.len+num)
        self.len += num

def get_rhythm_target(melody_one_hot):
    rhythm_target = torch.unsqueeze(melody_one_hot[:,:,:-2].sum(-1), -1)
    rhythm_target = torch.cat((rhythm_target, melody_one_hot[:, :, -2:]), -1)
    rhythm_target = rhythm_target.float()
    rhythm_target = rhythm_target.view(-1, rhythm_target.size(-1)).max(-1)[1]
    return rhythm_target

def write_note2ins(data, min_time = 1.0*0.25, ins_name = 'Acoustic Grand Piano'):
    # data: shape of [t, 128], duration matrix. data[i, j] means the duration of pitch j at time step i.
    instrument = pyd.Instrument(program = pyd.instrument_name_to_program(ins_name))
    for i_time in range(data.shape[0]):
        pitches = np.where(data[i_time]>0)[0]
        start_time = i_time*min_time
        for pitch in pitches:
            end_time = start_time + data[i_time, pitch]*min_time
            instrument.notes.append(pyd.Note(velocity=100, pitch=pitch, start=start_time, end=end_time))
    
    return instrument

def write_note2file(data, file_name):
    f_midi = pyd.PrettyMIDI()
    pitch_ins = write_note2ins(data)
    f_midi.instruments.append(pitch_ins)
    f_midi.write(file_name)

def write_multinote2file(datas, file_name):
    f_midi = pyd.PrettyMIDI()
    ins = ['Violin', 'Acoustic Grand Piano']
    for i, data in enumerate(datas):
        pitch_ins = write_note2ins(data, ins_name=ins[i])
        f_midi.instruments.append(pitch_ins)
    f_midi.write(file_name)

def sustain2duration(sustain_data):
    # convert sustain data to duration matrix
    # param:
    #   sustain_data: shape of [t], sustain_data[i] means the note type of melody.
    #                 0~127 means onset of pitch, 128 means hold, 129 means rest.
    # return:
    #   duration_matrix: shape of [t, 128], duration matrix[i, j] means the duration of pitch j at time step i.

    midi_num = 128
    duration_matrix = np.zeros([sustain_data.shape[0], midi_num]).astype(np.int)
    pitch_ind = np.where(sustain_data<hold_pitch)[0]
    hold_list = (sustain_data==hold_pitch)

    pitch_ind = np.append(pitch_ind, sustain_data.shape[0]-1)
    for i in range(pitch_ind.shape[0]):
        if i>=pitch_ind.shape[0]-1:
            break
        duration = 1 + hold_list[pitch_ind[i]:pitch_ind[i+1]].sum()
        duration_matrix[pitch_ind[i], sustain_data[pitch_ind[i]]] = duration
    
    return duration_matrix

def pianogrid2duration(piano_grid, max_pitch, min_pitch):
    # convert piano_grid data to duration matrix
    # param:
    #   sustain_data: shape of (num_step, max_note_count-1, 6), In the last dim, the 0th column is for pitch, 1: 6 is for duration in binary repr
    # return:
    #   duration_matrix: shape of [t, 128], duration matrix[i, j] means the duration of pitch j at time step i.

    bin_weight = np.array([2**i for i in range(piano_grid.shape[2]-2, -1, -1)])

    midi_num = 128
    duration_matrix = np.zeros([piano_grid.shape[0], midi_num]).astype(np.int)
    for t in range(piano_grid.shape[0]):
        for p_i in range(piano_grid.shape[1]):
            pitch = piano_grid[t, p_i, 0]
            if (pitch>=min_pitch and pitch<=max_pitch):
                duration_bin = piano_grid[t, p_i, 1:]
                duration = (duration_bin*bin_weight).sum()+1
                duration_matrix[t, pitch] = duration
            else:
                break
    
    return duration_matrix


def pianogrid2pianogrid(piano_grid, pitch_pad, dur_pad, pitch_sos, pitch_eos):
    # convert piano_grid data to full standard piano_grid
    # param:
    #   piano_grid: shape of (B, num_step, max_note_count-1, 6), In the last dim, the 0th column is for pitch, 1: 6 is for duration in binary repr
    # return:
    #   pr_mat3d: shape of (B, num_step, max_note_count, 6)

    pr_mat3d = torch.ones((piano_grid.shape[0], piano_grid.shape[1], piano_grid.shape[2]+1, piano_grid.shape[3]), device=piano_grid.device).long() * dur_pad
    pr_mat3d[:, :, :, 0] = pitch_pad
    pr_mat3d[:, :, 0, 0] = pitch_sos

    eos_flag = (piano_grid[:,:,:,0]==pitch_eos).int() #(B, num_step, max_note_count-1)
    first_eos_ind = torch.argmax(eos_flag, dim=-1)

    index_arrange_16 = torch.arange(start=0, end=pr_mat3d.shape[2], device=piano_grid.device).view(1,1,-1).repeat([pr_mat3d.shape[0], pr_mat3d.shape[1], 1]) # (B, num_step, max_note_count)
    valid_pitch_flag_16 = (index_arrange_16<=(first_eos_ind+1).unsqueeze(-1))
    valid_pitch_flag_16[:,:,0]=False # (B, num_step, max_note_count)

    eos_pitch_flag_16 = (index_arrange_16==(first_eos_ind+1).unsqueeze(-1)).unsqueeze(-1).repeat(1,1,1,pr_mat3d.shape[3]) # (B, num_step, max_note_count, 6)
    eos_pitch_flag_16[:,:,:,0]=False

    index_arrange_15 = torch.arange(start=0, end=piano_grid.shape[2], device=piano_grid.device).view(1,1,-1).repeat([pr_mat3d.shape[0], pr_mat3d.shape[1], 1]) # (B, num_step, max_note_count-1)
    valid_pitch_flag_15 = (index_arrange_15<=(first_eos_ind).unsqueeze(-1)) # (B, num_step, max_note_count-1)

    pr_mat3d[valid_pitch_flag_16,:] = piano_grid[valid_pitch_flag_15, :]
    pr_mat3d[eos_pitch_flag_16] = dur_pad
    
    return pr_mat3d

def get_melody_onehot_target(melody_dur):
    melody_one_hot = np.zeros([melody_dur.shape[0], notes_dim])
    melody_target = []

    i_time = 0
    while i_time < melody_dur.shape[0]:
        pitch_ind = np.where(melody_dur[i_time]>0)[0]
        assert pitch_ind.shape[0]<2

        if pitch_ind.shape[0]==0:
            melody_one_hot[i_time, rest_pitch] = 1
            melody_target.append(rest_pitch)
            i_time+=1
            continue
        
        duration = int(melody_dur[i_time, pitch_ind[0]])
        duration = min(duration, melody_dur.shape[0]-i_time)
        melody_one_hot[i_time, pitch_ind[0]] = 1
        melody_one_hot[i_time+1:i_time+duration, hold_pitch] = 1

        melody_target.append(pitch_ind[0])
        for _ in range(duration-1):
            melody_target.append(hold_pitch)

        i_time += duration
    
    melody_one_hot = torch.Tensor(melody_one_hot).long()
    melody_target = torch.Tensor(melody_target).long()

    return melody_one_hot, melody_target


def target_to_3dtarget(pr_mat, max_note_count=11, min_pitch=22,
                       pitch_pad_ind=88, dur_pad_ind=2,
                       pitch_sos_ind=86, pitch_eos_ind=87):
    """
    :param pr_mat: (32, 128) matrix. pr_mat[t, p] indicates a note of pitch p,
    started at time step t, has a duration of pr_mat[t, p] time steps.
    :param max_note_count: the maximum number of notes in a time step,
    including <sos> and <eos> tokens.
    :param max_pitch: the highest pitch in the dataset.
    :param min_pitch: the lowest pitch in the dataset.
    :param pitch_pad_ind: see return value.
    :param dur_pad_ind: see return value.
    :param pitch_sos_ind: sos token.
    :param pitch_eos_ind: eos token.
    :return: pr_mat3d is a (32, max_note_count, 6) matrix. In the last dim,
    the 0th column is for pitch, 1: 6 is for duration in binary repr. Output is
    padded with <sos> and <eos> tokens in the pitch column, but with pad token
    for dur columns.
    """
    # pitch_range = max_pitch - min_pitch + 1  # including pad
    pr_mat3d = np.ones((32, max_note_count, 6), dtype=int) * dur_pad_ind
    pr_mat3d[:, :, 0] = pitch_pad_ind
    pr_mat3d[:, 0, 0] = pitch_sos_ind
    cur_idx = np.ones(32, dtype=int)
    for t, p in zip(*np.where(pr_mat != 0)):
        pr_mat3d[t, cur_idx[t], 0] = p - min_pitch
        duration = min(int(pr_mat[t, p]), 32)
        binary = np.binary_repr(duration - 1, width=5)
        pr_mat3d[t, cur_idx[t], 1: 6] = \
            np.fromstring(' '.join(list(binary)), dtype=int, sep=' ')
        cur_idx[t] += 1
    pr_mat3d[np.arange(0, 32), cur_idx, 0] = pitch_eos_ind
    return pr_mat3d


def pitch_dur_accuracy(piano_grid, recon_pitch, recon_dur, pitch_pad, dur_pad):
    # piano_grid #(B, num_step, max_simu_note, 6)
    # recon_pitch (B, num_step, max_simu_note-1, pitch_range); 
    # recon_dur (B, num_step, max_simu_note-1, dur_width, 2)

    pitch_valid = (piano_grid[:, :, 1:, 0]!=pitch_pad) # (B, num_step, max_simu_note-1)
    dur_valid = (piano_grid[:, :, 1:, 1:]!=dur_pad) # (B, num_step, max_simu_note-1, dur_width)

    recon_pitch_targ = recon_pitch.max(-1)[1] #(B, num_step, max_simu_note-1)
    recon_pitch_equal_batch = torch.logical_and(recon_pitch_targ==piano_grid[:, :, 1:, 0], pitch_valid).view(recon_pitch_targ.shape[0], -1) #(B, num_step*(max_simu_note-1))
    recon_pitch_acc_batch = recon_pitch_equal_batch.sum(1).float()/(pitch_valid.view(pitch_valid.shape[0], -1).sum(1).float()) #(B)
    recon_pitch_acc = recon_pitch_acc_batch.mean()

    recon_dur_targ = recon_dur.max(-1)[1] # (B, num_step, max_simu_note-1, dur_width)
    recon_dur_equal_batch = torch.logical_and(recon_dur_targ==piano_grid[:, :, 1:, 1:], dur_valid).view(recon_dur_targ.shape[0], -1) # (B, num_step*(max_simu_note-1)*dur_width)
    recon_dur_acc_batch = recon_dur_equal_batch.sum(1).float()/(dur_valid.view(dur_valid.shape[0], -1).sum(1).float()) #(B)
    recon_dur_acc = recon_dur_acc_batch.mean()
    
    return recon_pitch_acc, recon_dur_acc, recon_pitch_acc_batch, recon_dur_acc_batch

def chord_accuracy(chords_multihot, recon_root, recon_chroma, recon_bass):
    # chords_multihot (B, 8, 36), root, bass, chroma np.float32
    # recon_root (B, 8, 12); recon_chroma (B, 8, 12, 2); recon_bass (B, 8, 12)
    def one_hot_accuracy(target, output):
        batch_equal = (target==output).view(target.shape[0], -1)
        batch_accuracy = batch_equal.float().mean(1)
        accuracy = batch_accuracy.mean()
        return batch_accuracy, accuracy

    root_batch_acc, root_acc = one_hot_accuracy(chords_multihot[:, :, :12].max(-1)[1], recon_root.max(-1)[1])
    bass_batch_acc, bass_acc = one_hot_accuracy(chords_multihot[:, :, 12:24].max(-1)[1], recon_bass.max(-1)[1])
    chroma_batch_acc, chroma_acc = one_hot_accuracy(chords_multihot[:, :, 24:].long(), recon_chroma.max(-1)[1])

    return root_acc, bass_acc, chroma_acc, root_batch_acc, bass_batch_acc, chroma_batch_acc

def chord_name2multihot(chord_name):
    # pattern = '([CDEFGAB])([#b])?:((?:maj[6]?)|(?:min[7]?)|7|(?:aug))?(sus[24])?(/b?[357])?(\(b?7\))?$'
    pattern = '([CDEFGAB])([#b])?:((?:maj[67]?)|(?:min[67]?)|7|(?:aug)|(?:sus[24])|(?:dim[7]?)|(?:minmaj7)|(?:hdim7))(/b?[357])?(\(b?7\))?$'
    try:
        match_groups = re.search(pattern, chord_name).groups()
    except:
        import pdb
        pdb.set_trace()

    ####### unvalid chord name ############
    # both (b7),(7) and min7/7/hdim7
    if match_groups[4] and '7' in match_groups[2]:
        import pdb
        pdb.set_trace()

    # both (b7),(7) and /b3
    if match_groups[4] and match_groups[3]:
        import pdb
        pdb.set_trace()

    if match_groups[1]=='b':
        acc='-'
    elif match_groups[1]=='#':
        acc='#'
    else:
        acc=''
    root_name = match_groups[0]+acc

    
    if match_groups[2]=='maj6':
        name = root_name + '6'
    elif match_groups[2]=='hdim7':
        name = root_name+'m7b5'
    else:
        name = root_name+match_groups[2]
        
    pitches = harmony.ChordSymbol(name).pitches

    ########### root #############
    root_num = music21.pitch.Pitch(root_name).pitchClass
    root_one_hot = np.zeros([12]).astype(np.int)
    root_one_hot[root_num]=1

    ########### bass ##############
    relative_num = {'b3':3, '3':4, 'b5':6, '5':7, 'b7':10, '7':11}
    bass_one_hot = np.zeros([12]).astype(np.int)
    if match_groups[3]:
        bass_num = (root_num + relative_num[match_groups[3][1:]])%12
    else:
        bass_num = root_num
    bass_one_hot[bass_num] = 1

    ########### chroma #############
    chroma = np.zeros([12]).astype(np.int)
    for pitch in pitches:
        p_i = pitch.pitchClass
        chroma[p_i] = 1
    
    if match_groups[4]:
        seven_num = (root_num + relative_num[match_groups[4][1:-1]])%12
        chroma[seven_num] = 1

    return np.concatenate([root_one_hot, bass_one_hot, chroma])

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def mean_parallel_loss(loss, gpus_bsz):
    gpus_bsz = torch.Tensor(gpus_bsz).cuda()
    if isinstance(loss, tuple):
        # assert len(loss[0])==len(gpus_bsz)            
        return tuple([(x*gpus_bsz).sum()/gpus_bsz.sum() for x in loss])
    else:
        # assert len(loss)==len(gpus_bsz)
        return (loss*gpus_bsz).sum()/gpus_bsz.sum()


def set_log_file(fname, file_only=False):
    # set log file
    # simple tricks for duplicating logging destination in the logging module such as:
    # logging.getLogger().addHandler(logging.FileHandler(filename))
    # does NOT work well here, because python Traceback message (not via logging module) is not sent to the file,
    # the following solution (copied from : https://stackoverflow.com/questions/616645) is a little bit
    # complicated but simulates exactly the "tee" command in linux shell, and it redirects everything
    if file_only:
        # we only output messages to file, and stdout/stderr receives nothing.
        # this feature is designed for executing the script via ssh:
        # since ssh has a windowing kind of flow control, i.e., if the controller does not read data from a
        # ssh channel and its buffer fills up, the execution machine will not be able to write anything into the
        # channel and the process will be set to sleeping (S) status until someone reads all data from the channel.
        # this is not desired since we do not want to read stdout/stderr from the controller machine.
        # so, here we use a simple solution: disable output to stdout/stderr and only output messages to log file.
        log.logger.handlers[0].stream = log.handler.stream = sys.stdout = sys.stderr = open(fname, 'w', buffering=1)
    else:
        # we output messages to both file and stdout/stderr
        import subprocess
        tee = subprocess.Popen(['tee', fname], stdin=subprocess.PIPE)
        os.dup2(tee.stdin.fileno(), sys.stdout.fileno())
        os.dup2(tee.stdin.fileno(), sys.stderr.fileno())

def get_EC2_VAE(model_root):
    with open(os.path.join(model_root, 'EC2_VAE/model_config.json')) as f:
        args = json.load(f)
    
    MELODY_DIMS = 130
    RHYTHM_DIMS = 3
    
    condition_dims=0
    model = EC2_VAE(roll_dims = MELODY_DIMS, hidden_dims = args['hidden_dim'], rhythm_dims = RHYTHM_DIMS, 
            condition_dims = condition_dims, z1_dims = args['pitch_dim'],
            z2_dims = args['rhythm_dim'], n_step = args['time_step'], eps=args['eps'])
    
    params_path = os.path.join(model_root, 'params_EC2')
    checkpoint = torch.load(os.path.join(params_path, os.listdir(params_path)[0]))
    model.load_state_dict({k.replace('module.',''):v for k,v in checkpoint['state_dict'].items()})

    model.cuda()

    return model

def get_poly_model(model_root):
    config_fn = os.path.join(model_root, 'poly/model_config.json')
    with open(config_fn) as f:
        args = json.load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    chd_encoder = ChordEncoder(input_dim=args['chordEncoder_in_dim'], hidden_dim=args['chordEncoder_hidden_dim'], z_dim=args['chordEncoder_z_dim'])
    tex_encoder = TextureEncoder(emb_size=args['texEncoder_cnn_dim'], hidden_dim=args['texEncoder_gru_hidden_dim'], z_dim=args['texEncoder_z_dim'],
                                num_channel=args['texEncoder_cnn_channel'])

    chd_decoder = ChordDecoder(z_dim=args['chordEncoder_z_dim'], z_input_dim=args['chordDecoder_inpuy_dim'], hidden_dim=args['chordDecoder_hidden_dim'])
    pt_decoder = PtvaeDecoder(note_embedding=None, note_emb_size=args['ptdec_note_emb_size'], z_size=args['chordEncoder_z_dim']+args['texEncoder_z_dim'],
                                dec_emb_hid_size=args['ptdec_emb_hid_size'], dec_time_hid_size=args['ptdec_time_hid_size'], dec_notes_hid_size=args['ptdec_notes_hid_size'],
                                dec_z_in_size=args['ptdec_z_in_size'], dec_dur_hid_size=args['ptdec_dur_hid_size'])

    model = Poly_model('disvae-nozoth', device, chd_encoder, tex_encoder, pt_decoder, chd_decoder, amp=args['amp'])

    params_path = os.path.join(model_root, 'params_poly')
    checkpoint = torch.load(os.path.join(params_path, os.listdir(params_path)[0]))
    model.load_state_dict({k.replace('module.',''):v for k,v in checkpoint['state_dict'].items()})
    model.cuda()

    return model

def critic_accuracy(D_output, label, threshold=0.5):
    D_label = (D_output>threshold)
    batch_equal = (D_label==(label==1)).float()
    return batch_equal.mean(), batch_equal

def G_output_demo(base_root, z_melody, z_accompany, Gz_melody, Gz_accompany, EC2_model, poly_model, name_melody=None, name_accompany=None,  melody_target=None, piano_grid=None,
                     D_output_n=None, D_output_G=None, eval_n=None, eval_G=None, melody_pitch_distance_batch=None, melody_rhythm_distance_batch=None, accompany_chord_distance_batch=None, accompany_texture_distance_batch=None):
    MAX_PITCH = 127
    MIN_PITCH = 0
    PITCH_PAD = 130
    DUR_PAD=2

    melody_z1_dims = EC2_model.z1_dims
    accompany_chord_dim = poly_model.chd_encoder.z_dim
    with torch.no_grad():
        recon_melody, recon_piano_grid, pitch_outs, dur_outs = decodez2music(z_melody, z_accompany, EC2_model, poly_model)
        recon_G_melody, recon_G_piano_grid, _, _ = decodez2music(Gz_melody, Gz_accompany, EC2_model, poly_model)
        # [32, 130]
        recon_piano_grid = recon_piano_grid.cpu().numpy()
        recon_G_piano_grid = recon_G_piano_grid.cpu().numpy()
        
        if type(melody_target)!=type(None):
            melody_recon_acc_batch = (recon_melody==melody_target).float().mean(1)*100
            melody_recon_acc = melody_recon_acc_batch.mean()

            melody_target = melody_target.cpu().numpy()

        if type(piano_grid)!=type(None):
            accompany_recon_pitch_acc, accompany_recon_dur_acc, accompany_recon_pitch_acc_batch, accompany_recon_dur_acc_batch = \
                    pitch_dur_accuracy(piano_grid, pitch_outs, dur_outs, pitch_pad=PITCH_PAD, dur_pad=DUR_PAD)
            accompany_recon_pitch_acc *= 100
            accompany_recon_dur_acc *= 100
            accompany_recon_pitch_acc_batch *= 100
            accompany_recon_dur_acc_batch *= 100

            piano_grid = piano_grid.cpu().numpy()
    
    recon_melody = recon_melody.cpu().numpy()
    recon_G_melody = recon_G_melody.cpu().numpy()
    
    for i in range(z_melody.shape[0]):
        name_order = str(i)+ 'm' + name_melody[i] + 'a' + name_accompany[i] if name_melody else str(i)
        name_prior = "nmatch" if name_melody else "prior"

        recon_melody_duration = sustain2duration(recon_melody[i])
        recon_poly_duration = pianogrid2duration(recon_piano_grid[i], max_pitch=MAX_PITCH, min_pitch=MIN_PITCH) # [t, 128]
        D_output_str = '_dout%.2f'%(D_output_n[i]*100) if type(D_output_n)!=type(None) else ''
        eval_str = '_eval%.2f'%(eval_n[i]*100) if type(eval_n)!=type(None) else ''
        recon_str = '_mc%.2fapc%.2fadc%.2f'%(melody_recon_acc_batch[i], accompany_recon_pitch_acc_batch[i], accompany_recon_dur_acc_batch[i]) if type(melody_target)!=type(None) else ''
        name_i_recon = name_order + '_recon_' + name_prior + D_output_str + eval_str + recon_str +'.mid'
        write_multinote2file(datas=[recon_melody_duration, recon_poly_duration], file_name=os.path.join(base_root, name_i_recon))

        recon_G_melody_duration = sustain2duration(recon_G_melody[i])
        recon_G_poly_duration = pianogrid2duration(recon_G_piano_grid[i], max_pitch=MAX_PITCH, min_pitch=MIN_PITCH) # [t, 128]
        D_output_str = '_dout%.2f'%(D_output_G[i]*100) if type(D_output_G)!=type(None) else ''
        eval_str = '_eval%.2f'%(eval_G[i]*100) if type(eval_G)!=type(None) else ''
        name_i_reconG = name_order + '_reconG_' + name_prior + D_output_str + eval_str + '_mpdis%.2fmrdis%.2facdis%.2fatdis%.2f'%(melody_pitch_distance_batch[i], melody_rhythm_distance_batch[i], accompany_chord_distance_batch[i], accompany_texture_distance_batch[i]) +'.mid'
        write_multinote2file(datas=[recon_G_melody_duration, recon_G_poly_duration], file_name=os.path.join(base_root, name_i_reconG))

        if type(melody_target)!=type(None):
            origin_melody_duration = sustain2duration(melody_target[i])
            origin_poly_duration = pianogrid2duration(piano_grid[i, :, 1:, :], max_pitch=MAX_PITCH, min_pitch=MIN_PITCH) # [t, 128]
            name_i_recon = name_order + '_origin_' + name_prior +'.mid'
            write_multinote2file(datas=[origin_melody_duration, origin_poly_duration], file_name=os.path.join(base_root, name_i_recon))

    if type(melody_target)!=type(None):
        return melody_recon_acc, accompany_recon_pitch_acc, accompany_recon_dur_acc


def D_grad_G(critic_model, z_nmatch_melody, z_nmatch_accompany, average_scale_melody, average_scale_accompany, adam_melody_lr=0.01, adam_accompany_lr=0.01, adam_betas=[0.9, 0.999], adam_step=100):
    z_nmatch_melody_param = torch.nn.Parameter(z_nmatch_melody.clone())
    z_nmatch_accompany_param = torch.nn.Parameter(z_nmatch_accompany.clone())
    z_optimizer = torch.optim.Adam([{'params': z_nmatch_melody_param, 'lr':adam_melody_lr},
                                    {'params': z_nmatch_accompany_param, 'lr':adam_accompany_lr}], betas=adam_betas)

    first_output_D = critic_model(z_nmatch_melody_param, z_nmatch_accompany_param)
    label_D = torch.ones([z_nmatch_accompany.shape[0], 1], device="cuda")

    loss_D_all = []
    fool_acc_all = []
    melody_distance_mean_all = []
    accompany_distance_mean_all = []
    for i in range(adam_step):
        output_D = critic_model(z_nmatch_melody_param, z_nmatch_accompany_param)
        loss_D = F.binary_cross_entropy(output_D, label_D, reduction='mean')
        loss_D.backward()
        z_optimizer.step()
        fool_acc, _ = critic_accuracy(D_output=output_D, label=label_D)

        melody_distance = (z_nmatch_melody_param-z_nmatch_melody).pow(2)
        accompany_distance = (z_nmatch_accompany_param-z_nmatch_accompany).pow(2)
        melody_distance_mean = (melody_distance*average_scale_melody.pow(-2)).mean()
        accompany_distance_mean = (accompany_distance*average_scale_accompany.pow(-2)).mean()

        loss_D_all.append(loss_D.item())
        fool_acc_all.append(fool_acc.item())
        melody_distance_mean_all.append(melody_distance_mean.item())
        accompany_distance_mean_all.append(accompany_distance_mean.item())

    final_output_D = critic_model(z_nmatch_melody_param, z_nmatch_accompany_param)
    final_fool_acc, _ = critic_accuracy(D_output=final_output_D, label=label_D)
    loss_D = F.binary_cross_entropy(final_output_D, label_D, reduction='mean')

    loss_D_all.append(loss_D.item())
    fool_acc_all.append(final_fool_acc.item())
    return z_nmatch_melody_param, z_nmatch_accompany_param, loss_D_all, fool_acc_all, melody_distance_mean_all, accompany_distance_mean_all, final_output_D, first_output_D

def delte_and_makedir(dir_path):
    if os.path.exists(dir_path):
        re = os.system("rm -r "+dir_path)
        assert re==0
    os.makedirs(dir_path, exist_ok=False)

def eval_z(eval_model, EC2_model, poly_model, z_melody, z_accompany, batch_size=-1):
    melody_z1_dims = EC2_model.z1_dims
    with torch.no_grad():
        if batch_size==-1:
            batch_num = 1
            batch_size = z_melody.shape[0]
        else:
            batch_num = z_melody.shape[0]//batch_size+1

        eval_result = []
        for i in range(batch_num):
            recon_melody_target, recon_piano_grid, _, _ = decodez2music(z_melody[i*batch_size:(i+1)*batch_size], z_accompany[i*batch_size:(i+1)*batch_size], EC2_model, poly_model)
            recon_melody_one_hot = F.one_hot(recon_melody_target, num_classes=notes_dim)

            pr_mat3d = pianogrid2pianogrid(recon_piano_grid, pitch_pad=poly_model.decoder.pitch_pad, dur_pad=poly_model.decoder.dur_pad,
                                        pitch_sos=poly_model.decoder.pitch_sos, pitch_eos=poly_model.decoder.pitch_eos)

            del recon_piano_grid, recon_melody_target, _
            torch.cuda.empty_cache()

            eval_result_i = eval_model(x_accompany=pr_mat3d, x_melody=recon_melody_one_hot.float())            
            eval_result.append(eval_result_i.cpu())
            del eval_result_i
            torch.cuda.empty_cache()

        eval_result = torch.cat(eval_result, dim=0)
        
    return eval_result

def decodez2music(z_melody, z_accompany, EC2_model, poly_model):
    melody_z1_dims = EC2_model.z1_dims

    recon_melody = EC2_model.decoder(z1=z_melody[:, :melody_z1_dims], z2=z_melody[:, melody_z1_dims:]) # [B, 32, 130]
    recon_melody_target = recon_melody.max(-1)[1]

    pitch_outs, dur_outs = poly_model.decoder(z_accompany, True, None, None, 0., 0.)
    recon_pitch_targ = pitch_outs.max(-1)[1] # (B, num_step, max_simu_note-1)
    recon_dur_targ = dur_outs.max(-1)[1] # (B, num_step, max_simu_note-1, dur_width)
    recon_piano_grid = torch.cat((recon_pitch_targ.unsqueeze(-1), recon_dur_targ), dim=-1) # (B, num_step, max_simu_note-1, 6)

    return recon_melody_target, recon_piano_grid, pitch_outs, dur_outs

def z_distance_c(c_batch, distance_batch, c_threshold=0.5):
    # c_batch: [N, 1]
    # distance_batch: [N, 1]
    # distance should increase as c
    assert c_batch.shape[1]==1
    assert distance_batch.shape[1]==1

    N = c_batch.shape[0]
    distance_batch_norm = (distance_batch - distance_batch.min())/(distance_batch.max()-distance_batch.min()+1e-5) # [N, 1]

    c_batch_i = c_batch.repeat([1, N]) # [N, N], c_batch_i[i,j]=c_batch[i]
    c_batch_ij = c_batch_i - c_batch_i.t() # [N, N], c_batch_ij[i, j]=c_batch[i]-c_batch[j]

    distance_batch_norm_i = distance_batch_norm.repeat([1, N]) # [N, N], distance_batch_norm_i[i,j]=distance_batch_norm_i[i]
    distance_batch_norm_ij = distance_batch_norm_i - distance_batch_norm_i.t() # [N, N], distance_batch_norm_ij[i,j]=distance_batch_norm[i]-distance_batch_norm[j]

    distance_div_c = distance_batch_norm_ij/(c_batch_ij+1e-7)
    distance_div_c_angle = torch.atan(distance_div_c)/np.pi*180 #[N, N]
    
    return distance_div_c_angle[c_batch_ij>c_threshold].mean()