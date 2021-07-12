import os
import sys
import glog as log
import torch
import torch.utils.data as data
import torch.nn.functional as F
import numpy as np

p = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
if p not in sys.path:
    sys.path.append(p)
from utils import get_melody_onehot_target, target_to_3dtarget

hold_pitch = 128
rest_pitch = 129

class Melody_dataset(data.Dataset):

    def __init__(self, data_path, notes_dim, shift_low=0, shift_high=0):
        assert notes_dim == rest_pitch+1
        self.shift_low = shift_low
        self.shift_high = shift_high

        data = np.load(data_path, allow_pickle=True)
        self.data = data
        self.len = data.shape[0]

        self.all_melody_one_hot = []
        self.all_melody_target = []

        for index in range(data.shape[0]):
            melody_dur = data[index]["melody"]
            melody_one_hot, melody_target = get_melody_onehot_target(melody_dur)
            
            self.all_melody_one_hot.append(melody_one_hot)
            self.all_melody_target.append(melody_target)

        print("data len:", len(self.all_melody_one_hot), flush=True)

    def __len__(self):
        return len(self.all_melody_one_hot) * (self.shift_high - self.shift_low + 1)

    def __getitem__(self, index):
        no = index // (self.shift_high - self.shift_low + 1)
        shift = index % (self.shift_high - self.shift_low + 1) + self.shift_low
        
        melody = self.all_melody_one_hot[no] #[32, 130]
        melody_target = self.all_melody_target[no]

        # perform pitch shifting using np.roll and masking
        melody_shift = np.roll(melody, shift, axis=1)
        if shift > 0:
            melody_shift[:, :shift] = 0
        elif shift < 0:
            melody_shift[:, shift:] = 0
        
        melody_shift[:, hold_pitch:] = melody[:, hold_pitch:]

        melody_target_shift = torch.clone(melody_target)
        melody_target_shift[torch.logical_and(melody_target!=hold_pitch, melody_target!=rest_pitch)] += shift

        name = self.data[no]["name"]
        return melody_shift, melody_target_shift, name


class PolyphonicDataset(data.Dataset):

    def __init__(self, filepath, pitch_range, pitch_pad, shift_low, shift_high, contain_chord=False):
        super(PolyphonicDataset, self).__init__()
        self.filepath = filepath
        self.shift_low = shift_low
        self.shift_high = shift_high
        self.data = np.load(self.filepath, allow_pickle=True)
        self.contain_chord = contain_chord

        self.max_simu_note = 16
        self.pitch_pad = pitch_pad
        self.pitch_range = pitch_range # max_pitch - min_pitch + 1 + 2, 130, not including pad, including sos and eos.

    def __len__(self):
        # consider data augmentation here
        return len(self.data) * (self.shift_high - self.shift_low + 1)

    def __getitem__(self, id):
        # separate id into (no, shift) pair
        no = id // (self.shift_high - self.shift_low + 1)
        shift = id % (self.shift_high - self.shift_low + 1) + self.shift_low
        data = self.data[no]['accompany'] #[32, 128], duration matrix
        name = self.data[no]['name']
        # perform pitch shifting using np.roll and masking
        
        data = np.roll(data, shift, axis=1)
        if shift > 0:
            data[:, :shift] = 0
        elif shift < 0:
            data[:, shift:] = 0
        # if you want to convert data into MIDI message,
        # insert your converter code after this line.
        # pr: (32, 128, 3)
        pr_matrix = data # (32, 128), duration of each note of each step. duration is from 0~32
        # 14 - 115, 14
        # 21 - 105， 11
        piano_grid = target_to_3dtarget(pr_matrix, max_note_count=self.max_simu_note,
                                              min_pitch=0, pitch_pad_ind=self.pitch_pad,
                                              pitch_sos_ind=128,
                                              pitch_eos_ind=129) #(32, max_note_count, 6) matrix. In the last dim, the 0th column is for pitch, 1: 6 is for duration in binary repr
        
        ########## chord #############
        if self.contain_chord:
            chords_multihot = self.data[no]['chords_multihot'] # [8, 36]
            root_onehot = np.roll(chords_multihot[:, :12], shift, axis=1)
            bass_onehot = np.roll(chords_multihot[:, 12:24], shift, axis=1)
            chroma_multihot = np.roll(chords_multihot[:, 24:], shift, axis=1)
            chords_multihot_shift = np.concatenate([root_onehot, bass_onehot, chroma_multihot], axis=1) # [8, 36]

            return piano_grid.astype(np.int64), pr_matrix.astype(np.float32), chords_multihot_shift.astype(np.float32), name
        else:
            return piano_grid.astype(np.int64), pr_matrix.astype(np.float32), name

class ZDataset(data.Dataset):

    def __init__(self, melody_path, accompany_path, pitch_pad, pitch_range, z_nmatch_fix=False, sample=False, with_music=True):
        super(ZDataset, self).__init__()
        melody_data = np.load(melody_path)
        if with_music:
            self.melody_name_all = melody_data['name']
            self.melody_one_hot_all = melody_data['melody_one_hot']
            self.melody_target_all = melody_data['melody_target']

        self.z_mean_melody_all = melody_data['z']
        self.z_std_melody_all = melody_data['std']
        self.z_melody_std_average = melody_data['std_average']

        accompany_data = np.load(accompany_path)
        if with_music:
            self.accompany_name_all = accompany_data['name']
            self.pr_matrix_all = accompany_data['pr_matrix']
            self.chords_multihot_all = accompany_data['chords_multihot']
        self.z_mean_accompany_all = accompany_data['z']
        self.z_std_accompany_all = accompany_data['std']
        self.z_accompany_std_average = accompany_data['std_average']

        self.max_simu_note = 16
        self.pitch_pad = pitch_pad
        self.pitch_range = pitch_range # max_pitch - min_pitch + 1 + 2, 130, not including pad, including sos and eos.

        assert self.z_mean_melody_all.shape[0]==self.z_mean_accompany_all.shape[0]
        
        if z_nmatch_fix:
            self.nmatch_melody_ids = np.random.randint(low=0, high=self.z_mean_melody_all.shape[0], size=self.z_mean_melody_all.shape[0])
            self.nmatch_accompany_ids = np.random.randint(low=0, high=self.z_mean_melody_all.shape[0], size=self.z_mean_melody_all.shape[0])

        self.z_nmatch_fix = z_nmatch_fix
        self.sample = sample
        self.with_music=with_music

        log.info("data len: %d"%self.__len__())
        del melody_data, accompany_data

    def get_std_average(self):
        return torch.from_numpy(self.z_melody_std_average).cuda(), torch.from_numpy(self.z_accompany_std_average).cuda()

    def __len__(self):
        return self.z_mean_melody_all.shape[0]

    def get_z(self, melody_id, accompany_id):
        z_mean_melody = torch.from_numpy(self.z_mean_melody_all[melody_id])
        z_mean_accompany = torch.from_numpy(self.z_mean_accompany_all[accompany_id])
        
        if self.sample:
            z_std_melody = torch.from_numpy(self.z_std_melody_all[melody_id])
            z_std_accompany = torch.from_numpy(self.z_std_accompany_all[accompany_id])

            z_melody = torch.randn_like(z_mean_melody)*z_std_melody+z_mean_melody
            z_accompany = torch.randn_like(z_mean_accompany)*z_std_accompany+z_mean_accompany
        else:
            z_melody = z_mean_melody
            z_accompany = z_mean_accompany
        
        return z_melody, z_accompany

    def get_music(self, melody_id, accompany_id):
        if self.with_music:
            melody_target = self.melody_target_all[melody_id].astype(np.int)
            name_melody = self.melody_name_all[melody_id]

            pr_matrix = self.pr_matrix_all[accompany_id].astype(np.int) # (32, 128), duration of each note of each step. duration is from 0~32
            name_accompany = self.accompany_name_all[accompany_id]

            piano_grid = target_to_3dtarget(pr_matrix, max_note_count=self.max_simu_note,
                                                min_pitch=0, pitch_pad_ind=self.pitch_pad,
                                                pitch_sos_ind=128,
                                                pitch_eos_ind=129) #(32, max_note_count, 6) matrix. In the last dim, the 0th column is for pitch, 1: 6 is for duration in binary repr
        else:
            melody_target = 0
            name_melody = 0
            piano_grid = 0
            name_accompany = 0

        return melody_target, name_melody, piano_grid, name_accompany

    def __getitem__(self, id):
        # assert self.accompany_name_all[id]==self.melody_name_all[id]
        
        if self.z_nmatch_fix:
            nmatch_melody_id = self.nmatch_melody_ids[id]
            nmatch_accompany_id = self.nmatch_accompany_ids[id]
        else:
            nmatch_melody_id = np.random.randint(self.__len__())
            nmatch_accompany_id = np.random.randint(self.__len__())

            while nmatch_melody_id == nmatch_accompany_id:
                nmatch_melody_id = np.random.randint(self.__len__())
                nmatch_accompany_id = np.random.randint(self.__len__())

        z_match_melody, z_match_accompany = self.get_z(id, id)
        z_nmatch_melody, z_nmatch_accompany = self.get_z(nmatch_melody_id, nmatch_accompany_id)
        

        melody_target_match, name_match_melody, piano_grid_match, name_match_accompany = self.get_music(id, id)
        melody_target_nmatch, name_nmatch_melody, piano_grid_nmatch, name_nmatch_accompany = self.get_music(nmatch_melody_id, nmatch_accompany_id)
        
        return (z_match_melody, melody_target_match, name_match_melody), (z_match_accompany, piano_grid_match, name_match_accompany), \
                (z_nmatch_melody, melody_target_nmatch, name_nmatch_melody), (z_nmatch_accompany, piano_grid_nmatch, name_nmatch_accompany)

def save_z_distribute(tracker, name, eval=False):
    import pandas as pd
    import seaborn as sns
    df = pd.DataFrame.from_dict(tracker, orient='index')
    g = sns.lmplot(x='z_melody', y='z_accompany', hue='match', data=df, fit_reg=False)
    g.savefig(name, dpi=300)

if __name__ == '__main__':
    import json
    config_fn = './adversarial/model_config.json'
    with open(config_fn) as f:
        args = json.load(f)

    PITCH_PAD = 130
    MAX_PITCH = 127
    MIN_PITCH = 0
    pitch_range = MAX_PITCH-MIN_PITCH+1+2
    train_dataset = ZDataset(melody_path=os.path.join(args['data_path'], 'train_EC2.npz'), accompany_path=os.path.join(args['data_path'], 'train_poly.npz'),
                     pitch_pad=PITCH_PAD, pitch_range=pitch_range, z_nmatch_fix=args['z_nmatch_fix'], sample=args['z_sample'])
    val_dataset = ZDataset(melody_path=os.path.join(args['data_path'], 'test_EC2.npz'), accompany_path=os.path.join(args['data_path'], 'test_poly.npz'),
                     pitch_pad=PITCH_PAD, pitch_range=pitch_range, z_nmatch_fix=False, sample=False)
    
    z_melody_index = np.argmin(train_dataset.z_std_melody_all.mean(0))
    z_accompany_index = np.argmin(train_dataset.z_std_accompany_all.mean(0))

    output_dataset = train_dataset

    from collections import defaultdict
    tracker = defaultdict(lambda: defaultdict(dict))
    draw_num = 2000
    match_index = np.random.randint(low=0, high=output_dataset.__len__(), size=draw_num)
    nmatch_melody_indexes = np.random.randint(low=0, high=output_dataset.__len__(), size=draw_num)
    nmatch_accompany_indexes = np.random.randint(low=0, high=output_dataset.__len__(), size=draw_num)
    for i in range(draw_num):
        tracker[i*2]['z_melody'] = output_dataset.z_mean_melody_all[match_index[i]][z_melody_index]
        tracker[i*2]['z_accompany'] = output_dataset.z_mean_accompany_all[match_index[i]][z_accompany_index]
        tracker[i*2]['match'] = 1

        if nmatch_melody_indexes[i]!=nmatch_accompany_indexes[i]:
            tracker[i*2+1]['z_melody'] = output_dataset.z_mean_melody_all[nmatch_melody_indexes[i]][z_melody_index]
            tracker[i*2+1]['z_accompany'] = output_dataset.z_mean_accompany_all[nmatch_accompany_indexes[i]][z_accompany_index]
            tracker[i*2+1]['match'] = 0

    save_z_distribute(tracker, 'z_dis.jpg')