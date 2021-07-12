import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.distributions import Normal

class PianoTree_evaluation(nn.Module):
    def __init__(self, device=None, max_simu_note=16, max_pitch=127, min_pitch=0, pitch_sos=128, 
                pitch_eos=129, pitch_pad=130, dur_pad=2, dur_width=5, num_step=32, note_emb_size=128, 
                enc_notes_hid_size=256, enc_time_hid_size=512, melody_roll_dims=130, melody_hidden_dims=1024,
                before_cat_dim = 512, mid_sizes=[256, 64, 1], drop_out_p=0.5):
        super(PianoTree_evaluation, self).__init__()

        # Parameters
        # note and time
        self.max_pitch = max_pitch  # the highest pitch in train/val set.
        self.min_pitch = min_pitch  # the lowest pitch in train/val set.
        self.pitch_sos = pitch_sos
        self.pitch_eos = pitch_eos
        self.pitch_pad = pitch_pad
        self.pitch_range = max_pitch - min_pitch + 1 + 2  # not including pad.
        self.dur_pad = dur_pad
        self.dur_width = dur_width
        self.note_size = self.pitch_range + dur_width
        self.max_simu_note = max_simu_note  # the max # of notes at each ts.
        self.num_step = num_step  # 32
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        self.note_emb_size = note_emb_size
        self.enc_notes_hid_size = enc_notes_hid_size
        self.enc_time_hid_size = enc_time_hid_size

        self.melody_gru = nn.GRU(melody_roll_dims, melody_hidden_dims, batch_first=True, bidirectional=True) # melody encode

        self.note_embedding = nn.Linear(self.note_size, note_emb_size)
        self.enc_notes_gru = nn.GRU(note_emb_size, enc_notes_hid_size,
                                    num_layers=1, batch_first=True,
                                    bidirectional=True)
        self.enc_time_gru = nn.GRU(2 * enc_notes_hid_size, enc_time_hid_size,
                                   num_layers=1, batch_first=True,
                                   bidirectional=True)
        
        self.melody_before_cat_linear = nn.Linear(melody_hidden_dims*2, before_cat_dim)
        self.accompany_before_cat_linear = nn.Linear(enc_time_hid_size*2, before_cat_dim)

        linear_layers = [nn.ReLU(), nn.Dropout(drop_out_p)]
        mid_sizes.insert(0, 2*before_cat_dim)
        for i in range(len(mid_sizes)-2):
            linear_layers.append(nn.Linear(mid_sizes[i], mid_sizes[i+1]))
            linear_layers += [nn.ReLU(), nn.Dropout(drop_out_p)]
        linear_layers.append(nn.Linear(mid_sizes[-2], mid_sizes[-1]))
        if mid_sizes[-1]==1:
            linear_layers.append(nn.Sigmoid())

        self.linear_layers = nn.Sequential(*linear_layers)

    def get_len_index_tensor(self, ind_x):
        """Calculate the lengths ((B, 32), torch.LongTensor) of pgrid."""
        with torch.no_grad():
            lengths = self.max_simu_note - \
                      ((ind_x[:, :, :, 0] - self.pitch_pad) == 0).sum(dim=-1)
        return lengths

    def index_tensor_to_multihot_tensor(self, ind_x):
        """Transfer piano_grid to multi-hot piano_grid."""
        # ind_x: (B, 32, max_simu_note, 1 + dur_width)
        with torch.no_grad():
            dur_part = ind_x[:, :, :, 1:].float()
            out = torch.zeros(
                [ind_x.size(0) * self.num_step * self.max_simu_note,
                 self.pitch_range + 1],
                dtype=torch.float).to(self.device)

            out[range(0, out.size(0)), ind_x[:, :, :, 0].view(-1)] = 1.
            out = out.view(-1, 32, self.max_simu_note, self.pitch_range + 1)
            out = torch.cat([out[:, :, :, 0: self.pitch_range], dur_part],
                            dim=-1)
        return out

    def encoder(self, x_accompany, lengths, x_melody):
        embedded = self.note_embedding(x_accompany)
        # x: (B, num_step, max_simu_note, note_emb_size)
        # now x are notes
        x_accompany = embedded.view(-1, self.max_simu_note, self.note_emb_size)
        x_accompany = pack_padded_sequence(x_accompany, lengths.view(-1).cpu(), batch_first=True,
                                 enforce_sorted=False)
        x_accompany = self.enc_notes_gru(x_accompany)[-1].transpose(0, 1).contiguous()
        x_accompany = x_accompany.view(-1, self.num_step, 2 * self.enc_notes_hid_size)
        # now, x is simu_notes.
        x_accompany = self.enc_time_gru(x_accompany)[-1].transpose(0, 1).contiguous()
        # x: (B, 2, enc_time_hid_size)
        x_accompany = x_accompany.view(x_accompany.size(0), -1)
        x_accompany = self.accompany_before_cat_linear(x_accompany) # [batch, before_cat_dim]

        x_melody = self.melody_gru(x_melody)[-1] # h_n of shape [num_layers * num_directions, batch, hidden_size]
        x_melody = x_melody.transpose_(0, 1).contiguous() # [batch, 1 * 2, hidden_size]
        x_melody = x_melody.view(x_melody.size(0), -1) # [batch, 2*hidden_size]
        x_melody = self.melody_before_cat_linear(x_melody) # [batch, before_cat_dim]

        x = torch.cat([x_melody, x_accompany], dim=1) # [batch, 2*before_cat_dim]
        x = self.linear_layers(x) # (batch, 1)
        return x

    def forward(self, x_accompany, x_melody):
        self.melody_gru.flatten_parameters()
        self.enc_notes_gru.flatten_parameters()
        self.enc_time_gru.flatten_parameters()

        lengths = self.get_len_index_tensor(x_accompany)
        x_accompany = self.index_tensor_to_multihot_tensor(x_accompany)
        x = self.encoder(x_accompany, lengths, x_melody)
        
        return x