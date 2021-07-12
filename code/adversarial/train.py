import os
import sys
import json
import socket
import getpass
import glog as log
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import time

p = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
if p not in sys.path:
    sys.path.append(p)
from evaluation_model.model import PianoTree_evaluation
from utils import set_log_file, get_EC2_VAE, get_poly_model, critic_accuracy, G_output_demo, setup_seed, AverageMeter, eval_z, z_distance_c
from adversarial.model import Actor, Critic
from data_loader import ZDataset

def train_epoch(args, train_loader, actor_model, critic_model, actor_optimizer, critic_optimizer, step, average_scale_melody, average_scale_accompany, writer, EC2_model, poly_model):
    actor_model.train()
    critic_model.train()

    melody_pitch_zdim = EC2_model.z1_dims
    accompany_chord_zdim = poly_model.chd_encoder.z_dim

    num_batch = len(train_loader)
    batch_start_time = time.time()
    batch_secs = 0
    for batch_idx, (batch) in enumerate(train_loader):
        (z_match_melody, melody_target_match, name_match_melody), (z_match_accompany, piano_grid_match, name_match_accompany), \
                (z_nmatch_melody, melody_target_nmatch, name_nmatch_melody), (z_nmatch_accompany, piano_grid_nmatch, name_nmatch_accompany) = batch

        z_match_melody = z_match_melody.cuda()
        z_match_accompany = z_match_accompany.cuda()
        z_nmatch_melody = z_nmatch_melody.cuda()
        z_nmatch_accompany = z_nmatch_accompany.cuda()

        if step % args['train_G2D_times'] == 0 and (step>=args['train_D_start']):
            critic_optimizer.zero_grad()
            
            if args['prior_sample']:
                z_prior_melody = torch.randn_like(z_match_melody, device='cuda')
                z_prior_accompany = torch.randn_like(z_match_accompany, device='cuda')
            
            # train critic_model by sampling from p(z) at a rate 10 times less than actor_model(p(z))
            if  np.random.rand() < 1.0/(args['pure2G_times']+1):
                G_suffix = 'woG'
                if args['prior_sample']:
                    inputz_D_melody = torch.cat([z_match_melody, z_nmatch_melody, z_prior_melody], dim=0)
                    inputz_D_accompany = torch.cat([z_match_accompany, z_nmatch_accompany, z_prior_accompany], dim=0)
                else:
                    inputz_D_melody = torch.cat([z_match_melody, z_nmatch_melody], dim=0)
                    inputz_D_accompany = torch.cat([z_match_accompany, z_nmatch_accompany], dim=0)
                G_nmatch_start_ind = 0
            else:
                G_suffix = 'wG'
                if args['zmatch2G']:
                    inputz_G_melody = torch.cat([z_match_melody, z_nmatch_melody], dim=0)
                    inputz_G_accompany = torch.cat([z_match_accompany, z_nmatch_accompany], dim=0)
                    G_nmatch_start_ind = z_match_melody.shape[0]
                else:
                    inputz_G_melody = z_nmatch_melody
                    inputz_G_accompany = z_nmatch_accompany
                    G_nmatch_start_ind = 0

                if args['prior_sample']:
                    inputz_G_melody = torch.cat([inputz_G_melody, z_prior_melody], dim=0)
                    inputz_G_accompany = torch.cat([inputz_G_accompany, z_prior_accompany], dim=0)

                if len(args['actor_c_inputs'])>0:
                    c_mp = torch.rand([inputz_G_melody.shape[0], 1], device="cuda")
                    c_mr = torch.rand([inputz_G_melody.shape[0], 1], device="cuda")
                    c_ac = torch.rand([inputz_G_melody.shape[0], 1], device="cuda")
                    c_at = torch.rand([inputz_G_melody.shape[0], 1], device="cuda")
                else:
                    c_mp = c_mr = c_ac = c_at = None

                outputz_G_melody, outputz_G_accompany = actor_model(inputz_G_melody, inputz_G_accompany, c_mp, c_mr, c_ac, c_at)
                outputz_G_melody = outputz_G_melody.detach()
                outputz_G_accompany = outputz_G_accompany.detach()

                inputz_D_melody = torch.cat([z_match_melody, outputz_G_melody], dim=0)
                inputz_D_accompany = torch.cat([z_match_accompany, outputz_G_accompany], dim=0)

            label_D = torch.zeros([inputz_D_melody.shape[0], 1], device="cuda")
            label_D[:z_match_melody.shape[0], :]=1
            
            output_D = critic_model(inputz_D_melody, inputz_D_accompany)

            loss_D_all = F.binary_cross_entropy(output_D, label_D, reduction='none')
            loss_D_zmatch_all = loss_D_all[:z_match_melody.shape[0]+G_nmatch_start_ind].mean()*args['loss_D_zmatch_k']
            loss_D_others = loss_D_all[z_match_melody.shape[0]+G_nmatch_start_ind:].mean()
            loss_D = loss_D_zmatch_all + loss_D_others
            loss_D.backward()
            critic_optimizer.step()

            if (step % args['print_every']==0) or (batch_idx==0) or (batch_idx==len(train_loader)-1):
                loss_D_zmatch = loss_D_all[:z_match_melody.shape[0]].mean()
                loss_D_znmatch = loss_D_all[z_match_melody.shape[0]+G_nmatch_start_ind: z_match_melody.shape[0]+G_nmatch_start_ind+z_nmatch_melody.shape[0]].mean()
                
                D_accuracy, D_batch_equal = critic_accuracy(D_output=output_D, label=label_D)
                D_match_accuracy = D_batch_equal[:z_match_melody.shape[0]].mean()
                D_nmatch_accuracy = D_batch_equal[z_match_melody.shape[0]+G_nmatch_start_ind: z_match_melody.shape[0]+G_nmatch_start_ind+z_nmatch_melody.shape[0]].mean()

                if G_nmatch_start_ind>0:
                    loss_D_Gzmatch = loss_D_all[z_match_melody.shape[0]:z_match_melody.shape[0]+G_nmatch_start_ind].mean()
                    D_Gmatch_accuracy = D_batch_equal[z_match_melody.shape[0]:z_match_melody.shape[0]+G_nmatch_start_ind].mean()
                    writer.add_scalar('train_D_%s/loss_D_Gzmatch'%G_suffix, loss_D_Gzmatch.item(), step)
                    writer.add_scalar('train_D_%s/D_Gmatch_accuracy'%G_suffix, D_Gmatch_accuracy*100, step)

                if args['prior_sample']:
                    loss_D_zprior = loss_D_all[z_match_melody.shape[0]+G_nmatch_start_ind+z_nmatch_melody.shape[0]:].mean()
                    D_prior_accuray = D_batch_equal[z_match_melody.shape[0]+G_nmatch_start_ind+z_nmatch_melody.shape[0]:].mean()

                    writer.add_scalar('train_D_%s/loss_D_zprior'%G_suffix, loss_D_zprior.item(), step)
                    writer.add_scalar('train_D_%s/D_prior_accuray'%G_suffix, D_prior_accuray*100, step)

                writer.add_scalar('train_D_%s/loss_D'%G_suffix, loss_D.item(), step)
                writer.add_scalar('train_D_%s/loss_D_zmatch'%G_suffix, loss_D_zmatch.item(), step)
                writer.add_scalar('train_D_%s/loss_D_znmatch'%G_suffix, loss_D_znmatch.item(), step)
                writer.add_scalar('train_D_%s/D_accuracy'%G_suffix, D_accuracy*100, step)
                writer.add_scalar('train_D_%s/D_match_accuracy'%G_suffix, D_match_accuracy*100, step)
                writer.add_scalar('train_D_%s/D_nmatch_accuracy'%G_suffix, D_nmatch_accuracy*100, step)

                output_str = 'Train D {:s}: {:d}/{:d}, epoch i: {:d}/{:d}, Time: {:.2f}s, D_acc: {:.2f}, D_m_acc: {:.2f}, D_nm_acc: {:.2f}{:s}, loss_D: {:.5f}, loss_D_zmatch: {:.5f}, loss_D_znmatch: {:.5f}{:s}'
                if args['prior_sample']:
                    D_prior_acc_str = ', D_prior_acc: {:.2f}'.format(D_prior_accuray*100)
                    loss_D_zprior_str = ', loss_D_zprior: {:.5f}'.format(loss_D_zprior.item())
                else:
                    D_prior_acc_str = loss_D_zprior_str = ' '
                log.info(output_str.format(G_suffix, step, args['iteration_num'], batch_idx, num_batch, batch_secs, D_accuracy*100, D_match_accuracy*100,
                    D_nmatch_accuracy*100, D_prior_acc_str, loss_D.item(), loss_D_zmatch.item(), loss_D_znmatch.item(), loss_D_zprior_str))

        # training D and G in a 10:1 step ratio
        if ((step%args['train_D2G_times']==0) or (step<args['train_D_start'])) and args['train_G']:
            actor_optimizer.zero_grad()

            if args['Gprior_sample']:
                z_prior_melody = torch.randn_like(z_nmatch_melody, device='cuda')
                z_prior_accompany = torch.randn_like(z_nmatch_accompany, device='cuda')

                inputz_G_melody = torch.cat([z_nmatch_melody, z_prior_melody], dim=0)
                inputz_G_accompany = torch.cat([z_nmatch_accompany, z_prior_accompany], dim=0)
            else:
                inputz_G_melody = z_nmatch_melody
                inputz_G_accompany = z_nmatch_accompany

            if len(args['actor_c_inputs'])>0:
                c_mp = torch.rand([inputz_G_melody.shape[0], 1], device="cuda")
                c_mr = torch.rand([inputz_G_melody.shape[0], 1], device="cuda")
                c_ac = torch.rand([inputz_G_melody.shape[0], 1], device="cuda")
                c_at = torch.rand([inputz_G_melody.shape[0], 1], device="cuda")
            else:
                c_mp = torch.ones([inputz_G_melody.shape[0], 1], device="cuda")*0.5
                c_mr = torch.ones([inputz_G_melody.shape[0], 1], device="cuda")*0.5
                c_ac = torch.ones([inputz_G_melody.shape[0], 1], device="cuda")*0.5
                c_at = torch.ones([inputz_G_melody.shape[0], 1], device="cuda")*0.5

            outputz_G_melody, outputz_G_accompany = actor_model(inputz_G_melody, inputz_G_accompany, c_mp, c_mr, c_ac, c_at)
            output_D_Gz = critic_model(outputz_G_melody, outputz_G_accompany)
            fool_label = torch.ones([output_D_Gz.shape[0], 1], device="cuda")

            loss_G_all = F.binary_cross_entropy(output_D_Gz, fool_label, reduction='none')
            loss_G_z = loss_G_all.mean()

            melody_distance = (outputz_G_melody-inputz_G_melody).pow(2)
            melody_pitch_distance = melody_distance[:, :melody_pitch_zdim]
            melody_rhythm_distance = melody_distance[:, melody_pitch_zdim:]
            accompany_distance = (outputz_G_accompany-inputz_G_accompany).pow(2)
            accompany_chord_distance = accompany_distance[:, :accompany_chord_zdim]
            accompany_texture_distance = accompany_distance[:, accompany_chord_zdim:]

            average_scale_melody_pow2 = average_scale_melody.pow(-2)
            melody_pitch_distance_mean = (melody_pitch_distance*average_scale_melody_pow2[:melody_pitch_zdim]).mean()
            melody_rhythm_distance_mean = (melody_rhythm_distance*average_scale_melody_pow2[melody_pitch_zdim:]).mean()
            
            average_scale_accompany_pow2 = average_scale_accompany.pow(-2)
            accompany_chord_distance_mean = (accompany_chord_distance*average_scale_accompany_pow2[:accompany_chord_zdim]).mean()
            accompany_texture_distance_mean = (accompany_texture_distance*average_scale_accompany_pow2[accompany_chord_zdim:]).mean()

            loss_distance_z_melody_pitch = torch.mean((1 + melody_pitch_distance).log()*(average_scale_melody_pow2[:melody_pitch_zdim]), dim=1)
            loss_distance_z_melody_rhythm = torch.mean((1 + melody_rhythm_distance).log()*(average_scale_melody_pow2[melody_pitch_zdim:]), dim=1)
            loss_distance_z_accompany_chord = torch.mean((1 + accompany_chord_distance).log()*(average_scale_accompany_pow2[:accompany_chord_zdim]), dim=1)
            loss_distance_z_accompany_texture = torch.mean((1 + accompany_texture_distance).log()*(average_scale_accompany_pow2[accompany_chord_zdim:]), dim=1)

            c_mp = c_mp.squeeze(-1)
            c_mr = c_mr.squeeze(-1)
            c_ac = c_ac.squeeze(-1)
            c_at = c_at.squeeze(-1)
            loss_distance_z_melody = (loss_distance_z_melody_pitch*c_mp + loss_distance_z_melody_rhythm*c_mr).mean()
            loss_distance_z_accompany = (loss_distance_z_accompany_chord*c_ac + loss_distance_z_accompany_texture*c_at).mean()
            loss_distance = args['distance_lambda']*2*(loss_distance_z_melody*args['loss_melody_distance_k'] + loss_distance_z_accompany)/(1+args['loss_melody_distance_k'])
            
            loss_G = loss_G_z + loss_distance
            loss_G.backward()
            actor_optimizer.step()

            if (step % args['print_every']==0) or (batch_idx==0) or (batch_idx==len(train_loader)-1):
                D_fooled_accuracy, D_fooled_batch_equal = critic_accuracy(D_output=output_D_Gz, label=fool_label)
                D_fooled_nmatch_accuracy = D_fooled_batch_equal[:z_nmatch_melody.shape[0]].mean()

                loss_G_znmatch = loss_G_all[:z_nmatch_melody.shape[0]].mean()
                if args['Gprior_sample']:
                    loss_G_zprior = loss_G_all[z_nmatch_melody.shape[0]:].mean()
                    D_fooled_prior_accuracy = D_fooled_batch_equal[z_nmatch_melody.shape[0]:].mean()
                    writer.add_scalar('train_G/loss_G_zprior', loss_G_zprior.item(), step)
                    writer.add_scalar('train_G/D_fooled_prior_accuracy', D_fooled_prior_accuracy*100, step)
                writer.add_scalar('train_G/loss_G', loss_G.item(), step)
                writer.add_scalar('train_G/loss_G_znmatch', loss_G_znmatch.item(), step)
                writer.add_scalar('train_G/loss_distance', loss_distance.item(), step)
                writer.add_scalar('train_G/loss_distance_z_melody', loss_distance_z_melody.item(), step)
                writer.add_scalar('train_G/loss_distance_z_accompany', loss_distance_z_accompany.item(), step)
                writer.add_scalar('train_G/distance_melody_pitch_mean', melody_pitch_distance_mean, step)
                writer.add_scalar('train_G/distance_melody_rhythm_mean', melody_rhythm_distance_mean, step)
                writer.add_scalar('train_G/distance_accompany_chord_mean', accompany_chord_distance_mean, step)
                writer.add_scalar('train_G/distance_accompany_texture_mean', accompany_texture_distance_mean, step)
                writer.add_scalar('train_G/D_fooled_accuracy', D_fooled_accuracy*100, step)
                writer.add_scalar('train_G/D_fooled_nmatch_accuracy', D_fooled_nmatch_accuracy*100, step)
                
                output_str = 'Train G: {:d}/{:d}, epoch i: {:d}/{:d}, Time: {:.2f}s, D_f_acc: {:.2f}, D_f_nm_acc: {:.2f}{:s}, loss_G: {:.5f}, loss_G_znmatch: {:.5f}{:s}, loss_distance: {:.5f}, melody_pitch_dist: {:.2e}, melody_rhythm_dist: {:.2e}, accompany_chord_dist: {:.2e}, accompany_texture_dist: {:.2e}'
                if args['Gprior_sample']:
                    D_f_prior_acc_str = ', D_f_prior_acc: {:.2f}'.format(D_fooled_prior_accuracy*100)
                    loss_G_zprior_str = ', loss_G_zprior: {:.5f}'.format(loss_G_zprior.item())
                else:
                    D_f_prior_acc_str = loss_G_zprior_str = ''

                log.info(output_str.format( step, args['iteration_num'], batch_idx, num_batch, batch_secs, D_fooled_accuracy*100, D_fooled_nmatch_accuracy*100, D_f_prior_acc_str,
                    loss_G.item(), loss_G_znmatch.item(), loss_G_zprior_str, loss_distance.item(), 
                    melody_pitch_distance_mean, melody_rhythm_distance_mean, accompany_chord_distance_mean, accompany_texture_distance_mean))

        step += 1
        batch_secs = time.time() - batch_start_time
        batch_start_time = time.time()
    
    return step

def eval_epoch(epoch, val_loader, actor_model, critic_model, eval_model, EC2_model, poly_model, step, average_scale_melody, average_scale_accompany, writer):
    actor_model.eval()
    critic_model.eval()

    melody_pitch_zdim = EC2_model.z1_dims
    accompany_chord_zdim = poly_model.chd_encoder.z_dim

    loss_D_avg = AverageMeter()
    loss_D_zmatch_avg = AverageMeter()
    loss_D_znmatch_avg = AverageMeter()
    loss_D_zprior_avg = AverageMeter()

    D_accuracy_avg = AverageMeter()
    D_match_accuracy_avg = AverageMeter()
    D_nmatch_accuracy_avg = AverageMeter()
    D_prior_accuray_avg = AverageMeter()

    loss_G_avg = AverageMeter()
    loss_G_z_avg = AverageMeter()
    loss_distance_avg = AverageMeter()
    loss_distance_z_melody_avg = AverageMeter()
    loss_distance_z_accompany_avg = AverageMeter()
    loss_G_znmatch_avg = AverageMeter()
    loss_G_zprior_avg = AverageMeter()
    loss_D_Gzmatch_avg = AverageMeter()

    D_fooled_accuracy_avg = AverageMeter()
    D_fooled_nmatch_accuracy_avg = AverageMeter()
    D_fooled_prior_accuracy_avg = AverageMeter()
    D_Gzmatch_accuracy_avg = AverageMeter()
    eval_beforeG_nmatch_acc_avg = AverageMeter()
    eval_beforeG_prior_acc_avg = AverageMeter()
    eval_afterG_nmatch_acc_avg = AverageMeter()
    eval_afterG_prior_acc_avg = AverageMeter()

    melody_pitch_distance_mean_avg = AverageMeter()
    melody_rhythm_distance_mean_avg = AverageMeter()
    accompany_chord_distance_mean_avg = AverageMeter()
    accompany_texture_distance_mean_avg = AverageMeter()

    start_time = time.time()
    for batch_idx, (batch) in enumerate(val_loader):
        assert batch_idx==0
        (z_match_melody, melody_target_match, name_match_melody), (z_match_accompany, piano_grid_match, name_match_accompany), \
                (z_nmatch_melody, melody_target_nmatch, name_nmatch_melody), (z_nmatch_accompany, piano_grid_nmatch, name_nmatch_accompany) = batch

        z_match_melody = z_match_melody.cuda()
        z_match_accompany = z_match_accompany.cuda()
        z_nmatch_melody = z_nmatch_melody.cuda()
        z_nmatch_accompany = z_nmatch_accompany.cuda()

        inputz_D_melody = torch.cat([z_match_melody, z_nmatch_melody], dim=0)
        inputz_D_accompany = torch.cat([z_match_accompany, z_nmatch_accompany], dim=0)
        if args['prior_sample']:
            z_prior_melody = torch.randn_like(z_nmatch_melody, device='cuda')
            z_prior_accompany = torch.randn_like(z_nmatch_accompany, device='cuda')
            inputz_D_melody = torch.cat([inputz_D_melody, z_prior_melody], dim=0)
            inputz_D_accompany = torch.cat([inputz_D_accompany, z_prior_accompany], dim=0)

        label_D = torch.zeros([inputz_D_melody.shape[0], 1], device="cuda")
        label_D[:z_match_melody.shape[0], :]=1
        
        output_D = critic_model(inputz_D_melody, inputz_D_accompany)

        loss_D_all = F.binary_cross_entropy(output_D, label_D, reduction='none')
        loss_D = loss_D_all.sum()

        D_accuracy, D_batch_equal = critic_accuracy(D_output=output_D, label=label_D)
        D_match_accuracy = D_batch_equal[:z_match_melody.shape[0]].sum()
        D_nmatch_accuracy = D_batch_equal[z_match_melody.shape[0]: z_match_melody.shape[0]+z_nmatch_melody.shape[0]].sum()
        loss_D_zmatch = loss_D_all[:z_match_melody.shape[0]].sum()
        loss_D_znmatch = loss_D_all[z_match_melody.shape[0]: z_match_melody.shape[0]+z_nmatch_melody.shape[0]].sum()

        loss_D_avg.update(loss_D.item(), output_D.shape[0])
        loss_D_zmatch_avg.update(loss_D_zmatch.item(), z_match_melody.shape[0])
        loss_D_znmatch_avg.update(loss_D_znmatch.item(), z_nmatch_melody.shape[0])
        D_accuracy_avg.update(D_accuracy*output_D.shape[0]*100, output_D.shape[0])
        D_match_accuracy_avg.update(D_match_accuracy*100, z_match_melody.shape[0])
        D_nmatch_accuracy_avg.update(D_nmatch_accuracy*100, z_nmatch_melody.shape[0])

        if args['prior_sample']:
            loss_D_zprior = loss_D_all[z_match_melody.shape[0]+z_nmatch_melody.shape[0]:].sum()
            D_prior_accuray = D_batch_equal[z_match_melody.shape[0]+z_nmatch_melody.shape[0]:].sum()

            loss_D_zprior_avg.update(loss_D_zprior.item(), z_prior_melody.shape[0])
            D_prior_accuray_avg.update(D_prior_accuray*100, z_prior_melody.shape[0])

        if args['train_G']:
            if args['zmatch2G']:
                inputz_G_melody = torch.cat([z_match_melody, z_nmatch_melody], dim=0)
                inputz_G_accompany = torch.cat([z_match_accompany, z_nmatch_accompany], dim=0)
                G_nmatch_start_ind = z_match_melody.shape[0]
            else:
                inputz_G_melody = z_nmatch_melody
                inputz_G_accompany = z_nmatch_accompany
                G_nmatch_start_ind = 0

            if args['Gprior_sample']:
                G_z_prior_melody = torch.randn_like(z_nmatch_melody, device='cuda')
                G_z_prior_accompany = torch.randn_like(z_nmatch_accompany, device='cuda')
                inputz_G_melody = torch.cat([inputz_G_melody, G_z_prior_melody], dim=0)
                inputz_G_accompany = torch.cat([inputz_G_accompany, G_z_prior_accompany], dim=0)

            if len(args['actor_c_inputs'])>0:
                c_mp = torch.rand([inputz_G_melody.shape[0], 1], device="cuda")
                c_mr = torch.rand([inputz_G_melody.shape[0], 1], device="cuda")
                c_ac = torch.rand([inputz_G_melody.shape[0], 1], device="cuda")
                c_at = torch.rand([inputz_G_melody.shape[0], 1], device="cuda")
            else:
                c_mp = torch.ones([inputz_G_melody.shape[0], 1], device="cuda")*0.5
                c_mr = torch.ones([inputz_G_melody.shape[0], 1], device="cuda")*0.5
                c_ac = torch.ones([inputz_G_melody.shape[0], 1], device="cuda")*0.5
                c_at = torch.ones([inputz_G_melody.shape[0], 1], device="cuda")*0.5

            outputz_G_melody, outputz_G_accompany = actor_model(inputz_G_melody, inputz_G_accompany, c_mp, c_mr, c_ac, c_at)

            torch.cuda.empty_cache()
            inputzG_eval_score = eval_z(eval_model, EC2_model, poly_model, z_melody=inputz_G_melody, z_accompany=inputz_G_accompany, batch_size=1650)
            inputzG_eval_score = inputzG_eval_score.cpu()
            torch.cuda.empty_cache()
            _, eval_inputzG_acc_batch = critic_accuracy(D_output=inputzG_eval_score, label = torch.ones([inputzG_eval_score.shape[0], 1]))

            outputzG_eval_score = eval_z(eval_model, EC2_model, poly_model, z_melody=outputz_G_melody, z_accompany=outputz_G_accompany, batch_size=1650)
            outputzG_eval_score = outputzG_eval_score.cpu()
            torch.cuda.empty_cache()
            _, eval_outputzG_acc_batch = critic_accuracy(D_output=outputzG_eval_score, label = torch.ones([outputzG_eval_score.shape[0], 1]))

            output_D_Gz = critic_model(outputz_G_melody, outputz_G_accompany)
            fool_label = torch.ones([output_D_Gz.shape[0], 1], device="cuda")

            loss_G_all = F.binary_cross_entropy(output_D_Gz, fool_label, reduction='none')
            loss_G_z = loss_G_all[G_nmatch_start_ind:].mean()

            melody_distance = (outputz_G_melody[G_nmatch_start_ind:]-inputz_G_melody[G_nmatch_start_ind:]).pow(2)
            melody_pitch_distance = melody_distance[:, :melody_pitch_zdim]
            melody_rhythm_distance = melody_distance[:, melody_pitch_zdim:]
            accompany_distance = (outputz_G_accompany[G_nmatch_start_ind:]-inputz_G_accompany[G_nmatch_start_ind:]).pow(2)
            accompany_chord_distance = accompany_distance[:, :accompany_chord_zdim]
            accompany_texture_distance = accompany_distance[:, accompany_chord_zdim:]

            average_scale_melody_pow2 = average_scale_melody.pow(-2)
            melody_pitch_distance_batch = (melody_pitch_distance*average_scale_melody_pow2[:melody_pitch_zdim]).mean(1)
            melody_rhythm_distance_batch = (melody_rhythm_distance*average_scale_melody_pow2[melody_pitch_zdim:]).mean(1)
            melody_pitch_distance_mean = melody_pitch_distance_batch.mean()
            melody_rhythm_distance_mean = melody_rhythm_distance_batch.mean()
            
            average_scale_accompany_pow2 = average_scale_accompany.pow(-2)
            accompany_chord_distance_batch = (accompany_chord_distance*average_scale_accompany_pow2[:accompany_chord_zdim]).mean(1)
            accompany_texture_distance_batch = (accompany_texture_distance*average_scale_accompany_pow2[accompany_chord_zdim:]).mean(1)
            accompany_chord_distance_mean = accompany_chord_distance_batch.mean()
            accompany_texture_distance_mean = accompany_texture_distance_batch.mean()

            loss_distance_z_melody_pitch = torch.mean((1 + melody_pitch_distance).log()*(average_scale_melody_pow2[:melody_pitch_zdim]), dim=1)
            loss_distance_z_melody_rhythm = torch.mean((1 + melody_rhythm_distance).log()*(average_scale_melody_pow2[melody_pitch_zdim:]), dim=1)
            loss_distance_z_accompany_chord = torch.mean((1 + accompany_chord_distance).log()*(average_scale_accompany_pow2[:accompany_chord_zdim]), dim=1)
            loss_distance_z_accompany_texture = torch.mean((1 + accompany_texture_distance).log()*(average_scale_accompany_pow2[accompany_chord_zdim:]), dim=1)

            c_mp = c_mp.squeeze(-1)
            c_mr = c_mr.squeeze(-1)
            c_ac = c_ac.squeeze(-1)
            c_at = c_at.squeeze(-1)
            loss_distance_z_melody = (loss_distance_z_melody_pitch*c_mp + loss_distance_z_melody_rhythm*c_mr).mean()
            loss_distance_z_accompany = (loss_distance_z_accompany_chord*c_ac + loss_distance_z_accompany_texture*c_at).mean()
            loss_distance = args['distance_lambda']*2*(loss_distance_z_melody*args['loss_melody_distance_k'] + loss_distance_z_accompany)/(1+args['loss_melody_distance_k'])

            loss_G = loss_G_z + loss_distance

            _, D_fooled_batch_equal = critic_accuracy(D_output=output_D_Gz, label=fool_label)
            D_fooled_accuracy = D_fooled_batch_equal[G_nmatch_start_ind:].mean()

            loss_G_znmatch = loss_G_all[G_nmatch_start_ind:G_nmatch_start_ind+z_nmatch_melody.shape[0]].mean()
            D_fooled_nmatch_accuracy = D_fooled_batch_equal[G_nmatch_start_ind:G_nmatch_start_ind+z_nmatch_melody.shape[0]].mean()

            eval_beforeG_nmatch_acc = eval_inputzG_acc_batch[G_nmatch_start_ind:G_nmatch_start_ind+z_nmatch_melody.shape[0]].mean()
            eval_afterG_nmatch_acc = eval_outputzG_acc_batch[G_nmatch_start_ind:G_nmatch_start_ind+z_nmatch_melody.shape[0]].mean()

            if G_nmatch_start_ind>0:
                loss_D_Gzmatch = loss_G_all[0:G_nmatch_start_ind].mean()
                D_Gmatch_accuracy = D_fooled_batch_equal[0:G_nmatch_start_ind].mean()
                
                loss_D_Gzmatch_avg.update(loss_D_Gzmatch.item()*G_nmatch_start_ind, G_nmatch_start_ind)
                D_Gzmatch_accuracy_avg.update(D_Gmatch_accuracy*100*G_nmatch_start_ind, G_nmatch_start_ind)
            
            if args['Gprior_sample']:
                loss_G_zprior = loss_G_all[G_nmatch_start_ind+z_nmatch_melody.shape[0]:].mean()
                D_fooled_prior_accuracy = D_fooled_batch_equal[G_nmatch_start_ind+z_nmatch_melody.shape[0]:].mean()
                eval_beforeG_prior_acc = eval_inputzG_acc_batch[G_nmatch_start_ind+z_nmatch_melody.shape[0]:].mean()
                eval_afterG_prior_acc = eval_outputzG_acc_batch[G_nmatch_start_ind+z_nmatch_melody.shape[0]:].mean()

                loss_G_zprior_avg.update(loss_G_zprior.item()*G_z_prior_melody.shape[0], G_z_prior_melody.shape[0])
                D_fooled_prior_accuracy_avg.update(D_fooled_prior_accuracy*100*G_z_prior_melody.shape[0], G_z_prior_melody.shape[0])
                eval_beforeG_prior_acc_avg.update(eval_beforeG_prior_acc*100*G_z_prior_melody.shape[0], G_z_prior_melody.shape[0])
                eval_afterG_prior_acc_avg.update(eval_afterG_prior_acc*100*G_z_prior_melody.shape[0], G_z_prior_melody.shape[0])
            
            loss_G_avg.update(loss_G.item()*(loss_G_all.shape[0]-G_nmatch_start_ind), loss_G_all.shape[0]-G_nmatch_start_ind)
            loss_G_z_avg.update(loss_G_z.item()*(loss_G_all.shape[0]-G_nmatch_start_ind), loss_G_all.shape[0]-G_nmatch_start_ind)
            loss_distance_avg.update(loss_distance.item()*(loss_G_all.shape[0]-G_nmatch_start_ind), loss_G_all.shape[0]-G_nmatch_start_ind)
            loss_distance_z_melody_avg.update(loss_distance_z_melody.item()*(loss_G_all.shape[0]-G_nmatch_start_ind), loss_G_all.shape[0]-G_nmatch_start_ind)
            loss_distance_z_accompany_avg.update(loss_distance_z_accompany.item()*(loss_G_all.shape[0]-G_nmatch_start_ind), loss_G_all.shape[0]-G_nmatch_start_ind)
            loss_G_znmatch_avg.update(loss_G_znmatch.item()*z_nmatch_melody.shape[0], z_nmatch_melody.shape[0])
            
            D_fooled_accuracy_avg.update(D_fooled_accuracy*100*(loss_G_all.shape[0]-G_nmatch_start_ind), (loss_G_all.shape[0]-G_nmatch_start_ind))
            D_fooled_nmatch_accuracy_avg.update(D_fooled_nmatch_accuracy*100*z_nmatch_melody.shape[0], z_nmatch_melody.shape[0])
            eval_beforeG_nmatch_acc_avg.update(eval_beforeG_nmatch_acc*100*z_nmatch_melody.shape[0], z_nmatch_melody.shape[0])
            eval_afterG_nmatch_acc_avg.update(eval_afterG_nmatch_acc*100*z_nmatch_melody.shape[0], z_nmatch_melody.shape[0])
            
            melody_pitch_distance_mean_avg.update(melody_pitch_distance_mean*(loss_G_all.shape[0]-G_nmatch_start_ind), loss_G_all.shape[0]-G_nmatch_start_ind)
            melody_rhythm_distance_mean_avg.update(melody_rhythm_distance_mean*(loss_G_all.shape[0]-G_nmatch_start_ind), loss_G_all.shape[0]-G_nmatch_start_ind)
            accompany_chord_distance_mean_avg.update(accompany_chord_distance_mean*(loss_G_all.shape[0]-G_nmatch_start_ind), loss_G_all.shape[0]-G_nmatch_start_ind)
            accompany_texture_distance_mean_avg.update(accompany_texture_distance_mean*(loss_G_all.shape[0]-G_nmatch_start_ind), loss_G_all.shape[0]-G_nmatch_start_ind)

    if args['prior_sample']:
        writer.add_scalar('test_D/loss_D_zprior', loss_D_zprior_avg.avg, step)
        writer.add_scalar('test_D/D_prior_accuray', D_prior_accuray_avg.avg, step)

    writer.add_scalar('test_D/loss_D', loss_D_avg.avg, step)
    writer.add_scalar('test_D/loss_D_zmatch', loss_D_zmatch_avg.avg, step)
    writer.add_scalar('test_D/loss_D_znmatch', loss_D_znmatch_avg.avg, step)
    writer.add_scalar('test_D/D_accuracy', D_accuracy_avg.avg, step)
    writer.add_scalar('test_D/D_match_accuracy', D_match_accuracy_avg.avg, step)
    writer.add_scalar('test_D/D_nmatch_accuracy', D_nmatch_accuracy_avg.avg, step)

    if args['train_G'] and args['zmatch2G']:
        writer.add_scalar('test_D/loss_D_Gzmatch', loss_D_Gzmatch_avg.avg, step)
        writer.add_scalar('test_D/D_Gzmatch_accuracy_avg', D_Gzmatch_accuracy_avg.avg, step)
    
    if args['train_G']:
        writer.add_scalar('test_G/loss_G', loss_G_avg.avg, step)
        writer.add_scalar('test_G/loss_G_z', loss_G_z_avg.avg, step)
        writer.add_scalar('test_G/loss_distance', loss_distance_avg.avg, step)
        writer.add_scalar('test_G/loss_distance_z_melody', loss_distance_z_melody_avg.avg, step)
        writer.add_scalar('test_G/loss_distance_z_accompany', loss_distance_z_accompany_avg.avg, step)
        writer.add_scalar('test_G/distance_melody_pitch_mean', melody_pitch_distance_mean_avg.avg, step)
        writer.add_scalar('test_G/distance_melody_rhythm_mean', melody_rhythm_distance_mean_avg.avg, step)
        writer.add_scalar('test_G/distance_accompany_chord_mean', accompany_chord_distance_mean_avg.avg, step)
        writer.add_scalar('test_G/distance_accompany_texture_mean', accompany_texture_distance_mean_avg.avg, step)
        writer.add_scalar('test_G/loss_G_znmatch', loss_G_znmatch_avg.avg, step)    
        writer.add_scalar('test_G/D_fooled_accuracy', D_fooled_accuracy_avg.avg, step)
        writer.add_scalar('test_G/D_fooled_nmatch_accuracy', D_fooled_nmatch_accuracy_avg.avg, step)
        writer.add_scalar('test_G/eval_beforeG_nmatch_acc_avg', eval_beforeG_nmatch_acc_avg.avg, step)
        writer.add_scalar('test_G/eval_afterG_nmatch_acc_avg', eval_afterG_nmatch_acc_avg.avg, step)

        if args['Gprior_sample']:
            writer.add_scalar('test_G/loss_G_zprior', loss_G_zprior_avg.avg, step)
            writer.add_scalar('test_G/D_fooled_prior_accuracy', D_fooled_prior_accuracy_avg.avg, step)
            writer.add_scalar('test_G/eval_beforeG_prior_acc_avg', eval_beforeG_prior_acc_avg.avg, step)
            writer.add_scalar('test_G/eval_afterG_prior_acc_avg', eval_afterG_prior_acc_avg.avg, step)

    output_str = 'Test D: {:d}/{:d}, epoch: {:d}, Time: {:.2f}s, D_acc: {:.2f}, D_m_acc: {:.2f}{:s}, D_nm_acc: {:.2f}{:s}, loss_D: {:.5f}, loss_D_zmatch: {:.5f}{:s}, loss_D_znmatch: {:.5f}{:s}'
    if args['prior_sample']:
        D_prior_acc_str = ', D_prior_acc: {:.2f}'.format(D_prior_accuray_avg.avg)
        loss_D_zprior_str = ', loss_D_zprior: {:.5f}'.format(loss_D_zprior_avg.avg)
    else:
        D_prior_acc_str = loss_D_zprior_str = ''

    if args['train_G'] and args['zmatch2G']:
        D_Gzmatch_acc_str = ', D_Gmatch_acc: {:.2f}'.format(D_Gzmatch_accuracy_avg.avg)
        loss_D_Gzmatch_str = ', loss_D_Gzmatch: {:.5f}'.format(loss_D_Gzmatch_avg.avg)
    else:
        D_Gzmatch_acc_str = loss_D_Gzmatch_str = ''

    log.info(output_str.format(step, args['iteration_num'], epoch, time.time()-start_time, D_accuracy_avg.avg, D_match_accuracy_avg.avg, D_Gzmatch_acc_str,
            D_nmatch_accuracy_avg.avg, D_prior_acc_str, loss_D_avg.avg, loss_D_zmatch_avg.avg, loss_D_Gzmatch_str, loss_D_znmatch_avg.avg, loss_D_zprior_str))

    if args['train_G']:
        output_str = 'Test G: {:d}/{:d}, epoch: {:d}, avg_acc: {:.2f}, evalG_nmatch_d_acc: {:.2f}{:s}, D_f_acc: {:.2f}, D_f_nm_acc: {:.2f}{:s}, loss_G: {:.5f}, loss_G_znmatch: {:.5f}{:s}, loss_distance: {:.5f}, melody_pitch_dist: {:.2e}, melody_rhythm_dist: {:.2e}, accompany_chord_dist: {:.2e}, accompany_texture_dist: {:.2e}'
        eval_deltaG_nmatch_acc = eval_afterG_nmatch_acc_avg.avg-eval_beforeG_nmatch_acc_avg.avg

        if args['Gprior_sample']:
            D_f_prior_acc_str = ', D_f_prior_acc: {:.2f}'.format(D_fooled_prior_accuracy_avg.avg)
            loss_G_zprior_str = ', loss_G_zprior: {:.5f}'.format(loss_G_zprior_avg.avg)
            eval_deltaG_prior_acc = eval_afterG_prior_acc_avg.avg-eval_beforeG_prior_acc_avg.avg
            eval_deltaG_prior_acc_str = ', evalG_prior_d_acc: {:.2f}'.format(eval_deltaG_prior_acc)
        else:
            D_f_prior_acc_str = loss_G_zprior_str = eval_deltaG_prior_acc_str = ''
            eval_deltaG_prior_acc = eval_deltaG_nmatch_acc

        avg_acc = (eval_deltaG_prior_acc + eval_deltaG_nmatch_acc)/2 - \
            ((melody_pitch_distance_mean_avg.avg+melody_rhythm_distance_mean_avg.avg)*args['loss_melody_distance_k']+(accompany_chord_distance_mean_avg.avg+accompany_texture_distance_mean_avg.avg))/(1+args['loss_melody_distance_k'])/4*10
    
        log.info(output_str.format(step, args['iteration_num'], epoch, avg_acc, eval_deltaG_nmatch_acc, eval_deltaG_prior_acc_str,
             D_fooled_accuracy_avg.avg, D_fooled_nmatch_accuracy_avg.avg, D_f_prior_acc_str, loss_G_avg.avg, loss_G_znmatch_avg.avg, loss_G_zprior_str,
              loss_distance_avg.avg, melody_pitch_distance_mean_avg.avg, melody_rhythm_distance_mean_avg.avg, accompany_chord_distance_mean_avg.avg, accompany_texture_distance_mean_avg.avg))
    else:
        avg_acc = D_accuracy_avg.avg

    return avg_acc

def main(args):
    setup_seed(args['random_seed'])

    PITCH_PAD = 130
    MAX_PITCH = 127
    MIN_PITCH = 0
    pitch_range = MAX_PITCH-MIN_PITCH+1+2
    train_dataset = ZDataset(melody_path=os.path.join(args['data_path'], 'train_EC2.npz'), accompany_path=os.path.join(args['data_path'], 'train_poly.npz'),
                     pitch_pad=PITCH_PAD, pitch_range=pitch_range, z_nmatch_fix=args['z_nmatch_fix'], sample=args['z_sample'], with_music=False)
    val_dataset = ZDataset(melody_path=os.path.join(args['data_path'], 'val_EC2.npz'), accompany_path=os.path.join(args['data_path'], 'val_poly.npz'),
                     pitch_pad=PITCH_PAD, pitch_range=pitch_range, z_nmatch_fix=False, sample=False, with_music=False)
    train_loader = DataLoader(train_dataset, args['batch_size'], shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, args['batch_size'], shuffle=False, drop_last=False)
    average_scale_melody, average_scale_accompany = train_dataset.get_std_average()

    EC2_model = get_EC2_VAE(args['EC2_VAE_path'])
    poly_model = get_poly_model(args['poly_model_path'])
    EC2_model.eval()
    poly_model.eval()
    EC2_model.cuda()
    poly_model.cuda()
    
    if args['D_load']:
        with open(os.path.join(args['D_model_param_path'], 'adversarial/model_config.json')) as f:
            adv_args = json.load(f)
        args['critic_d_input'] = adv_args['adversarial_d_input']
        args['critic_d_mid'] = adv_args['adversarial_d_mid']
        args['critic_layer_num'] = adv_args['adversarial_layer_num']
        args['critic_dropout_rate'] = adv_args['dropout_rate']
        args['critic_bn_flag'] = adv_args['bn_flag']
        args['critic_leaky_relu'] = adv_args['leaky_relu']

    actor_model = Actor(d_z_melody=EC2_model.z1_dims+EC2_model.z2_dims, d_z_accompany=poly_model.decoder.z_size, d_input=args['actor_d_input'], c_input_dims=args['actor_c_inputs'],
     d_mid=args['actor_d_mid'], layer_num=args['actor_layer_num'], dropout_rate=args['actor_dropout_rate'], bn_flag=args['actor_bn_flag'], leaky_relu=args['actor_leaky_relu'])
    critic_model = Critic(d_z_melody=EC2_model.z1_dims+EC2_model.z2_dims, d_z_accompany=poly_model.decoder.z_size, d_input=args['critic_d_input'],
     d_mid=args['critic_d_mid'], layer_num=args['critic_layer_num'], dropout_rate=args['critic_dropout_rate'], bn_flag=args['critic_bn_flag'], leaky_relu=args['critic_leaky_relu'])
     
    if args['D_load']:
        checkpoint = torch.load(os.path.join(args['D_model_param_path'], 'params_adversarial/best-model.pt'))
        critic_model.load_state_dict({k.replace('module.',''):v for k,v in checkpoint['critic_state_dict'].items()})
        log.info("loaded critic_model from:"+args['D_model_param_path'])

    if args['parallel']:
        actor_model = torch.nn.DataParallel(actor_model)
        critic_model = torch.nn.DataParallel(critic_model)
    actor_model.cuda()
    critic_model.cuda()

    with open(os.path.join(args['eval_path'], 'evaluation_model/model_config.json')) as f:
        eval_args = json.load(f)
    eval_model = PianoTree_evaluation(max_simu_note=eval_args['max_simu_note'], max_pitch=MAX_PITCH, min_pitch=MIN_PITCH, note_emb_size=eval_args['note_emb_size'], melody_hidden_dims=eval_args['melody_hid_size'],
                                      enc_notes_hid_size=eval_args['enc_notes_hid_size'], enc_time_hid_size=eval_args['enc_time_hid_size'], before_cat_dim=eval_args['before_cat_dim'], mid_sizes=eval_args['mid_sizes'], drop_out_p=eval_args['dropout_rate'])
    eval_checkpoint = torch.load(os.path.join(args['eval_path'], "params_evaluation/best-model.pt"))
    eval_model.load_state_dict({k.replace('module.', ''): v for k, v in eval_checkpoint['state_dict'].items()})
    eval_model.eval()
    eval_model.cuda()
    log.info("load eval model from "+args['eval_path'])

    log.info("model loaded")
    actor_optimizer = torch.optim.Adam(actor_model.parameters(), lr=args['lr'], betas=args['adam_betas'])
    critic_optimizer = torch.optim.Adam(critic_model.parameters(), lr=args['lr'], betas=args['adam_betas'])

    writer = SummaryWriter('run_adversarial')
    MODEL_PATH = 'params_adversarial'
    os.makedirs(MODEL_PATH, exist_ok=True)

    dummy_input = (torch.rand(1, actor_model.d_z_melody, device="cuda"), torch.rand(1, actor_model.d_z_accompany, device="cuda"))
    writer.add_graph(critic_model, dummy_input)

    step = 0
    epoch_num = args['iteration_num']//len(train_loader) + 1

    best_step=0
    with torch.no_grad():
            best_avg_acc = eval_epoch(-1, val_loader, actor_model, critic_model, eval_model, EC2_model, poly_model, step, average_scale_melody, average_scale_accompany, writer)
    for epoch in range(epoch_num):
        step = train_epoch(args, train_loader, actor_model, critic_model, actor_optimizer, critic_optimizer, step, average_scale_melody, average_scale_accompany, writer, EC2_model, poly_model)
        with torch.no_grad():
            avg_acc = eval_epoch(epoch, val_loader, actor_model, critic_model, eval_model, EC2_model, poly_model, step, average_scale_melody, average_scale_accompany, writer)

        checkpoint = {'step': step,
                    'avg_acc':avg_acc,
                    'actor_state_dict': actor_model.state_dict(),
                    'critic_state_dict': critic_model.state_dict(),}
        
        if avg_acc > best_avg_acc:
            name = 'best-model.pt'
            best_avg_acc = avg_acc
            best_step = step
        else:
            name = 'epoch-model.pt'
        torch.save(checkpoint, os.path.join(MODEL_PATH, name))
    
    log.info("best step: %d; avg_acc: %.2f"%(best_step, best_avg_acc))

if __name__ == '__main__':
    config_fn = './adversarial/model_config.json'
    with open(config_fn) as f:
        args = json.load(f)

    assert args["pure2G_times"]>=0
    if args['train_D2G_times']<1:
        args['train_G2D_times'] = int(1.0/args['train_D2G_times'])
        args['train_D2G_times'] = 1
    else:
        args['train_G2D_times'] = 1

    set_log_file('train_log_adversarial.txt', file_only=args['ssh'])

    log.info('Host: {}, user: {}, CUDA_VISIBLE_DEVICES: {}, cwd: {}'.format(
        socket.gethostname(), getpass.getuser(), os.environ.get('CUDA_VISIBLE_DEVICES', ''), os.getcwd()))
    
    main(args)