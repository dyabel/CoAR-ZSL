import torch
import numpy as np

import torch.distributed as dist
from MODEL.utils.comm import *
from .inferencer import eval_zs_gzsl
from apex import amp
import os
from copy import deepcopy
# if is_main_process():
#     import wandb

def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses

def do_train(
        model,
        tr_dataloader,
        tu_loader,
        ts_loader,
        res,
        optimizer,
        scheduler,
        lamd,
        test_gamma,
        device,
        max_epoch,
        model_file_path,
        resume_from
    ):
    best_performance = [0, 0, 0, 0]
    best_epoch = -1

    att_seen = res['att_seen'].to(device)
    att_unseen = res['att_unseen'].to(device)

    losses = []
    cls_losses = []
    reg_losses = []
    ad_losses = []
    cpt_losses = []
    cls_att_losses = []
    proto_losses = []
    proto_align_losses = []
    scale_all = []
    start_epoch = 0
    if resume_from is not None:
        checkpoint = torch.load(resume_from,map_location=torch.device('cpu'))
        # print(len(checkpoint['optimizer_state_dict']))
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'][0])
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        synchronize()
        acc_seen, acc_novel, H, acc_zs = eval_zs_gzsl(
            tu_loader,
            ts_loader,
            res,
            model,
            test_gamma,
            device)

        synchronize()

        if is_main_process():
            print('zsl: %.4f, gzsl: seen=%.4f, unseen=%.4f, h=%.4f' % (acc_zs, acc_seen, acc_novel, H))


    model.train()

    for epoch in range(start_epoch, max_epoch):

        loss_epoch = []
        cls_loss_epoch = []
        reg_loss_epoch = []
        ad_loss_epoch = []
        cpt_loss_epoch = []
        lcls_att_epoch = []
        proto_loss_epoch = []
        proto_align_loss_epoch = []

        scale_epoch = []

        scheduler.step()
        for iteration, (batch_img, batch_att, batch_label) in enumerate(tr_dataloader):
            batch_img = batch_img.to(device)
            batch_att = batch_att.to(device)
            batch_label = batch_label.to(device)

            loss_dict = model(x=batch_img, att=batch_att, label=batch_label, seen_att=att_seen,att_unseen=att_unseen)

            Lreg = loss_dict['Reg_loss']
            Lcls = loss_dict['Cls_loss']
            Lad = loss_dict['AD_loss']
            Lcpt = loss_dict['CPT_loss']
            Lcls_att = loss_dict['ATTCLS_loss']
            L_proto = loss_dict['Proto_loss']
            L_proto_align = loss_dict['Proto_align_loss']

            scale = loss_dict['scale']

            loss_dict.pop('scale')

            loss = Lcls + lamd[1]*Lreg + lamd[2]*Lad + lamd[3]*Lcpt + lamd[4]*Lcls_att + lamd[5]*L_proto + lamd[6]*L_proto_align

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = reduce_loss_dict(loss_dict)

            lreg = loss_dict_reduced['Reg_loss']
            lcls = loss_dict_reduced['Cls_loss']
            lad = loss_dict_reduced['AD_loss']
            lcpt = loss_dict_reduced['CPT_loss']
            lcls_att = loss_dict_reduced['ATTCLS_loss']
            l_proto = loss_dict_reduced['Proto_loss']
            l_proto_align = loss_dict_reduced['Proto_align_loss']

            losses_reduced = lcls + lamd[1]*lreg + lamd[2]*lad + lamd[3]*lcpt + lamd[4]*lcls_att + lamd[5]*l_proto + lamd[6]*l_proto_align

            optimizer.zero_grad()

            # Note: If mixed precision is not used, this ends up doing nothing
            # Otherwise apply loss scaling for mixed-precision recipe
            with amp.scale_loss(loss, optimizer) as scaled_losses:
                scaled_losses.backward()
            # print([group['lr'] for group in optimizer.param_groups])
            optimizer.step()

            loss_epoch.append(losses_reduced.item())
            cls_loss_epoch.append(lcls.item())
            reg_loss_epoch.append(lreg.item())
            ad_loss_epoch.append(lad.item())
            cpt_loss_epoch.append(lcpt.item())
            lcls_att_epoch.append(lcls_att.item())
            proto_loss_epoch.append(l_proto.item())
            proto_align_loss_epoch.append(l_proto_align.item())
            scale_epoch.append(scale)

        if is_main_process():
            losses += loss_epoch
            cls_losses += cls_loss_epoch
            reg_losses += reg_loss_epoch
            ad_losses += ad_loss_epoch
            cpt_losses += cpt_loss_epoch
            cls_att_losses += lcls_att_epoch
            proto_losses += proto_loss_epoch
            proto_align_losses += proto_align_loss_epoch
            scale_all += scale_epoch

            loss_epoch_mean = sum(loss_epoch)/len(loss_epoch)
            cls_loss_epoch_mean = sum(cls_loss_epoch)/len(cls_loss_epoch)
            reg_loss_epoch_mean = sum(reg_loss_epoch)/len(reg_loss_epoch)
            ad_loss_epoch_mean = sum(ad_loss_epoch)/len(ad_loss_epoch)
            cpt_loss_epoch_mean = sum(cpt_loss_epoch)/len(cpt_loss_epoch)
            cls_att_epoch_mean = sum(lcls_att_epoch) / len(lcls_att_epoch)
            proto_loss_epoch_mean = sum(proto_loss_epoch) / len(proto_loss_epoch)
            proto_align_loss_epoch_mean = sum(proto_align_loss_epoch) / len(proto_align_loss_epoch)
            scale_epoch_mean = sum(scale_epoch) / len(scale_epoch)

            losses_mean = sum(losses) / len(losses)
            cls_losses_mean = sum(cls_losses) / len(cls_losses)
            reg_losses_mean = sum(reg_losses) / len(reg_losses)
            ad_losses_mean = sum(ad_losses) / len(ad_losses)
            cpt_losses_mean = sum(cpt_losses) / len(cpt_losses)
            cls_att_losses_mean = sum(cls_att_losses) / len(cls_att_losses)
            proto_losses_mean = sum(proto_losses) / len(proto_losses)
            proto_align_losses_mean = sum(proto_align_losses) / len(proto_align_losses)
            scale_all_mean = sum(scale_all) / len(scale_all)


            log_info = 'epoch: %d  |  loss: %.4f (%.4f), cls_loss: %.4f (%.4f),   reg_loss: %.4f (%.4f),   ad_loss: %.4f (%.4f),   cpt_loss: %.4f (%.4f), cls_att_losses: %.4f(%.4f), proto_losses: %.4f(%.4f), proto_align_losses: %.4f(%.4f),scale:  %.4f (%.4f),    lr: %.6f' % \
                       (epoch, loss_epoch_mean, losses_mean, cls_loss_epoch_mean, cls_losses_mean, reg_loss_epoch_mean,
                        reg_losses_mean, ad_loss_epoch_mean, ad_losses_mean, cpt_loss_epoch_mean, cpt_losses_mean,cls_att_epoch_mean,cls_att_losses_mean,proto_loss_epoch_mean,proto_losses_mean,proto_align_loss_epoch_mean,proto_align_losses_mean,
                        scale_epoch_mean, scale_all_mean, optimizer.param_groups[0]["lr"])
            print(log_info)

        synchronize()
        acc_seen, acc_novel, H, acc_zs = eval_zs_gzsl(
            tu_loader,
            ts_loader,
            res,
            model,
            test_gamma,
            device)

        synchronize()

        if is_main_process():
            print('zsl: %.4f, gzsl: seen=%.4f, unseen=%.4f, h=%.4f' % (acc_zs, acc_seen, acc_novel, H))

            if H > best_performance[-1] or epoch % 1 == 0:
                data = {}
                # model_save = deepcopy(model).cpu()
                data["model_state_dict"] = model.state_dict()
                data["optimizer_state_dict"] = optimizer.state_dict(),
                data['epoch'] = epoch
                if H > best_performance[-1]:
                    best_epoch = epoch + 1
                    best_performance[1:] = [acc_seen, acc_novel, H]
                    torch.save(data, model_file_path)
                    print('save model: ' + model_file_path)
                else:
                    torch.save(data, os.path.dirname(model_file_path) + '/epoch_{}.pth'.format(epoch))
                    print('save model epoch_{}.pth'.format(epoch))
            # if H > best_performance[-1]:
            #     wandb.config.update({'gzsl_h_best':H}, allow_val_change=True)
            #     wandb.config.update({'gzsl_s_best':acc_seen}, allow_val_change=True)
            #     wandb.config.update({'gzsl_u_best':acc_novel}, allow_val_change=True)
            #
            if acc_zs > best_performance[0]:
                best_performance[0] = acc_zs
            #     wandb.config.update({'zsl_best':acc_zs}, allow_val_change=True)

    if is_main_process():
        print("best: ep: %d" % best_epoch)
        print('zsl: %.4f, gzsl: seen=%.4f, unseen=%.4f, h=%.4f' % tuple(best_performance))