import os
from os.path import join
import sys
import argparse
# import wandb

sys.path.append("/workspace/zsl-res")


import torch
import numpy as np
import random

from MODEL.data import build_dataloader
from MODEL.modeling import build_zsl_pipeline
from MODEL.solver import make_optimizer, make_lr_scheduler
from MODEL.engine.trainer import do_train

from MODEL.config import cfg
from MODEL.utils.comm import *

from MODEL.utils import ReDirectSTD

try:
    from apex import amp
except ImportError:
    raise ImportError('Use APEX for multi-precision via apex.amp')

def train_model(cfg, local_rank, distributed, seed=214):

    # seed = 523
    # seed = 123
    # seed = 656
    # seed = 777
    # seed = 214
    # print('#'*100)
    # print(local_rank)
    # local_seed = seed
    local_seed = local_rank*100 + seed
    torch.manual_seed(local_seed)
    torch.cuda.manual_seed_all(local_seed)
    np.random.seed(local_seed)
    random.seed(local_seed)
    #TODO
    """
    g = torch.Generator()
    g.manual_seed(local_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic=True
    """
    model = build_zsl_pipeline(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model = model.to(device)

    optimizer = make_optimizer(cfg, model)
    scheduler = make_lr_scheduler(cfg, optimizer)

    use_mixed_precision = cfg.DTYPE == "float16"
    amp_opt_level = 'O1' if use_mixed_precision else 'O0'
    model, optimizer = amp.initialize(model, optimizer, opt_level=amp_opt_level)

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,find_unused_parameters=True,
        )

    tr_dataloader, tu_loader, ts_loader, res = build_dataloader(cfg, is_distributed=distributed)

    output_dir = cfg.OUTPUT_DIR
    model_file_name = cfg.MODEL_FILE_NAME
    model_file_path = join(output_dir, model_file_name)

    test_gamma = cfg.TEST.GAMMA
    max_epoch = cfg.SOLVER.MAX_EPOCH
    resume_from = cfg.MODEL.RESUME_FROM
    lamd = {
        1: cfg.MODEL.LOSS.LAMBDA1,
        2: cfg.MODEL.LOSS.LAMBDA2,
        3: cfg.MODEL.LOSS.LAMBDA3,
        4: cfg.MODEL.LOSS.LAMBDA4,
        5: cfg.MODEL.LOSS.LAMBDA5,
        6: cfg.MODEL.LOSS.LAMBDA6,
        7: cfg.MODEL.LOSS.LAMBDA7
    }

    do_train(
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
    )

    return model


def main():
    parser = argparse.ArgumentParser(description="PyTorch Zero-Shot Learning Training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--temp",
        default=0.1,
        type=float,
    )
    parser.add_argument(
        "--margin",
        default=0.8,
        type=float,
    )
    parser.add_argument(
        "--lambda_proto",
        default=1.,
        type=float,
    )
    parser.add_argument(
        "--lambda_att",
        default=1.,
        type=float,
    )
    parser.add_argument(
        "--lambda_cont",
        default=1.,
        type=float,
    )
    parser.add_argument(
        "--atten_thr",
        default=9.,
        type=float,
    )
    parser.add_argument(
        "--prefix",
        default="",
        type=str,
    )
    parser.add_argument(
        "--seed",
        default=214,
        type=int,
    )
    parser.add_argument(
        "--scale",
        default=25.,
        type=float,
    ) 
    parser.add_argument(
        "--scale_semantic",
        default=25.,
        type=float,
    ) 
    parser.add_argument(
        "--gamma",
        default=0.7,
        type=float,
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--alpha",
        default=0.5,
        type=float,
    )
    parser.add_argument(
        "--beta",
        default=0.,
        type=float,
    )
    parser.add_argument(
        "--orth",
        default=False,
        type=bool,
    )
    parser.add_argument(
        "--way",
        default=4,
        type=int,
    )
    parser.add_argument(
        "--shot",
        default=2,
        type=int,
    )
    parser.add_argument("--local_rank", type=int, default=0)

    args = parser.parse_args()
    # if is_main_process():
    #     wandb.init(project='ZSL')
    #     wandb.config.zsl_best = 0.
    #     wandb.config.gzsl_h_best = 0.
    #     wandb.config.gzsl_s_best = 0.
    #     wandb.config.gzsl_u_best = 0.

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    cfg.merge_from_file(args.config_file)
    opts = ['MODEL.LOSS.TEMP',args.temp,'MODEL.LOSS.MARGIN',args.margin,'PREFIX',args.prefix,'MODEL.LOSS.LAMBDA6',args.lambda_cont,'MODEL.LOSS.LAMBDA4',args.lambda_att,'MODEL.LOSS.LAMBDA5',args.lambda_proto,'MODEL.LOSS.ALPHA', args.alpha ,'MODEL.LOSS.BETA', args.beta ,'MODEL.ATTEN_THR',args.atten_thr,'MODEL.SCALE',args.scale,'TEST.GAMMA',args.gamma,'MODEL.ORTH', args.orth,'DATASETS.WAYS',args.way,'DATASETS.SHOTS',args.shot,'MODEL.SCALE_SEMANTIC',args.scale_semantic]
    cfg.merge_from_list(opts)
    cfg.freeze()
    output_dir = cfg.OUTPUT_DIR if args.output_dir is None else args.output_dir
    log_file_name = cfg.LOG_FILE_NAME
    current_time = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()).replace(' ','-')
    log_file_path = join(output_dir, f'{cfg.PREFIX}_{current_time}_seed_{args.seed}_temp_{args.temp}_margin_{args.margin}_lc_{args.lambda_cont}_la_{args.lambda_att}_lp_{args.lambda_proto}_alpha_{args.alpha}_beta_{args.beta}_thr_{args.atten_thr}_scale_{args.scale}_gamma_{args.gamma}_way_{args.way}_shot_{args.shot}-{log_file_name}')
    print(log_file_path)

    if is_main_process():
        ReDirectSTD(log_file_path, 'stdout', True)

    print("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        print(config_str)
    print("Running with config:\n{}".format(cfg))
        
    # torch.backends.cudnn.benchmark = True
    model = train_model(cfg, args.local_rank, args.distributed,args.seed)


if __name__ == '__main__':
    main()