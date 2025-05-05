import sys
import os
import argparse
sys.path.append("..")
sys.path.append(".")
from guided_diffusion import logger
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.lidcloader import LIDCDatasetBeta
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
import torch
from guided_diffusion.train_util import TrainLoop
from visdom import Visdom

viz = Visdom(port=8097)

def main():
    args = create_argparser().parse_args()

    # 设置CUDA设备为cuda:0
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    logger.configure()

    logger.log("Creating model, diffusion, prior, and posterior distribution...")
    model, diffusion, prior, posterior = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    # 将模型和相关组件转移到GPU 0
    torch.cuda.set_device(0)
    model.to(torch.device('cuda:0'))
    prior.to(torch.device('cuda:0'))
    posterior.to(torch.device('cuda:0'))

    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion, maxt=1000)

    data_dir = "/data1/xjlDatasets/LIDC/manifest-1600709154662/data_AMISDM_Beta/training"
    logger.log("Creating data loader...")

    ds = LIDCDatasetBeta(data_dir, test_flag=False)  # 使用 Beta 分布的加载器

    datal = torch.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True  # 保持随机性
    )
    data = iter(datal)

    logger.log("Training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        classifier=None,
        data=data,
        dataloader=datal,
        prior=prior,
        posterior=posterior,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="/data1/xjlDatasets/LIDC/manifest-1600709154662/data_AMISDM_Beta/training",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=100,
        save_interval=5000,
        resume_checkpoint='',  # Set path to pretrained model checkpoint if needed
        use_fp16=False,
        fp16_scale_growth=1e-3,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=1)
    parser.add_argument('--ngpu', type=int, default=1)
   
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
