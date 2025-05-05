"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os
import torch
import pandas as pd
from visdom import Visdom

from predict import generalized_energy_distance, calculate_ci_score

viz = Visdom(port=8097)
import sys
import random
sys.path.append(".")
import numpy as np
import torch as th
from guided_diffusion import dist_util, logger
from guided_diffusion.lidcloader import LIDCDatasetBeta
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
seed=10
th.manual_seed(seed)
th.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min)/ (_max - _min)
    return normalized_img

def dice_score(pred, targs):
    pred = (pred>0).float()
    return 2. * (pred*targs).sum() / (pred+targs).sum()


def main():
    args = create_argparser().parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    os.environ["MASTER_PORT"] = str(1028)

    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion, prior, posterior = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    torch.cuda.set_device(0)  # 假设使用第一个GPU
    model.to(torch.device('cuda:0'))
    model = model.to(torch.device('cuda:0'))

    model.to(dist_util.dev())
    prior.to(torch.device('cuda:0'))

    ds = LIDCDatasetBeta(args.data_dir, test_flag=True)

    datal = torch.utils.data.DataLoader(
        ds,
        batch_size=1,
        shuffle=False
    )
    data = iter(datal)

    all_images = []
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()
    ID = []
    GED = []
    Dm = []
    CI_Score = []
    # while len(all_images) * args.batch_size < args.num_samples:
    #     b, labels, path = next(data)  #should return an image from the dataloader "data"

    # 定义保存路径
    csv_file_path = '/data/jupyter/xjl/AMISDM_Beta/results/eval_1129.csv'

    # 如果文件不存在，先写入表头
    if not os.path.exists(csv_file_path):
        with open(csv_file_path, 'w') as f:
            f.write('ID,GED,Dm,CI_Score\n')

    for b, labels, path in data:
        c = th.randn_like(b[:, :1, ...])
        img = th.cat((b, c), dim=1)     #add a noise channel$
        slice_ID = path[0].split("/")[-2]
        viz.image(visualize(img[0,0,...]), opts=dict(caption="image input"))

        label_img = labels.squeeze(0)  #torch.Size([4, 128, 128])
        label_img = label_img.unsqueeze(0)
        # for i in range(label_img.shape[1]):  # 遍历每个标签
        #     viz.image(label_img[0, i], opts=dict(caption=f"gt_{i + 1}"))
        viz.image(label_img[0], opts=dict(caption=f"gt"))
        viz.image(visualize(img[0, 4, ...]), opts=dict(caption="noise"))

        logger.log("sampling...")

        start = th.cuda.Event(enable_timing=True)
        end = th.cuda.Event(enable_timing=True)

        gen_preds = []
        for i in range(args.num_ensemble):  #this is for the generation of an ensemble of 5 masks.
            model_kwargs = {}
            start.record()
            sample_fn = (
                diffusion.p_sample_loop_known if not args.use_ddim else diffusion.ddim_sample_loop_known
            )

            sample, x_noisy, org = sample_fn(
                model,
                (args.batch_size, 3, args.image_size, args.image_size), img,
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
            )

            end.record()
            th.cuda.synchronize()
            print('time for 1 sample', start.elapsed_time(end))  #time measurement for the generation of 1 sample
            s = th.tensor(sample)

            viz.image(visualize(sample[0, 0, ...]), opts=dict(caption="sampled output"))
            # th.save(s, './results/'+str(slice_ID)+'_output'+str(i)) #save the generated mask
            # 将生成的样本加入到列表中 (保持 s 的形状为 [1, 1, 128, 128])
            gen_preds.append(s)

        # 将列表中的所有样本合并为一个张量，形成大小为 (1, 4, 128, 128) 的张量
        # 假设 args.num_ensemble 为 4，则拼接成一个包含 4 个样本的张量
        preds = th.cat(gen_preds, dim=1)  # 在通道维度（dim=1）拼接
        device = label_img.device

        label_img = label_img.to(device)  # 将 label_img 转移到 cuda:1
        preds = preds.to(device)  # 将 preds 转移到 cuda:1
        preds = (preds > 0.5).to(torch.bool)

        ged_score = generalized_energy_distance(label_img, preds)
        print("ID:", slice_ID)
        print(f"Generalized Energy Distance (GED): {ged_score.item()}")

        dm, ci_score = calculate_ci_score(label_img, preds)
        print("dm:", dm)
        print(f"CI Score: {ci_score.item()}")
        ID.append(slice_ID)
        GED.append(ged_score)
        Dm.append(dm)
        CI_Score.append(ci_score)

        # 将当前结果保存到CSV文件
        current_data = pd.DataFrame({
            'ID': [slice_ID],
            'GED': [ged_score.item()],
            'Dm': [dm.item()],
            'CI_Score': [ci_score.item()]
        })
        current_data.to_csv(csv_file_path, mode='a', header=False, index=False)
        print(f"Metrics for ID={slice_ID} saved to {csv_file_path}")

    # 计算平均值
    mean_GED = torch.tensor([ged.item() for ged in GED]).mean()
    mean_Dm = torch.tensor([dm.item() for dm in Dm]).mean()
    mean_CI_Score = torch.tensor([ci.item() for ci in CI_Score]).mean()

    # 打印平均值
    print(f"Mean GED: {mean_GED.item()}")
    print(f"Mean Dm: {mean_Dm.item()}")
    print(f"Mean CI_Score: {mean_CI_Score.item()}")

    # 将平均值追加到CSV文件
    mean_data = pd.DataFrame({
        'ID': ['Average'],
        'GED': [mean_GED.item()],
        'Dm': [mean_Dm.item()],
        'CI_Score': [mean_CI_Score.item()]
    })
    mean_data.to_csv(csv_file_path, mode='a', header=False, index=False)
    print("Average metrics added.")


def create_argparser():
    defaults = dict(
        data_dir="/data/jupyter/xjl/AMISDM/data/testing/data/jupyter/xjl/AMISDM/data/testing",
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        use_ddim=False,
        model_path="",
        num_ensemble=6      #number of samples in the ensemble
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=1)
    parser.add_argument('--ngpu', type=int, default=1)
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":

    main()
