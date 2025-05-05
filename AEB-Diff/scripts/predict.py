import torch
import os

from sympy.codegen import Print

EPS = 1e-6  # 避免除零
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

def pdist(a, b):
    """
    计算两个图像集之间的IoU距离矩阵
    """
    N = a.shape[1]
    M = b.shape[1]
    H = a.shape[-2]
    W = a.shape[-1]

    # 使用repeat来扩展a和b的维度以计算成对的交并比
    aRep = a.repeat(1, M, 1, 1).view(-1, N, M, H, W)
    bRep = b.repeat(1, N, 1, 1).view(-1, M, N, H, W).transpose(1, 2)

    # 计算交集和并集
    inter = (aRep & bRep).float().sum(-1).sum(-1) + EPS
    union = (aRep | bRep).float().sum(-1).sum(-1) + EPS
    IoU = inter / union
    dis = (1 - IoU).mean(-1).mean(-1)  # 取平均值
    return dis

def generalized_energy_distance(seg, prd):
    """
    计算广义能力距离 (GED)

    参数:
        seg: 真实标签图像集（Tensor），形状为 (batch, 4, height, width)
        prd: 预测标签图像集（Tensor），形状为 (batch, 4, height, width)

    返回:
        ged_score: 广义能力距离分数
    """
    seg = seg.type(torch.ByteTensor)  # 将seg转换为Byte类型
    prd = prd.type_as(seg)            # 确保prd和seg的类型一致

    # 计算三个距离期望项
    dSP = pdist(seg, prd)  # 计算 E[d(S, Y)]
    dSS = pdist(seg, seg)  # 计算 E[d(S, S')]
    dPP = pdist(prd, prd)  # 计算 E[d(Y, Y')]

    # 计算GED
    ged_score = (2 * dSP - dSS - dPP)
    return ged_score

def dice_coefficient(img1, img2):
    """计算两个二值图像的Dice系数"""
    img1 = img1.to(torch.bool)  # 确保是布尔类型
    img2 = img2.to(torch.bool)  # 确保是布尔类型
    intersection = torch.logical_and(img1, img2).sum()
    return 2 * intersection / (img1.sum() + img2.sum() + EPS)

def combined_sensitivity(gt, pred):
    """计算组合敏感度 (Sc)"""
    device = gt.device
    # 初始化联合区域
    union_all_gt = torch.zeros_like(gt[0], dtype=torch.bool)  # 真实标签的联合区域
    union_all_pred = torch.zeros_like(pred[0], dtype=torch.bool)  # 预测标签的联合区域
    # 计算所有真实标签图像的联合区域（按位“或”）
    for c in range(gt.shape[1]):  # 遍历所有通道
        union_all_gt = torch.logical_or(union_all_gt, gt[0, c])
    # 计算所有预测标签图像的联合区域（按位“或”）
    for c in range(pred.shape[1]):  # 遍历所有预测通道
        union_all_pred = torch.logical_or(union_all_pred, pred[0, c])
    # 确保gt和pred的张量都在相同的设备上
    union_all_gt = union_all_gt.to(device)  # 将union_all_gt转移到gt所在的设备
    union_all_pred = union_all_pred.to(device)  # 将union_all_pred转移到gt所在的设备
    # 计算交集区域：预测区域和真实区域的交集
    intersection_all = torch.logical_and(union_all_gt, union_all_pred)

    # 计算TP（True Positive）：交集区域的像素值数量
    TP = intersection_all.sum()

    # 计算FN（False Negative）：真实区域减去交集区域
    FN = (union_all_gt & ~intersection_all).sum()  # 真实标签区域减去交集区域
    # 如果交集不为零，计算Sc = TP / (TP + FN)
    if TP > 0:
        Sc = TP / (TP + FN)
    else:
        # 如果交集为零，则返回Sc = 1
        Sc = torch.tensor(1.0)
    return Sc

def dice_matching(gt, pred):
    """计算Dice Matching (Dm)"""
    dice_scores = []
    for i in range(gt.shape[1]):  # 遍历每个通道（图像）
        g = gt[0, i, :, :]  # 获取第i个真实标签图像
        best_dice = -1  # 初始化最好的Dice值
        for j in range(pred.shape[1]):  # 遍历每个预测图像的通道
            p = pred[0, j, :, :]  # 获取第j个预测图像
            best_dice = max(best_dice, dice_coefficient(g, p))  # 找到最好的Dice系数
        dice_scores.append(best_dice)
    return torch.mean(torch.tensor(dice_scores))

def diversity_agreement(gt, pred):
    """计算Diversity Agreement (Da)"""

    # 获取gt和pred的图像数量
    num_images = gt.shape[1]

    # 存储所有方差差异
    gt_var_diff = []
    pred_var_diff = []

    # 遍历所有图像对
    for i in range(num_images):
        for j in range(i + 1, num_images):
            # 计算每对图像的方差差异
            gt_var_diff.append(torch.abs(torch.var(gt[0, i, :, :].float()) - torch.var(gt[0, j, :, :].float())))
            pred_var_diff.append(torch.abs(torch.var(pred[0, i, :, :].float()) - torch.var(pred[0, j, :, :].float())))

    # 计算最大差异和最小差异
    delta_max = abs(max(gt_var_diff) - max(pred_var_diff))
    delta_min = abs(min(gt_var_diff) - min(pred_var_diff))

    # 计算Diversity Agreement
    Da = 1 - (delta_max + delta_min) / 2
    return Da

def calculate_ci_score(gt, pred):
    """计算CI Score (组合敏感度, Dice匹配, 多样性一致性)"""
    sc = combined_sensitivity(gt, pred)
    print("sc:", sc)
    dm = dice_matching(gt, pred)
    print("dm:", dm)
    da = diversity_agreement(gt, pred)
    print("da:", da)
    return dm, 3 * sc * dm * da / (sc + dm + da)


# 示例用法GED
# # 假设 seg 和 prd 是形状为 (batch, 4, height, width) 的二值图像
# seg = torch.randint(0, 2, (1, 4, 128, 128))  # 真实标签图像集
# prd = torch.randint(0, 2, (1, 4, 128, 128))  # 预测标签图像集
# ged_score = generalized_energy_distance(seg, prd)
# print(f"Generalized Energy Distance (GED): {ged_score.item()}")
#
# ci_score = calculate_ci_score(seg, prd)
# print(f"CI Score: {ci_score.item()}")
