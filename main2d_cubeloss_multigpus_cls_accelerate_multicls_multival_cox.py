import os
import argparse
import torch
import sys
from time import time
import numpy as np
from torch.optim import AdamW, SGD, Adam
from optimizers.lr_scheduler import WarmupCosineSchedule
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import tqdm
import yaml
from utils.dataset_clip import dataset_clip_cls, dataset_clip_cox
from clip_model import CLIP_CLS
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
import pandas as pd
from accelerate import Accelerator
from sklearn.metrics import ConfusionMatrixDisplay
import SimpleITK as sitk
from torch.autograd import Variable
import xlsxwriter
sys.path.append('/data1/lcc/code/code/ubuntu/EGFR/clip/models/')
from densenet import DenseNet
sys.path.append('/data1/lcc/code/code/ubuntu/EGFR/M3D_main/LaMed/src/model/')
from CLIP import M3DCLIPConfig, M3D_CLS
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

# 新增：用于生存分析评估的工具
from lifelines.utils import concordance_index
import warnings
warnings.filterwarnings("ignore")

accelerator = Accelerator()

import torch
import numpy as np


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
from lifelines.utils import concordance_index

_eps = 1e-12

def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        print(f"张量原始类型: {x.dtype}")  # 调试用，确认是否为 bfloat16
        if x.dtype == torch.bfloat16:
            x = x.to(torch.float32)
        x = x.detach().cpu().numpy()
    return np.asarray(x)

def sanitize_matrix(mat, fill_value=0.0):
    """
    将输入矩阵中的 +/-Inf -> NaN -> 用 fill_value 填充；同时把完全 NaN 的行删除并返回 mask。
    返回: mat_clean (二维), valid_mask (一维布尔，True 表示该样本有效)
    """
    mat = _to_numpy(mat)
    if mat.ndim == 1:
        mat = mat.reshape(1, -1)
    mat = mat.astype(float)
    mat[np.isinf(mat)] = np.nan
    # 行上全是 NaN 的标记
    row_all_nan = np.all(np.isnan(mat), axis=1)
    # 用列均值/全局 fill_value 填充剩余 NaN（列均值更稳健）
    col_mean = np.nanmean(mat, axis=0)
    # 如果某列全 NaN，nanmean 返回 nan -> 替成 fill_value
    col_mean = np.where(np.isnan(col_mean), fill_value, col_mean)
    inds = np.where(np.isnan(mat))
    if inds[0].size > 0:
        mat[inds] = np.take(col_mean, inds[1])
    # 最终仍然存在 NaN 的行（理论上不应该），标为无效
    still_nan_rows = np.any(np.isnan(mat), axis=1)
    valid_mask = ~(row_all_nan | still_nan_rows)
    return mat, valid_mask

def hazards_to_cumhaz(hazards):
    """
    hazards: (n_samples, n_bins), 每元素在 [0,1) 表示离散区间风险率 h_k
    返回: cumhaz: shape (n_samples,) = -log(S_last)
    """
    h, mask = sanitize_matrix(hazards, fill_value=0.0)
    # clip 保证 1-h 在数值上不为0或负
    one_minus_h = np.clip(1.0 - h, a_min=_eps, a_max=1.0)
    S = np.cumprod(one_minus_h, axis=1)
    S_last = S[:, -1]
    cumhaz = -np.log(np.clip(S_last, _eps, 1.0))
    # 对无效样本赋 NaN
    cumhaz[~mask] = np.nan
    return cumhaz, mask

def surv_to_neg_expected_time(surv, times=None):
    """
    surv: (n_samples, n_times) survival probabilities S(t)
    times: length n_times array of time grid (如果 None，则等间隔处理)
    返回: -E[T] per sample, mask
    """
    s, mask = sanitize_matrix(surv, fill_value=1.0)  # survival 默认填 1
    n_samples, n_times = s.shape
    if times is None:
        times = np.arange(n_times, dtype=float)
    times = np.asarray(times, dtype=float)
    if times.shape[0] != n_times:
        # 如果长度不匹配尝试生成等间距
        times = np.linspace(0, n_times - 1, n_times)
    # trapezoid 积分
    expected_T = np.trapz(s, x=times, axis=1)
    neg_et = -expected_T
    neg_et[~mask] = np.nan
    return neg_et, mask

def surv_horizon_risk(surv, times, t_star):
    """
    在 horizon t_star 处计算 risk = 1 - S(t_star)
    surv: (n_samples, n_times)
    times: time array len n_times
    t_star: 单个时间 (数值)
    """
    s, mask = sanitize_matrix(surv, fill_value=1.0)
    times = np.asarray(times, dtype=float)
    # 找到最接近 t_star 的索引（向下取最近）
    if t_star <= times[0]:
        idx = 0
    elif t_star >= times[-1]:
        idx = len(times) - 1
    else:
        idx = np.searchsorted(times, t_star, side='right') - 1
    S_t = s[:, idx]
    risk = 1.0 - S_t
    risk[~mask] = np.nan
    return risk, mask

def preds_to_risk(preds, *, mode='auto', times=None, horizon=None):
    """
    将模型输出 preds 转为单一 risk 分数。
    preds: 可以是 hazards 或 survival 矩阵 (n, K) 或 torch.Tensor
    mode:
      - 'auto'：若值范围在 [0,1] 且逐列乘积接近 1 则视为 hazards -> cumhaz；
                否则视为 survival -> neg_expected_time
      - 'cumhaz'：强制按 hazards->cumhaz
      - 'neg_et'：按 survival->-E[T]
      - 'horizon'：按 horizon，需要提供 times 和 horizon
    times: 必要时提供时间轴（用于 neg_et 或 horizon）
    horizon: 当 mode=='horizon' 时需要指定（数值）
    返回: (risk_array, valid_mask)
    """
    p = _to_numpy(preds)
    if p.ndim == 1:
        # 单一数值输出（本身已是 risk），直接返回
        risk = p.astype(float)
        mask = ~np.isnan(risk) & np.isfinite(risk)
        return risk, mask

    # auto 判定
    if mode == 'auto':
        # 判断是否很像 hazards：多数值在 [0,1) 且列乘积（S_last）在 (0,1]
        all_in_01 = np.all((p >= -_eps) & (p <= 1.0 + _eps))
        if all_in_01:
            # 如果列乘积接近 1（S_last>0），认为是 hazards（因为1-h乘积通常<1）
            # 计算一行的 S_last 近似
            row_prod = np.prod(np.clip(1.0 - np.clip(p, 0.0, 1.0), _eps, 1.0), axis=1)
            # 如果平均 S_last 小于 0.999，则认为是 hazards（heuristic）
            if np.nanmean(row_prod) < 0.999:
                mode = 'cumhaz'
            else:
                mode = 'neg_et'
        else:
            mode = 'neg_et'

    if mode == 'cumhaz':
        risk, mask = hazards_to_cumhaz(p)
        return risk, mask
    elif mode == 'neg_et':
        risk, mask = surv_to_neg_expected_time(p, times=times)
        return risk, mask
    elif mode == 'horizon':
        if times is None or horizon is None:
            raise ValueError("mode='horizon' 时必须提供 times 和 horizon 参数")
        risk, mask = surv_horizon_risk(p, times=times, t_star=horizon)
        return risk, mask
    else:
        raise ValueError(f"未知 mode: {mode}")

def safe_concordance_index(durations, preds_risk, events):
    """
    对 NaN/Inf 做保护，最终调用 lifelines.utils.concordance_index
    durations, preds_risk, events: 一维数组或可转换为 numpy
    返回: cindex (float)
    """
    durations = _to_numpy(durations).astype(float)
    events = _to_numpy(events).astype(float)
    preds_risk = _to_numpy(preds_risk).astype(float)

    # mask: 需同时满足三者为 finite 且事件为 0/1
    finite_mask = np.isfinite(durations) & np.isfinite(preds_risk) & np.isfinite(events)
    # 事件二值化
    events_bin = (events != 0).astype(int)
    finite_mask &= ~np.isnan(preds_risk)
    # 至少保留 2 个样本
    if finite_mask.sum() < 2:
        return np.nan

    try:
        cind = concordance_index(durations[finite_mask], preds_risk[finite_mask], events_bin[finite_mask])
    except Exception as e:
        print("concordance_index 计算失败:", e)
        cind = np.nan
    return cind

# # ----------------------------
# # 示例：如何使用
# # ----------------------------
# if __name__ == "__main__":
#     # 假设 preds_hazards 是模型输出的 hazards 矩阵 (n, K)
#     # durations, events 为真实标签
#     # 1) hazards -> cumhaz -> cindex
#     # preds_hazards = ... (numpy or torch tensor)
#     # durations = ...
#     # events = ...
#     # risk, mask = preds_to_risk(preds_hazards, mode='cumhaz')
#     # cidx = safe_concordance_index(durations, risk, events)
#     # print("C-index:", cidx)

#     # 2) 如果模型输出的是 survival 矩阵 S(t)：
#     # surv_matrix = ...
#     # times = np.array([...])  # time grid 对应 surv_matrix 列
#     # risk, mask = preds_to_risk(surv_matrix, mode='neg_et', times=times)
#     # cidx = safe_concordance_index(durations, risk, events)
#     # print("C-index:", cidx)

#     pass




import torch
import torch.nn as nn

class PCHazardLoss(nn.Module):
    """
    Piecewise-constant hazard negative log-likelihood (vectorized, robust).
    Inputs:
      - pred_prob: (B, K) tensor. Can be either:
          * hazards h_k in (0,1) per interval (recommended), 或
          * survival S_k = P(T > t_k) per interval (decreasing in k).
      - true_time: (B,) continuous times (same unit as time_bins)
      - true_event: (B,) 0/1 (1=event observed, 0=censored)
      - time_bins: optional tensor/list of interval edges length K+1,
                   representing edges [t0=0, t1, t2, ..., tK]; interval k is [t_k, t_{k+1})
                   If None, will construct uniform bins from 0..max(true_time) with K intervals.
    Returns:
      - scalar loss (mean negative log-likelihood)
    """
    def __init__(self, num_durations, eps=1e-7):
        super().__init__()
        self.K = int(num_durations)
        self.eps = float(eps)

    def forward(self, pred_prob, true_time, true_event, time_bins=None):
        device = pred_prob.device if isinstance(pred_prob, torch.Tensor) else torch.device('cpu')

        # to tensors
        pred = torch.as_tensor(pred_prob, dtype=torch.float32, device=device)  # (B, K)
        durations = torch.as_tensor(true_time, dtype=torch.float32, device=device).reshape(-1)
        events = torch.as_tensor(true_event, dtype=torch.long, device=device).reshape(-1)

        B, K = pred.shape
        assert K == self.K, f"pred_prob.shape[1] ({K}) must equal num_durations ({self.K})"

        # --- 构建 time_bins (K intervals => K+1 edges) ---
        if time_bins is None:
            max_t = torch.clamp(durations.max(), min=1e-6)
            edges = torch.linspace(0.0, float(max_t), steps=K+1, device=device)
        else:
            edges = torch.as_tensor(time_bins, dtype=torch.float32, device=device)
            if edges.numel() == K:  # user passed K points -> interpret as K interval end points, prepend 0
                edges = torch.cat([torch.tensor([0.], device=device), edges])
            assert edges.numel() == K+1, "time_bins must have length K+1 (edges)"

        # --- 将每个样本的持续时间映射到区间索引 j in [0..K-1] ---
        # torch.bucketize: returns index i such that edges[i-1] <= x < edges[i] with side='right'
        # we want interval index j = i-1
        idx = torch.bucketize(durations, edges, right=False) - 1  # (B,)
        # clamp 以防极端时间等于最后边界 -> 属于最后区间
        idx = idx.clamp(min=0, max=K-1)

        # --- 如果 pred 是 survival S(t) 而不是 hazard h_k，转换为 hazard ---
        # Heuristic: 如果每行都单调不增并值在 [0,1]，我们认为输入为 survival
        is_in_01 = (pred >= -self.eps).all() and (pred <= 1.0 + self.eps).all()
        row_decreasing = torch.all(pred[:, 1:] <= pred[:, :-1] + 1e-6)
        if is_in_01 and row_decreasing:
            # pred is survival S_k. Convert to discrete hazard:
            # S_{k} = P(T > t_k). Let S_{-1} = 1. Then discrete hazard h_k = 1 - S_k / S_{k-1}
            S = torch.clamp(pred, min=self.eps, max=1.0)  # (B,K)
            S_prev = torch.cat([torch.ones((B,1), device=device), S[:, :-1]], dim=1)
            h = 1.0 - (S / S_prev)
            h = torch.clamp(h, min=self.eps, max=1.0 - self.eps)
        else:
            # Assume pred are hazards directly
            h = torch.clamp(pred, min=self.eps, max=1.0 - self.eps)

        # --- 计算 per-interval log(1-h) 和 prefix sums ---
        log1m = torch.log1p(-h)  # log(1-h_k), shape (B,K)
        # prefix sum of log1m across intervals: prefix_log1m[b,k] = sum_{t=0..k-1} log(1-h_t)
        # we can compute cumulative sum then shift by 1
        cumsum_log1m = torch.cumsum(log1m, dim=1)  # sum up to and including k
        # prefix (sum up to k-1):
        prefix_log1m = torch.cat([torch.zeros((B,1), device=device), cumsum_log1m[:, :-1]], dim=1)  # (B,K)

        # --- 对每个样本取其对应区间 j 的值 ---
        # gather needs indices of shape (B,1)
        idx_long = idx.long().unsqueeze(1)  # (B,1)
        log1m_prefix_at_j = prefix_log1m.gather(1, idx_long).squeeze(1)  # sum_{k<j} log(1-h_k)
        log_h_at_j = torch.log(h.gather(1, idx_long).squeeze(1))  # log(h_j)

        # --- 构建样本级别 log-likelihood ---
        # 如果 event==1 (uncensored): ll = prefix + log(h_j)
        # 如果 event==0 (censored):  ll = prefix + log(1-h_j)  (survive through interval j)
        log1m_at_j = log1m.gather(1, idx_long).squeeze(1)  # log(1-h_j)

        is_event = (events != 0)
        ll = torch.where(is_event,
                         log1m_prefix_at_j + log_h_at_j,
                         log1m_prefix_at_j + log1m_at_j)  # (B,)

        # Numerical protections: if any ll is -inf/nan, clamp
        ll = torch.where(torch.isfinite(ll), ll, torch.tensor(-1e6, device=device, dtype=ll.dtype))

        # Negative log-likelihood (mean over batch)
        loss = - ll.mean()
        return loss


def nll_loss(hazards, S, Y, c, alpha=0.4, eps=1e-7):
    """核心负对数似然损失计算（内部使用，无需直接调用）"""
    batch_size = len(Y)
    Y = Y.view(batch_size, 1)  # 离散时间bin索引 (batch_size, 1)
    c = c.view(batch_size, 1).float()  # 删失状态 (batch_size, 1)，1=删失，0=事件发生
    
    if S is None:
        # 生存函数S(t) = 累积乘积(1 - 风险率h(t))
        S = torch.cumprod(1 - hazards, dim=1)  # (batch_size, num_bins)
    
    # 填充S(-1) = 1（定义：所有患者在时间<0时均存活）
    S_padded = torch.cat([torch.ones_like(c), S], 1)  # (batch_size, num_bins + 1)
    
    # 未删失样本损失：-log(S(Y) * h(Y))
    uncensored_loss = -(1 - c) * (
        torch.log(torch.gather(S_padded, 1, Y).clamp(min=eps)) +  # S(Y)
        torch.log(torch.gather(hazards, 1, Y).clamp(min=eps))     # h(Y)
    )
    
    # 删失样本损失：-log(S(Y+1))
    censored_loss = -c * torch.log(
        torch.gather(S_padded, 1, Y + 1).clamp(min=eps)  # S(Y+1)
    )
    
    # 混合损失（alpha控制未删失样本权重）
    loss = (1 - alpha) * (censored_loss + uncensored_loss) + alpha * uncensored_loss
    return loss.mean()


class NLLSurvLoss(object):
    """
    适配原始生存数据的负对数似然损失
    
    输入：
    - pred: 模型输出的风险率 (hazards)，形状为 (batch_size, num_bins)
            其中num_bins是时间区间的数量，每个元素h[i]表示在第i个区间发生事件的概率
    - duration: 连续生存时间（原始输入），形状为 (batch_size,) 或 (batch_size, 1)
    - event: 事件指示器（原始输入），形状同上，1=发生事件，0=删失
    - time_bins: 时间区间划分，例如 [180, 365, 540] 表示区间为
                 [0,180), [180,365), [365,540), [540, inf)，对应bin索引0,1,2,3
    
    输出：
    - 损失值（标量）
    """
    def __init__(self, time_bins, alpha=0.15):
        self.alpha = alpha
        self.time_bins = torch.tensor(time_bins, dtype=torch.float32)  # 时间区间边界
        self.num_bins = len(time_bins) + 1  # 区间数量 = 边界数 + 1（最后一个区间是[last_bin, inf)）
    
    def _duration_to_bin(self, duration):
        """将连续生存时间转换为离散区间bin索引"""
        # 确保duration为张量且形状正确
        if isinstance(duration, np.ndarray):
            duration = torch.tensor(duration, dtype=torch.float32)
        duration = duration.view(-1)  # 展平为 (batch_size,)
        
        # 计算每个duration属于哪个区间（bin索引）
        # 例如：duration=200，time_bins=[180,365] → 属于[180,365) → bin=1
        bin_indices = torch.zeros_like(duration, dtype=torch.long)
        for i, bin_edge in enumerate(self.time_bins):
            bin_indices += (duration >= bin_edge).long()
        
        return bin_indices  # (batch_size,)
    
    def __call__(self, pred, duration, event, S=None):
        """
        计算损失主函数
        
        参数：
        - pred: 模型输出的风险率 (hazards)，形状 (batch_size, num_bins)
        - duration: 连续生存时间（原始输入）
        - event: 事件指示器（1=事件发生，0=删失）
        - S: 可选，预计算的生存函数，形状 (batch_size, num_bins)。若为None则自动计算
        """
        # 1. 转换事件指示器为删失状态c（1=删失，0=事件发生）
        if isinstance(event, np.ndarray):
            event = torch.tensor(event, dtype=torch.float32)
        c = 1 - event  # c=1表示删失，c=0表示未删失
        
        # 2. 将连续时间duration转换为离散bin索引Y
        Y = self._duration_to_bin(duration)
        
        # 3. 检查风险率维度是否与区间数量匹配
        if pred.shape[1] != self.num_bins:
            raise ValueError(
                f"风险率pred的列数必须等于时间区间数num_bins={self.num_bins}，"
                f"但实际为{pred.shape[1]}"
            )
        
        # 4. 计算损失
        return nll_loss(
            hazards=pred, 
            S=S, 
            Y=Y, 
            c=c, 
            alpha=self.alpha
        )





class MUltilabelCELoss(torch.nn.Module):
    def __init__(self, class_num, weight=None, reduction='mean',ifsigmoid=True):
        super(MUltilabelCELoss, self).__init__()
        self.class_num = class_num
        self.weight = weight
        self.reduction = reduction
        if self.weight is not None:
            self.weight = torch.tensor(self.weight, dtype=torch.float32)
        self.ifsigmoid = ifsigmoid
    def forward(self, predict, target):
        device = predict.device
        if self.weight is not None:
            self.weight = self.weight.to(device)
        else:
            self.weight = torch.ones(self.class_num, dtype=torch.float32).to(device)
        ### pred ： batchsize, class_num
        ### label ： batchsize, class_num
        ### 对于多标签分类，使用sigmoid激活函数，然后依次对每个标签计算二元交叉熵损失
        target = target.float()
        if self.ifsigmoid:
            predict = torch.sigmoid(predict)
        
        for i in range(self.class_num):
            if i == 0:
                loss = F.binary_cross_entropy_with_logits(predict[:, i], target[:, i], weight=self.weight[i], reduction=self.reduction)
            else:
                loss += F.binary_cross_entropy_with_logits(predict[:, i], target[:, i], weight=self.weight[i], reduction=self.reduction)
        
        if self.reduction == 'mean':
            loss = loss / self.class_num
        elif self.reduction == 'sum':
            loss = loss
        else:
            raise ValueError("Reduction must be 'mean' or 'sum'")

        return loss

class MultiCEFocalLoss(torch.nn.Module):
    def __init__(self, class_num, gamma=2, alpha=None, reduction='mean'):
        super(MultiCEFocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.class_num = class_num

    def forward(self, predict, target):
        device = predict.device
        pt = F.softmax(predict, dim=1)
        class_mask = F.one_hot(target, self.class_num)
        ids = target.view(-1, 1)
        self.alpha = self.alpha.to(device)
        alpha = self.alpha[ids.data.view(-1)]
        probs = (pt * class_mask).sum(1).view(-1, 1)
        log_p = probs.log()
        loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss
    

# class CoxLoss(nn.Module):
#     """
#     Cox partial log-likelihood loss for survival analysis.

#     Args:
#         ties (str): How to handle ties at identical event times.
#             - 'breslow' (default): Breslow approximation
#             - 'efron' : Efron approximation (more accurate when ties are frequent)
#         reduction (str): 'mean' (default), 'sum', or 'none'.

#     Forward inputs:
#         pred      : (N,) float tensor, model risk scores (higher -> higher hazard)
#         durations : (N,) float tensor, time-to-event or censoring time
#         events    : (N,) float/bool tensor, 1 = event observed, 0 = censored

#     Returns:
#         loss: scalar (if reduction != 'none') or (G,) if you pass a grouping mask (not used here).

#     Notes:
#         - We sort samples by durations DESC so that the risk set for index i is {0..i}.
#         - For ties:
#             Breslow:  sum_{groups g} [ sum(pred_events_g) - m_g * log(R_g) ]
#             Efron  :  sum_{groups g} [ sum(pred_events_g) - sum_{l=0}^{m_g-1} log(R_g - l/m_g * D_g) ]
#             where:
#                 m_g = number of events in tie group g,
#                 R_g = risk-set sum exp(pred) at the last index of group g,
#                 D_g = sum exp(pred) over the events in group g.
#         - The final loss is the NEGATIVE partial log-likelihood.
#     """

#     def __init__(self, ties: str = 'breslow', reduction: str = 'mean'):
#         super().__init__()
#         assert ties in ('breslow', 'efron')
#         assert reduction in ('mean', 'sum', 'none')
#         self.ties = ties
#         self.reduction = reduction

#     @staticmethod
#     def _unique_consecutive_with_counts(x: torch.Tensor):
#         # x is 1D sorted tensor; returns values, counts, last_indices
#         # Example: x = [5,5,4,4,4,3] (already sorted desc) ->
#         # vals=[5,4,3], counts=[2,3,1], last_idx=[1,4,5]
#         neq = torch.ones_like(x, dtype=torch.bool)
#         neq[1:] = x[1:] != x[:-1]
#         vals = x[neq]
#         counts = torch.diff(torch.nonzero(neq, as_tuple=False).flatten(), prepend=torch.tensor([0], device=x.device))
#         last_idx = torch.nonzero(neq, as_tuple=False).flatten() - 1
#         last_idx[0] = counts[0] - 1
#         for i in range(1, len(counts)):
#             last_idx[i] = last_idx[i-1] + counts[i]
#         return vals, counts, last_idx

#     def forward(self, pred: torch.Tensor, durations: torch.Tensor, events = None) -> torch.Tensor:
#         if events is None:
#             durations, events = durations[:,0],durations[:,1]
#         pred = pred.reshape(-1).to(dtype=torch.float32)
#         durations = durations.reshape(-1).to(dtype=torch.float32)
#         events = events.reshape(-1).to(dtype=torch.float32)

#         # Sort by time DESC so risk set for i is [0..i]
#         order = torch.argsort(durations, descending=True)
#         y = pred[order]
#         e = events[order]
#         t = durations[order]

#         # Precompute cumulative sums of exp(y)
#         # Use log-sum-exp stability by exponentiating y, which is safe under FP32 typically; y一般在[-20,20]范围
#         exp_y = torch.exp(y)
#         cum_exp_y = torch.cumsum(exp_y, dim=0)

#         # Group consecutive equal times (ties by time); only groups with events>0 contribute
#         # We'll find runs of equal t
#         # (values, counts, last_idx)
#         vals, counts, last_idx = self._unique_consecutive_with_counts(t)

#         total_ll = y.new_zeros(())  # scalar

#         # We also need cumulative sums to get sum(pred) over event individuals within a tie group:
#         # Build helpers: cumulative sums of y*e and exp_y*e to slice ranges quickly
#         ye = y * e
#         exp_ye = exp_y * e
#         csum_ye = torch.cumsum(ye, dim=0)
#         csum_exp_ye = torch.cumsum(exp_ye, dim=0)

#         start = 0
#         for k in range(len(vals)):
#             end = last_idx[k].item()         # inclusive
#             m = int(torch.sum(e[start:end+1]).item())  # number of events in this tie group
#             if m == 0:
#                 start = end + 1
#                 continue

#             # sums over events in group
#             sum_pred_events = (csum_ye[end] - (csum_ye[start-1] if start > 0 else 0.0))
#             D_g = (csum_exp_ye[end] - (csum_exp_ye[start-1] if start > 0 else 0.0))

#             # risk-set sum at the last index of the group
#             R_g = cum_exp_y[end]

#             if self.ties == 'breslow':
#                 # L_g = sum(pred_events) - m * log(R_g)
#                 ll_g = sum_pred_events - m * torch.log(R_g)
#             else:  # Efron
#                 # L_g = sum(pred_events) - sum_{l=0}^{m-1} log(R_g - l/m * D_g)
#                 # construct the m terms at once
#                 l = torch.arange(m, device=y.device, dtype=y.dtype)
#                 denom_terms = R_g - (l / m) * D_g
#                 # guard against numerical issues
#                 denom_terms = torch.clamp(denom_terms, min=1e-12)
#                 ll_g = sum_pred_events - torch.sum(torch.log(denom_terms))

#             total_ll = total_ll + ll_g
#             start = end + 1

#         # Negative partial log-likelihood
#         # Normalize by number of events to make scale comparable across batches
#         n_events = e.sum().clamp_min(1.0)
#         loss = - total_ll / n_events

#         if self.reduction == 'sum':
#             loss = loss * n_events
#         elif self.reduction == 'none':
#             # Here we only computed a scalar; if you truly need per-sample loss,
#             # you'd have to expand the derivation. Most users want a scalar.
#             loss = loss.unsqueeze(0)

#         return loss
class CoxLoss(nn.Module):
    """
    修复了梯度计算问题的Cox部分似似然损失函数，用于生存分析。
    确保计算图完整，避免出现梯度缺失问题。
    """
    def __init__(self, ties: str = 'breslow', reduction: str = 'mean'):
        super().__init__()
        assert ties in ('breslow', 'efron'), "ties必须是'breslow'或'efron'"
        assert reduction in ('mean', 'sum', 'none'), "reduction必须是'mean'、'sum'或'none'"
        self.ties = ties
        self.reduction = reduction

    @staticmethod
    def _unique_consecutive_with_counts(x: torch.Tensor):
        # 识别连续相同的值并计算计数和最后索引
        neq = torch.ones_like(x, dtype=torch.bool)
        neq[1:] = x[1:] != x[:-1]
        vals = x[neq]
        counts = torch.diff(torch.nonzero(neq, as_tuple=False).flatten(), prepend=torch.tensor([0], device=x.device))
        last_idx = torch.nonzero(neq, as_tuple=False).flatten() - 1
        last_idx[0] = counts[0] - 1
        for i in range(1, len(counts)):
            last_idx[i] = last_idx[i-1] + counts[i]
        return vals, counts, last_idx

    def forward(self, pred: torch.Tensor, durations: torch.Tensor, events = None) -> torch.Tensor:
        # 确保输入张量保留梯度
        pred = pred.reshape(-1).to(dtype=torch.float32).requires_grad_(True)
        if events is None:
            # 处理同时包含持续时间和事件的输入格式
            durations, events = durations[:,0], durations[:,1]
        
        durations = durations.reshape(-1).to(dtype=torch.float32, device=pred.device)
        events = events.reshape(-1).to(dtype=torch.float32, device=pred.device)

        # 检查是否有事件发生，避免全为删失的情况
        if events.sum() == 0:
            # 添加微小扰动避免梯度计算问题
            events = events + 1e-8
            # 输出警告但继续计算
            if self.training:
                print("警告: 批次中没有事件发生，可能导致梯度计算不稳定")

        # 按时间降序排序，使风险集为[0..i]
        order = torch.argsort(durations, descending=True)
        y = pred[order]  # 风险分数
        e = events[order]  # 事件指示器
        t = durations[order]  # 持续时间

        # 预计算exp(y)的累积和，使用clamp避免数值溢出
        exp_y = torch.exp(torch.clamp(y, min=-20, max=20))  # 限制数值范围
        cum_exp_y = torch.cumsum(exp_y, dim=0)

        # 分组处理相同时间点（时间结）
        vals, counts, last_idx = self._unique_consecutive_with_counts(t)

        total_ll = y.new_zeros(1, requires_grad=True)  # 确保损失张量有梯度

        # 准备累积和计算辅助变量
        ye = y * e
        exp_ye = exp_y * e
        csum_ye = torch.cumsum(ye, dim=0)
        csum_exp_ye = torch.cumsum(exp_ye, dim=0)

        start = 0
        for k in range(len(vals)):
            end = last_idx[k].item()  # 当前组的最后索引
            m = int(torch.sum(e[start:end+1]).item())  # 当前组的事件数
            
            if m == 0:
                start = end + 1
                continue

            # 计算当前组的事件预测值总和
            sum_pred_events = csum_ye[end] - (csum_ye[start-1] if start > 0 else 0.0)
            D_g = csum_exp_ye[end] - (csum_exp_ye[start-1] if start > 0 else 0.0)
            R_g = cum_exp_y[end]  # 当前组的风险集总和

            # 防止除零错误
            R_g = torch.clamp(R_g, min=1e-12)
            
            if self.ties == 'breslow':
                # Breslow近似处理时间结
                ll_g = sum_pred_events - m * torch.log(R_g)
            else:  # Efron近似
                # Efron近似处理时间结（更精确但计算成本更高）
                l = torch.arange(m, device=y.device, dtype=y.dtype)
                denom_terms = R_g - (l / m) * D_g
                denom_terms = torch.clamp(denom_terms, min=1e-12)  # 防止数值问题
                ll_g = sum_pred_events - torch.sum(torch.log(denom_terms))

            total_ll = total_ll + ll_g
            start = end + 1

        # 负的部分似然（因为我们要最小化损失）
        n_events = e.sum().clamp_min(1.0)  # 避免除以零
        loss = -total_ll / n_events

        # 根据reduction参数处理损失
        if self.reduction == 'sum':
            loss = loss * n_events
        elif self.reduction == 'none':
            loss = loss.squeeze()

        # 确保返回的损失张量有梯度
        return loss.requires_grad_(True)


# # 新增：计算多个时间点的生存状态预测AUC
# def calculate_timepoint_auc(risk_scores, durations, events, time_points):
#     """
#     计算指定时间点的生存状态预测AUC
    
#     参数:
#     - risk_scores: 模型输出的风险分数
#     - durations: 生存时间
#     - events: 事件发生指示器(1=发生事件,0=删失)
#     - time_points: 要评估的时间点列表
    
#     返回:
#     - 每个时间点的AUC值字典
#     """
#     results = {}
    
#     for time_point in time_points:
#         # 确定在该时间点的状态: 1=已发生事件, 0=存活
#         status = np.where((durations <= time_point) & (events == 1), 1, 0)
        
#         # 过滤掉在该时间点前被删失的样本
#         valid_indices = np.where(~((durations <= time_point) & (events == 0)))[0]
        
#         if len(valid_indices) < 2:
#             results[time_point] = np.nan
#             continue
            
#         # 计算AUC
#         try:
#             auc = roc_auc_score(
#                 status[valid_indices], 
#                 risk_scores[valid_indices]
#             )
#             results[time_point] = auc
#         except:
#             results[time_point] = np.nan
    
#     return results


from lifelines import KaplanMeierFitter
import numpy as np
from sklearn.metrics import roc_auc_score

def calculate_timepoint_auc(risk_scores, durations, events, time_points, is_train=True, train_km_data=None):
    """
    计算指定时间点的生存状态预测AUC（基于HR分组的Kaplan-Meier生存曲线）
    
    参数:
    - risk_scores: 模型输出的HR风险值
    - durations: 生存时间
    - events: 事件发生指示器(1=发生事件,0=删失)
    - time_points: 要评估的时间点列表
    - is_train: 是否为训练集（训练集需构建生存曲线，测试集直接使用训练集的曲线）
    - train_km_data: 训练集构建的生存曲线数据（仅测试集需要，格式为(hr_cutoff, km_high, km_low)）
    
    返回:
    - 每个时间点的AUC值字典 + 训练集的生存曲线数据（仅当is_train=True时）
    """
    if train_km_data is None:
        print("train_km_data is None")
        return {}, None
    
    results = {}
    hr_scores = risk_scores.flatten()  # 确保HR是一维数组
    n_samples = len(hr_scores)
    
    # --------------------------
    # 1. 处理训练集：构建生存曲线
    # --------------------------
    if is_train:
        # 按HR中位数分组（高风险/低风险）
        hr_cutoff = np.median(hr_scores)  # 中位数作为分组阈值
        high_risk_mask = hr_scores >= hr_cutoff
        low_risk_mask = ~high_risk_mask
        
        # 分别为高低风险组构建Kaplan-Meier生存曲线
        km_high = KaplanMeierFitter()
        km_high.fit(
            durations[high_risk_mask], 
            event_observed=events[high_risk_mask],
            label='High Risk'
        )
        
        km_low = KaplanMeierFitter()
        km_low.fit(
            durations[low_risk_mask], 
            event_observed=events[low_risk_mask],
            label='Low Risk'
        )
        
        # 保存训练集的生存曲线数据（供测试集使用）
        train_km_data = (hr_cutoff, km_high, km_low)
    
    # --------------------------
    # 2. 预测生存状态并计算AUC
    # --------------------------
    if not is_train and train_km_data is None:
        raise ValueError("测试集计算必须提供训练集的生存曲线数据(train_km_data)")
    
    # 从训练集数据中获取分组阈值和生存曲线（测试集用）
    hr_cutoff, km_high, km_low = train_km_data if not is_train else train_km_data
    
    for time_point in time_points:
        # 筛选有效样本：排除在time_point前删失的样本（无法判断真实状态）
        valid_mask = ~((durations <= time_point) & (events == 0))
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_indices) < 2:  # 样本量不足，无法计算AUC
            results[time_point] = np.nan
            continue
        
        # --------------------------
        # a. 确定真实生存状态
        # --------------------------
        # 真实状态：1=在time_point前发生事件；0=存活到time_point及以后
        true_status = np.where(
            (durations[valid_indices] <= time_point) & (events[valid_indices] == 1),
            1, 0
        )
        
        # --------------------------
        # b. 预测生存状态（基于训练集的生存曲线）
        # --------------------------
        valid_hr = hr_scores[valid_indices]
        
        # 对每个有效样本，根据HR分组选择对应的生存曲线
        pred_surv_prob = []
        for hr in valid_hr:
            if hr >= hr_cutoff:  # 高风险组：用训练集高风险组的生存曲线
                # 若时间点超出曲线范围，取最后一个生存概率
                if time_point > km_high.event_table.index.max():
                    prob = km_high.survival_function_.iloc[-1].values[0]
                else:
                    prob = km_high.predict(time_point)
            else:  # 低风险组：用训练集低风险组的生存曲线
                if time_point > km_low.event_table.index.max():
                    prob = km_low.survival_function_.iloc[-1].values[0]
                else:
                    prob = km_low.predict(time_point)
            pred_surv_prob.append(prob)
        
        # 生存概率 <= 0.5 预测为事件发生（1），否则预测为存活（0）
        pred_status = np.where(np.array(pred_surv_prob) <= 0.5, 1, 0)
        
        # --------------------------
        # c. 计算AUC
        # --------------------------
        try:
            auc = roc_auc_score(true_status, pred_status)
            results[time_point] = round(auc, 4)
        except:
            results[time_point] = np.nan
    
    # 训练集返回结果+生存曲线数据，测试集只返回结果
    return (results, train_km_data) if is_train else results

from lifelines import KaplanMeierFitter
import numpy as np
from sklearn.metrics import roc_auc_score

def calculate_timepoint_auc_weighted_km(risk_scores, durations, events, time_points, is_train=True, train_surv_data=None):
    """
    基于加权Kaplan-Meier的时间点AUC计算（连续HR值，无需分组）
    
    参数:
    - risk_scores: 模型输出的HR风险值
    - durations: 生存时间
    - events: 事件发生指示器(1=发生事件,0=删失)
    - time_points: 要评估的时间点列表
    - is_train: 是否为训练集（训练集需拟合加权生存曲线）
    - train_surv_data: 训练集的加权生存曲线数据（测试集使用）
    
    返回:
    - 每个时间点的AUC值字典 + 训练集生存曲线数据（仅is_train=True时）
    """
    results = {}
    hr_scores = risk_scores.flatten()
    n_samples = len(hr_scores)
    
    # --------------------------
    # 1. 训练集：拟合加权KM生存曲线
    # --------------------------
    if is_train:
        # 权重计算：HR越高，权重越大（用指数化HR避免负值影响）
        weights = np.exp(hr_scores) / np.exp(hr_scores).mean()  # 标准化权重
        # 假设权重变量为 weights，添加转换逻辑
        weights = weights.astype(np.int64)  # 或 np.uint64，确保无负数

        
        # 拟合加权Kaplan-Meier曲线
        wkm = KaplanMeierFitter()
        wkm.fit(
            durations, 
            event_observed=events,
            weights=weights,  # 传入权重
            label='Weighted Survival'
        )
        train_surv_data = wkm  # 保存训练集的加权生存曲线
    
    # --------------------------
    # 2. 预测生存状态并计算AUC
    # --------------------------
    if not is_train and train_surv_data is None:
        raise ValueError("测试集需传入训练集的加权生存曲线数据(train_surv_data)")
    
    # 训练集的生存曲线（测试集使用）
    wkm = train_surv_data if not is_train else train_surv_data
    
    for time_point in time_points:
        # 筛选有效样本（排除time_point前删失的样本）
        valid_mask = ~((durations <= time_point) & (events == 0))
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_indices) < 2:
            results[time_point] = np.nan
            continue
        
        # a. 真实生存状态
        true_status = np.where(
            (durations[valid_indices] <= time_point) & (events[valid_indices] == 1),
            1, 0
        )
        
        # b. 预测生存状态（基于HR和加权生存曲线）
        valid_hr = hr_scores[valid_indices]
        # 从训练集加权曲线获取基准生存概率
        if time_point > wkm.event_table.index.max():
            base_surv = wkm.survival_function_.iloc[-1].values[0]
        else:
            base_surv = wkm.predict(time_point)
        
        # 用HR调整基准生存概率（HR越高，生存概率越低）
        # 公式：pred_surv = base_surv ^ (HR / mean_HR)，确保HR=mean时与基准一致
        mean_hr_train = np.mean(hr_scores) if is_train else np.mean(train_surv_data.event_table.index)
        pred_surv_prob = base_surv **(valid_hr / mean_hr_train)
        
        # 生存概率 <= 0.5 预测为事件发生（1）
        pred_status = np.where(pred_surv_prob <= 0.5, 1, 0)
        
        # c. 计算AUC
        try:
            auc = roc_auc_score(true_status, pred_status)
            results[time_point] = round(auc, 4)
        except:
            results[time_point] = np.nan
    
    return (results, train_surv_data) if is_train else results


from lifelines import KaplanMeierFitter
import numpy as np
from sklearn.metrics import roc_auc_score

def gaussian_kernel(hr, target_hr, bandwidth=0.5):
    """高斯核函数：衡量样本HR与目标HR的相似度（用于加权）"""
    return np.exp(-0.5 * ((hr - target_hr) / bandwidth)** 2)


from lifelines import KaplanMeierFitter
import numpy as np
from sklearn.metrics import roc_auc_score

def gaussian_kernel(hr, target_hr, bandwidth=0.5):
    """高斯核函数：衡量样本HR与目标HR的相似度（用于加权）"""
    return np.exp(-0.5 * ((hr - target_hr) / bandwidth) **2)

def calculate_timepoint_auc_kernel_surv(risk_scores, durations, events, time_points, is_train=True, train_kernel_data=None):
    """
    基于核密度平滑的时间点AUC计算（连续HR值，精细建模）
    修复：权重转换为整数类型，避免lifelines内部类型转换错误
    
    参数:
    - risk_scores: 模型输出的HR风险值
    - durations: 生存时间（需为正值）
    - events: 事件发生指示器(1=发生事件,0=删失)
    - time_points: 要评估的时间点列表
    - is_train: 是否为训练集（需保存原始数据用于核平滑）
    - train_kernel_data: 训练集原始数据（测试集使用，格式为(hr, durations, events)）
    
    返回:
    - 每个时间点的AUC值字典 + 训练集原始数据（仅is_train=True时）
    """
    results = {}
    hr_scores = risk_scores.flatten()
    n_samples = len(hr_scores)
    
    # 检查生存时间是否为正值（避免KM拟合错误）
    if np.any(durations <= 0):
        raise ValueError("生存时间(durations)必须全部为正值")
    
    # --------------------------
    # 1. 训练集：保存原始数据（用于核平滑）
    # --------------------------
    if is_train:
        # 保存训练集原始数据（HR、生存时间、事件）
        train_kernel_data = (hr_scores.copy(), durations.copy(), events.copy())
    else:
        # 测试集必须传入训练集数据
        if train_kernel_data is None:
            raise ValueError("测试集计算必须提供训练集原始数据(train_kernel_data)")
        train_hr, train_dur, train_evt = train_kernel_data
        # 检查训练集数据有效性
        if np.any(train_dur <= 0):
            raise ValueError("训练集生存时间(train_dur)必须全部为正值")
    
    # --------------------------
    # 2. 预测生存状态并计算AUC
    # --------------------------
    for time_point in time_points:
        # 筛选有效样本：排除在time_point前删失的样本（无法判断真实状态）
        valid_mask = ~((durations <= time_point) & (events == 0))
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_indices) < 2:  # 样本量不足，无法计算AUC
            results[time_point] = np.nan
            continue
        
        # a. 确定真实生存状态
        true_status = np.where(
            (durations[valid_indices] <= time_point) & (events[valid_indices] == 1),
            1, 0
        )
        
        # b. 预测生存状态（核平滑计算生存概率）
        valid_hr = hr_scores[valid_indices]
        pred_surv_prob = []
        
        for hr in valid_hr:
            # 对每个测试样本的HR，用训练集数据进行核平滑
            if is_train:
                # 训练集使用自身数据计算
                current_hr = hr_scores
                current_dur = durations
                current_evt = events
            else:
                # 测试集使用训练集数据计算
                current_hr = train_hr
                current_dur = train_dur
                current_evt = train_evt
            
            # 1) 计算训练集每个样本与当前HR的相似度权重（高斯核）
            weights = gaussian_kernel(current_hr, hr)
            weights /= weights.sum()  # 归一化权重（确保和为1）
            
            # 2) 权重转换为整数（修复核心：避免浮点数转换错误）
            # 缩放因子：平衡精度和整数大小（可根据样本量调整）
            scale_factor = 10**6  # 1e6放大倍数
            weights_int = (weights * scale_factor).astype(np.int64)
            
            # 避免全零权重（极端情况下的保护机制）
            if weights_int.sum() == 0:
                weights_int = np.ones_like(weights_int, dtype=np.int64)
            
            # 3) 用加权KM计算当前HR在time_point的生存概率
            km = KaplanMeierFitter()
            try:
                km.fit(
                    current_dur, 
                    event_observed=current_evt, 
                    weights=weights_int  # 传入整数权重
                )
            except Exception as e:
                print(f"KM拟合失败（HR={hr}, time_point={time_point}）: {str(e)}")
                pred_surv_prob.append(0.5)  # 失败时默认0.5（随机猜测）
                continue
            
            # 获取时间点的生存概率（处理超出曲线范围的情况）
            if time_point > km.event_table.index.max():
                # 超出最大生存时间，取最后一个生存概率
                prob = km.survival_function_.iloc[-1].values[0]
            else:
                # 直接预测指定时间点
                prob = km.predict(time_point)
            
            pred_surv_prob.append(prob)
        
        # 生存概率 <= 0.5 预测为事件发生（1），否则为存活（0）
        pred_status = np.where(np.array(pred_surv_prob) <= 0.5, 1, 0)
        
        # c. 计算AUC
        try:
            auc = roc_auc_score(true_status, pred_status)
            results[time_point] = round(auc, 4)
        except Exception as e:
            print(f"AUC计算失败（time_point={time_point}）: {str(e)}")
            results[time_point] = np.nan
    
    # 训练集返回结果+原始数据，测试集只返回结果
    return (results, train_kernel_data) if is_train else results


def main():
    def create_args_yaml(args):
        num = 0
        while True:
            savepath = os.path.join(args.logdir, f"args{num}.yaml")
            if os.path.exists(savepath):
                num += 1
            else:
                break
        with open(savepath, "w") as f:
            yaml.dump(vars(args), f, default_flow_style=False)

    def save_ckp(state, checkpoint_dir):
        torch.save(state, checkpoint_dir)

    def metric_func(preds, labels, args=None):
        if args.metric_func == 'accuracy' or args.metric_func == None:
            return accuracy_score(labels, np.argmax(preds, axis=1))
        elif args.metric_func == 'mAP' or args.metric_func == 'map':
            return average_precision_score(labels, preds)
        # 新增：Cox模型的C-index作为指标
        elif args.metric_func == 'cindex' and args.loss.lower() in ['cox','cox_log_loss','coxloss','nll']:
            # preds是风险分数, labels是(持续时间, 事件)的元组
            durations, events = labels[:, 0], labels[:, 1]
            try:
                return concordance_index(durations, -preds, events)  # 注意这里用负号，因为风险分数越高预后越差
            except:
                return 0.5  # 如果计算失败，返回随机水平
        elif args.metric_func == 'cindex' and args.loss.lower() in ['pchazard']:
            # preds是风险分数, labels是(持续时间, 事件)的元组
            durations, events = labels[:, 0], labels[:, 1]
            risk, mask = preds_to_risk(preds, mode='cumhaz')
            cidx = concordance_index(durations, risk, events)
            return cidx
        else:
            raise ValueError("Invalid metric function. Choose 'accuracy' or 'mAP' or 'cindex'.")

    def auc_func(preds, labels):
        try:
            preds = preds.cpu().float().detach().numpy()
        except:
            pass
        try:
            labels = labels.cpu().float().detach().numpy()
        except:
            pass
        labels = np.squeeze(labels)
        preds = np.squeeze(preds)
        return roc_auc_score(labels, preds)

    def calculate_auc_and_plot_confusion_matrix(pred_array, label_array, save_path=None):
        auc = roc_auc_score(label_array, pred_array)
        cm = confusion_matrix(label_array, pred_array.round())
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f'AUC: {auc}')
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        return auc

    def plot_line_graph(data_list, save_path):
        plt.figure(figsize=(len(data_list), 5))
        plt.plot(data_list, marker='o')
        plt.title('Line Graph')
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.grid(True)
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

    def label2onehot(label):
        try:
            label = label.cpu().detach().numpy()
        except:
            pass
        label = np.squeeze(label)
        label = np.eye(args.num_classes)[label]
        return label

    def save_tensor_2_nii(tensor, save_path):
        assert save_path.endswith('.nii.gz')
        tensor_save = tensor.clone().detach().cpu().numpy()
        tensor_save = np.squeeze(tensor_save)
        tensor_save = sitk.GetImageFromArray(tensor_save)
        sitk.WriteImage(tensor_save, save_path)

    def save_results(pred_one_epoch, label_one_epoch, imgPathOneEpoch, epoch, mode, args, time_auc_results=None):
    # 检查并调整预测结果形状
        print(f"预测结果形状: {pred_one_epoch.shape}, 标签长度: {len(label_one_epoch)}")
        
        # 确保预测结果与标签长度匹配
        if len(pred_one_epoch) != len(label_one_epoch):
            # 处理多GPU训练导致的结果重复或拆分问题
            if len(pred_one_epoch) == 2 * len(label_one_epoch):
                # 如果是双卡训练可能导致的重复
                pred_one_epoch = pred_one_epoch[:len(label_one_epoch)]
            else:
                # 其他情况尝试调整形状
                pred_one_epoch = pred_one_epoch.flatten()[:len(label_one_epoch)]
    
        # 对于Cox模型，label_one_epoch包含持续时间和事件
        if args.loss.lower() in ['cox','cox_log_loss','coxloss','nll','pchazard']:
            columns = ['path', 'pred_risk', 'duration', 'event']
            result_df = pd.DataFrame(columns=columns)
            result_df['path'] = imgPathOneEpoch
            result_df['pred_risk'] = pred_one_epoch.flatten()
            result_df['duration'] = label_one_epoch[:, 0]
            result_df['event'] = label_one_epoch[:, 1]
            
            # 计算C-index
            try:
                if args.loss.lower() in ['pchazard']:
                    cindex = concordance_index(
                        label_one_epoch[:, 0],  # 持续时间
                        pred_one_epoch.flatten(),  # 风险分数（负号因为C-index期望风险高的样本事件发生早）
                        label_one_epoch[:, 1]   # 事件指示器
                    )
                else:
                    cindex = concordance_index(
                        label_one_epoch[:, 0],  # 持续时间
                        -pred_one_epoch.flatten(),  # 风险分数（负号因为C-index期望风险高的样本事件发生早）
                        label_one_epoch[:, 1]   # 事件指示器
                    )
            except:
                cindex = np.nan
        else:
            columns = ['path'] + [f'pred{i}' for i in range(args.num_classes)] + ['label']
            result_df = pd.DataFrame(columns=columns)
            result_df['path'] = imgPathOneEpoch
            for i in range(args.num_classes):
                result_df[f'pred{i}'] = pred_one_epoch[:, i]
            result_df['label'] = label_one_epoch
            cindex = None

        result_path = args.save_path + f'{mode}_Result_epoch_{epoch}.xlsx'
        result_df.to_excel(result_path, index=False)

        # 计算并保存指标
        if args.loss.lower() in ['cox','cox_log_loss','coxloss','nll','hazard','pchazard']:
            # 对于Cox模型，使用C-index作为主要指标
            acc = cindex
            f1 = None
            pre = None
            rec = None
            cls_report = None
            cm = None
        else:
            y_pred = np.argmax(pred_one_epoch, axis=1)
            acc = accuracy_score(label_one_epoch, y_pred)
            f1 = f1_score(label_one_epoch, y_pred, average='weighted')
            pre = precision_score(label_one_epoch, y_pred, average='weighted')
            rec = recall_score(label_one_epoch, y_pred, average='weighted')
            cls_report = classification_report(label_one_epoch, y_pred, output_dict=True)
            cm = confusion_matrix(label_one_epoch, y_pred)

        report_path = args.save_path + f'{mode}_Report_epoch_{epoch}.xlsx'
        with pd.ExcelWriter(report_path) as writer:
            metrics_data = {'Accuracy/C-index': [acc]}
            if f1 is not None:
                metrics_data['F1-score'] = [f1]
                metrics_data['Precision'] = [pre]
                metrics_data['Recall'] = [rec]
            
            # 添加时间点AUC结果
            # if time_auc_results is not None:
            #     for time_point, auc in time_auc_results[0].items():
            #         metrics_data[f'AUC0 at {time_point}'] = [auc]
            #     for time_point, auc in time_auc_results[1].items():
            #         metrics_data[f'AUC1 at {time_point}'] = [auc]
            #     for time_point, auc in time_auc_results[2].items():
            #         metrics_data[f'AUC2 at {time_point}'] = [auc]
                    
            metrics_df = pd.DataFrame(metrics_data)
            metrics_df.to_excel(writer, sheet_name='Metrics', index=False)

            if cls_report is not None:
                cls_report_df = pd.DataFrame(cls_report).transpose()
                cls_report_df.to_excel(writer, sheet_name='Classification Report')

            if cm is not None:
                cm_df = pd.DataFrame(cm)
                cm_df.to_excel(writer, sheet_name='Confusion Matrix')

        return acc

    def train_cls(args, global_step, train_loader, val_best, scaler, scheduler=None):
        if accelerator.mixed_precision == 'fp16':
            accelerator_dtype = torch.float16
        elif accelerator.mixed_precision == 'bf16':
            accelerator_dtype = torch.bfloat16
        else:
            accelerator_dtype = torch.float32

        if args.freeze_feaExtractor >= 0:
            accelerator.print(model.module.iffreeze)

        accelerator.print(model, file=open(args.save_path + "model.txt", "a"))
        accelerator.print(args, file=open(args.save_path + "args.txt", "a"))

        if args.only_test:
            epoch = 0
            for idx, test_loader in enumerate(test_loaders):
                val_loss, val_metric, val_cindex, time_auc_results = validation_cls(args, test_loader, epoch, mode=f'test{idx+1}')
                accelerator.print(f"Test Set {idx+1} Loss: {val_loss}, C-index: {val_cindex}, Metric: {val_metric}")
                # if time_auc_results:
                #     for t, auc in time_auc_results.items():
                #         accelerator.print(f"  Time {t} AUC: {auc:.4f}")
                accelerator.print(f"Test Set {idx+1} time_auc_results: {time_auc_results}",
                                  file=open(args.save_path + "Metrics.txt", "a"))
            exit()
            return global_step, val_loss, val_best

        all_results = []

        for epoch in range(args.epochs):
            torch.cuda.empty_cache()
            model.train()

            for name, param in model.named_parameters():
                if accelerator.is_main_process:
                    writer.add_histogram(f'parameters/{name}', param.data, epoch)
                if param.grad is not None and accelerator.is_main_process:
                    writer.add_histogram(f'gradients/{name}', param.grad.data, epoch)

            if args.freeze_feaExtractor > 0:
                if epoch < args.freeze_feaExtractor and model.module.iffreeze == "false":
                    model.module.freeze_feaExtractor()
                    model.module.iffreeze = "all"
                elif epoch == args.freeze_feaExtractor:
                    model.module.unfreeze_feaExtractor()
                    model.module.iffreeze = "none"
                else:
                    accelerator.print(f"freeze_feaExtractor is {model.module.iffreeze} in epoch {epoch}")
            if args.freeze_feaExtractor == 0 and epoch == 0:
                model.module.freeze_feaExtractor()
                model.module.iffreeze = "all"

            pred_one_epoch = []
            label_one_epoch = []
            image_path_one_epoch = []
            loss_train = []
            
            # 用于Cox模型的额外存储
            durations_train = []
            events_train = []

            for step, batch in tqdm.tqdm(enumerate(train_loader), total=len(train_loader)):
                try:
                    t1 = time()
                    if args.model.lower() == 'densenetbigtwoforkssmall':
                        image = batch[1].to(accelerator.device, dtype=accelerator_dtype)
                    else:
                        image = batch[0].to(accelerator.device, dtype=accelerator_dtype)
                    
                    # 区分Cox模型和分类模型的标签处理
                    if args.loss.lower() in ['cox','cox_log_loss','coxloss','nll','pchazard']:
                        # Cox模型: 标签是(持续时间, 事件)
                        # 确保我们处理的是张量而不是列表
                        duration = torch.tensor(batch[-2][0], device=accelerator.device)
                        event = torch.tensor(batch[-2][1], device=accelerator.device)
                        print("duration:", duration, "event:", event)
                        label = (duration, event)

                    else:
                        # 分类模型
                        label = batch[-2].to(accelerator.device)
                        
                    
                    imgpath = batch[-1]
                    

                    if args.save_fea:
                        logits_per_image, fea = model(image)
                    else:
                        logits_per_image = model(image)
                    
                    if args.loss.lower() in ['nll']:
                        logits_per_image = F.sigmoid(logits_per_image)

                    # 计算损失
                    if args.loss.lower() in ['cox','cox_log_loss','coxloss','nll','pchazard']:
                        loss = loss_function(logits_per_image, duration, event)
                    else:
                        loss = loss_function(logits_per_image, label)
                    if args.loss.lower() in ['pchazard']:
                        logits_per_image,_ = preds_to_risk(logits_per_image)
                    accelerator.backward(loss)
                    if args.grad_clip:
                        accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    if scheduler:
                        scheduler.step()
                    optimizer.zero_grad()

                    if args.loss.lower() in ['cox','cox_log_loss','coxloss','nll','pchazard']:
                        # Cox模型: 标签是(持续时间, 事件)
                        # 确保我们处理的是张量而不是列表
                        durations_train.append(duration.cpu().numpy())
                        events_train.append(event.cpu().numpy())
                        label_one_epoch.append(np.stack([duration.cpu().numpy(), event.cpu().numpy()], axis=1))
                    else:
                        # 分类模型
                        label_one_epoch.append(label.cpu().numpy())
                    
                    imgpath = batch[-1]
                    image_path_one_epoch.append(imgpath)



                    loss_train.append(loss.item())
                    # Cox模型输出的是风险分数，不需要softmax
                    if args.loss.lower() not in ['cox','cox_log_loss','coxloss','nll','pchazard']:
                        logits_per_image = F.softmax(logits_per_image, dim=-1)
                    if args.loss.lower() not in ['pchazard']:
                        logits_per_image = logits_per_image.float().detach().cpu().numpy()
                    pred_one_epoch.append(logits_per_image)

                    accelerator.log({"train/loss_step": loss.item()}, step=global_step)
                    accelerator.print(f"Step: {global_step}, Loss: {loss.item()}, Time: {time() - t1}")
                    if accelerator.is_main_process:
                        writer.add_scalar("train/loss_step", np.mean(loss_train), global_step)

                    global_step += 1
                except Exception as e:
                    global_step += 1
                    print(e)
                    continue

            pred_one_epoch = np.concatenate(pred_one_epoch)
            label_one_epoch = np.concatenate(label_one_epoch)
            image_path_one_epoch = np.concatenate(image_path_one_epoch)

            # 计算时间点AUC（仅对Cox模型）
            time_auc_results = None
            if args.loss.lower() in ['cox','cox_log_loss','coxloss','nll']:
                durations = label_one_epoch[:, 0]
                events = label_one_epoch[:, 1]
                train_result = calculate_timepoint_auc(
                    pred_one_epoch.flatten(), 
                    durations, 
                    events, 
                    args.time_points,
                    is_train=True
                )
                time_auc_results, train_km_data = train_result
                args.train_km_data.append(train_km_data)
                train_result2 = calculate_timepoint_auc_weighted_km(
                    pred_one_epoch.flatten(), 
                    durations, 
                    events, 
                    args.time_points,
                    is_train=True
                )
                time_auc_results2, train_km_data = train_result2
                args.train_km_data.append(train_km_data)
                train_result3 = calculate_timepoint_auc_kernel_surv(
                    pred_one_epoch.flatten(), 
                    durations, 
                    events, 
                    args.time_points,
                    is_train=True
                )
                time_auc_results3, train_km_data = train_result3
                args.train_km_data.append(train_km_data)


            # 保存训练结果
            if args.loss.lower() in ['cox','cox_log_loss','coxloss','nll']:
                # 对于Cox模型，metric_func需要(持续时间, 事件)作为标签
                train_metric = metric_func(pred_one_epoch, label_one_epoch, args)
                train_acc = train_metric  # C-index作为主要指标
                all_trainauc = [time_auc_results, time_auc_results2, time_auc_results3]
            else:
                train_acc = save_results(pred_one_epoch, label_one_epoch, image_path_one_epoch, epoch, 'train', args)
                train_metric = metric_func(pred_one_epoch, label_one_epoch, args)

            accelerator.print(f"Step: {global_step}, Train C-index/Accuracy: {train_acc}")
            accelerator.print(f"Step: {global_step}, Train Metric: {train_metric}")

            if accelerator.is_main_process:
                writer.add_scalar("train/Accuracy/C-index", train_acc, epoch)
                writer.add_scalar("train/loss_total", np.mean(loss_train), epoch)
                writer.add_scalar("train/Metric", train_metric, epoch)
                accelerator.print(f"Epoch: {epoch}, Train C-index/Accuracy: {train_metric}",
                                  file=open(args.save_path + "Metrics.txt", "a"))
                # accelerator.print(f"Epoch: {epoch}, Train Metric: {train_metric}",
                                    # file=open(args.save_path + "Metrics.txt", "a"))
                
                # 记录时间点AUC
                if time_auc_results:
                    accelerator.print(f"Epoch: {epoch}, Train Time-point AUC: {all_trainauc}",
                                      file=open(args.save_path + "Metrics.txt", "a"))

            # Validate on all test sets
            val_results = []
            for idx, test_loader in enumerate(test_loaders):
                val_loss, val_metric, val_cindex, time_auc = validation_cls(args, test_loader, epoch, mode=f'test{idx+1}')
                val_results.append({
                    'test_set': f'test{idx+1}',
                    'val_loss': val_loss,
                    'val_cindex/accuracy': val_cindex,
                    'val_metric': val_metric,
                    'time_auc': time_auc
                })
                accelerator.print(f"Epoch: {epoch}, Test Set {idx+1} C-index/Accuracy: {val_cindex}",
                                  file=open(args.save_path + "Metrics.txt", "a"))
                accelerator.print(f"Epoch: {epoch}, Test Set {idx+1} Metric: {val_metric}",
                                  file=open(args.save_path + "Metrics.txt", "a"))
                
                if time_auc:
                    # for t, auc in time_auc.items():
                    #     accelerator.print(f"Epoch: {epoch}, Test Set {idx+1} AUC at {t}: {auc}",
                    #                       file=open(args.save_path + "Metrics.txt", "a"))
                    accelerator.print(f"Epoch: {epoch}, Test Set {idx+1} Time-point AUC: {time_auc}",
                                          file=open(args.save_path + "Metrics.txt", "a"))
                
                if accelerator.is_main_process:
                    writer.add_scalar(f"Test{idx+1}/C-index/Accuracy", val_cindex, epoch)
                    writer.add_scalar(f"Test{idx+1}/lossinval", val_loss, epoch)
                    writer.add_scalar(f"Test{idx+1}/Metric", val_metric, epoch)
                    if time_auc:
                        # for t, auc in time_auc.items():
                        #     writer.add_scalar(f"Test{idx+1}/AUC_at_{t}", auc, epoch)
                        accelerator.print(f"Epoch: {epoch}, Test Set {idx+1} Time-point AUC: {time_auc}",
                                          file=open(args.save_path + "Metrics.txt", "a"))

            # Use the metric from the first test set for checkpointing
            val_metric = val_results[0]['val_metric']
            if accelerator.is_main_process:
                writer.add_scalar("train/loss_total", np.mean(loss_train), global_step)
                writer.add_scalar("Test1/val_loss", val_results[0]['val_loss'], global_step)

                ifsaveckp0 = False
                ifsaveckp1 = False
                ifsaveckp2 = False
                if val_metric > args.max_result[0]:
                    args.max_result = [val_metric, args.max_result[1], args.max_result[2]]
                    ifsaveckp0 = True
                if val_metric > args.max_result[1]:
                    args.max_result = [args.max_result[0], val_metric, args.max_result[2]]
                    ifsaveckp1 = True
                if val_metric > args.max_result[2]:
                    args.max_result = [args.max_result[0], args.max_result[1], val_metric]
                    ifsaveckp2 = True

                checkpoint = {
                    "global_step": global_step,
                    "state_dict": accelerator.unwrap_model(model).state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "args": args
                }
                if args.saveallckp:
                    save_ckp(checkpoint, args.logdir + f"/model_epoch{epoch}.pt")
                    accelerator.print(f"Model saved! Epoch: {epoch}, Global Step: {global_step}")
                    
                if ifsaveckp0:
                    save_ckp(checkpoint, args.logdir + "/model_bestVal.pt")
                    accelerator.print(f"Model saved! Best Test1 Metric: {args.max_result}")
                    args.bind_num = 0
                if ifsaveckp1:
                    save_ckp(checkpoint, args.logdir + "/model_secondbestVal.pt")
                    accelerator.print(f"Model saved! Best Test1 Metric: {args.max_result}")
                    args.bind_num = 0
                if ifsaveckp2:
                    save_ckp(checkpoint, args.logdir + "/model_thirdbestVal.pt")
                    accelerator.print(f"Model saved! Best Test1 Metric: {args.max_result}")
                    args.bind_num = 0

                else:
                    args.bind_num += 1
                    accelerator.print(f"Model not saved! Best Test1 Metric: {args.max_result}")

            if args.bind_num == 3 and args.freeze_feaExtractor == 0:
                if model.module.iffreeze == "all":
                    model.module.unfreeze_last2block()
                    if not args.bind_lr:
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = 1e-3
                    args.bind_num = 0
                    model.module.iffreeze = "first3block"
                if model.module.iffreeze == "first3block":
                    model.module.unfreeze_feaExtractor()
                    if not args.bind_lr:
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = 1e-4
                    args.bind_num = 0
                    model.module.iffreeze = "none"

            # Save results for this epoch
            epoch_results = {
                'epoch': epoch,
                'train_cindex/accuracy': train_acc,
                'train_metric': train_metric,
                'train_loss': np.mean(loss_train),
            }
            for val_result in val_results:
                epoch_results[f"{val_result['test_set']}_cindex/accuracy"] = val_result['val_cindex/accuracy']
                epoch_results[f"{val_result['test_set']}_metric"] = val_result['val_metric']
                epoch_results[f"{val_result['test_set']}_loss"] = val_result['val_loss']
                # 添加时间点AUC
                # if val_result['time_auc']:
                #     for methodi in range(0,3):
                #         for t, auc in val_result['time_auc'][methodi].items():
                #             epoch_results[f"{val_result['test_set']}_method-{methodi}_auc_at_{t}"] = auc
                        
            all_results.append(epoch_results)

        # Save all results to a single Excel file
        all_results_df = pd.DataFrame(all_results)
        all_results_path = args.save_path + 'all_results.xlsx'
        all_results_df.to_excel(all_results_path, index=False)
        
        exit('Training completed!')
        return global_step, loss_train, val_best

    def validation_cls(args, test_loader, epoch=0, mode='val'):
        if accelerator.mixed_precision == 'fp16':
            accelerator_dtype = torch.float16
        elif accelerator.mixed_precision == 'bf16':
            accelerator_dtype = torch.bfloat16
        else:
            accelerator_dtype = torch.float32

        model.eval()
        loss_val = []
        pred_one_epoch = []
        label_one_epoch = []
        imgPathOneEpoch = []
        fea_one_epoch = []
        
        # 用于Cox模型的额外存储
        durations_all = []
        events_all = []

        with torch.no_grad():
            for step, batch in enumerate(test_loader):
                if args.model.lower() == 'densenetbigtwoforkssmall':
                    image = batch[1].to(accelerator.device, dtype=accelerator_dtype)
                else:
                    image = batch[0].to(accelerator.device, dtype=accelerator_dtype)
                
                # 区分Cox模型和分类模型的标签处理
                if args.loss.lower() in ['cox','cox_log_loss','coxloss','nll','pchazard']:
                    # Cox模型: 标签是(持续时间, 事件)
                    # 确保我们处理的是张量而不是列表
                    duration = torch.tensor(batch[-2][0], device=accelerator.device)
                    event = torch.tensor(batch[-2][1], device=accelerator.device)
                    label = (duration, event)
                    durations_all.append(duration.cpu().numpy())
                    events_all.append(event.cpu().numpy())
                    label_one_epoch.append(np.stack([duration.cpu().numpy(), event.cpu().numpy()], axis=1))
                else:
                    # 分类模型
                    label = batch[-2].to(accelerator.device)
                    label_one_epoch.append(label.cpu().numpy())
                    
                imgpath = batch[-1]
                imgPathOneEpoch.append(imgpath)

                if args.save_fea:
                    logits_per_image, fea = model(image)
                else:
                    logits_per_image = model(image)

                # 计算损失
                if args.loss.lower() in ['cox','cox_log_loss','coxloss','nll','pchazard']:
                    loss = loss_function(logits_per_image, duration, event)
                else:
                    loss = loss_function(logits_per_image, label)
                if args.loss.lower() in ['pchazard']:
                    logits_per_image,_ = preds_to_risk(logits_per_image)
                # Cox模型输出的是风险分数，不需要softmax
                if args.loss.lower() not in ['cox','cox_log_loss','coxloss','nll','pchazard']:
                    logits_per_image = F.softmax(logits_per_image, dim=-1)
                if args.loss.lower() not in ['pchazard']:
                    logits_per_image = logits_per_image.float().detach().cpu().numpy()
                
                pred_one_epoch.append(logits_per_image)
                if args.save_fea:
                    fea_one_epoch.append(fea.float().detach().cpu().numpy())

                loss_val.append(loss.item())
                accelerator.print(f"{mode} step: {step}, Loss: {loss.item()}")

        pred_one_epoch = np.concatenate(pred_one_epoch)
        label_one_epoch = np.concatenate(label_one_epoch)
        imgPathOneEpoch = np.concatenate(imgPathOneEpoch)

        # 计算时间点AUC（仅对Cox模型）
        time_auc_results = None
        if args.loss.lower() in ['cox','cox_log_loss','coxloss','nll']:
            if args.train_km_data is None or args.only_test:
                pass
            else:
                durations = label_one_epoch[:, 0]
                events = label_one_epoch[:, 1]
                time_auc_results1 = calculate_timepoint_auc(
                    pred_one_epoch.flatten(), 
                    durations, 
                    events, 
                    args.time_points,
                    is_train=False,
                    train_km_data=args.train_km_data[0]
                )
                time_auc_results2 = calculate_timepoint_auc_weighted_km(
                    pred_one_epoch.flatten(), 
                    durations, 
                    events, 
                    args.time_points,
                    is_train=False,
                    train_surv_data=args.train_km_data[1]
                )
                time_auc_results3 = calculate_timepoint_auc_kernel_surv(
                    pred_one_epoch.flatten(), 
                    durations, 
                    events, 
                    args.time_points,
                    is_train=False,
                    train_kernel_data=args.train_km_data[2]
                )
                time_auc_results = [time_auc_results1, time_auc_results2, time_auc_results3]
            # 保存验证结果
        if args.loss.lower() in ['cox','cox_log_loss','coxloss','nll']:
            # 对于Cox模型，计算C-index
            val_cindex = save_results(pred_one_epoch, label_one_epoch, imgPathOneEpoch, epoch, mode, args, time_auc_results)
            val_metric = metric_func(pred_one_epoch, label_one_epoch, args)
            val_acc = val_cindex
        else:
            val_acc = save_results(pred_one_epoch, label_one_epoch, imgPathOneEpoch, epoch, mode, args)
            val_metric = metric_func(pred_one_epoch, label_one_epoch, args)
            val_cindex = val_acc

        accelerator.print(f"{mode} C-index/Accuracy: {val_cindex}")
        if accelerator.is_main_process:
            writer.add_scalar(f"{mode}/C-index/Accuracy", val_cindex, epoch)
            writer.add_scalar(f"{mode}/lossinval", np.mean(loss_val), epoch)
            writer.add_scalar(f"{mode}/Metric", val_metric, epoch)
            # 记录时间点AUC
            if time_auc_results:
                # for t, auc in time_auc_results.items():
                #     writer.add_scalar(f"{mode}/AUC_at_{t}", auc, epoch)
                accelerator.print(f"Epoch: {epoch}, {mode} Timepoint AUC: {time_auc_results}",
                                file=open(args.save_path + "Metrics.txt", "a"))
                    
        accelerator.log({f"{mode}/C-index/Accuracy": val_cindex}, step=epoch)
        accelerator.print(f"Epoch: {epoch}, {mode} C-index/Accuracy: {val_cindex}",
                        file=open(args.save_path + "Metrics.txt", "a"))
        accelerator.print(f"Epoch: {epoch}, {mode} Metric: {val_metric}",
                        file=open(args.save_path + "Metrics.txt", "a"))

        return np.mean(loss_val), val_metric, val_cindex, time_auc_results

    parser = argparse.ArgumentParser(description="PyTorch Training")
    parser.add_argument("--trainpath", default="", type=str, help="directory to train data")
    parser.add_argument("--testpaths", default="", type=str, help="comma-separated directories to test data")
    parser.add_argument("--loss", default="ce", type=str, help="loss function")
    parser.add_argument("--seed", default=3407, type=int, help="random seed")
    parser.add_argument("--head", default="default", type=str, help="cls head")
    parser.add_argument("--train_km_data", default=[], type=list, help="KM data")
    parser.add_argument("--metric_func", default="accuracy", type=str, help="metric function")
    parser.add_argument('--datadir', type=str, default='none')
    parser.add_argument('--img_size', type=str, default=False)
    parser.add_argument('--img_aug','--img_augmentation', type=bool, default=False)
    parser.add_argument('--transform', type=bool, default=True)
    parser.add_argument('--val_use_train_trainsform', type=bool, default=False)
    parser.add_argument('--standard_func', type=str, default='max', help='zscore, minmax, none')
    parser.add_argument('--model', default="CLIP_CLS", type=str, help="clip_cls")
    parser.add_argument("--logdir", default="test", type=str, help="directory to save the tensorboard logs")
    parser.add_argument("--spatial_dims", default=3, type=int, help="spatial dimension of input data")
    parser.add_argument("--cube_size", default="24", type=str, help="the length of each edge of the cube cut from 3d img")
    parser.add_argument("--a_min", default=-1000, type=float, help="a_min in ScaleIntensityRanged")
    parser.add_argument("--a_max", default=1000, type=float, help="a_max in ScaleIntensityRanged")
    parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
    parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
    parser.add_argument("--space_x", default=1.5, type=float, help="spacing in x direction")
    parser.add_argument("--space_y", default=1.5, type=float, help="spacing in y direction")
    parser.add_argument("--space_z", default=2.0, type=float, help="spacing in z direction")
    parser.add_argument("--roi_x", default=64, type=int, help="roi size in x direction")
    parser.add_argument("--roi_y", default=64, type=int, help="roi size in y direction")
    parser.add_argument("--roi_z", default=64, type=int, help="roi size in z direction")
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--save_path', type=str, default='')
    parser.add_argument('--save_fea', type=str, default=None)
    parser.add_argument("--save_feature", default=False, type=bool, help="if save feature")
    parser.add_argument("--saveallckp", default=False, type=bool, help="if save all checkpoints")
    parser.add_argument("--freeze", default="None", type=str, help="if freeze")
    parser.add_argument("--epochs", default=200, type=int, help="number of training epochs")
    parser.add_argument("--num_steps", default=100000000, type=int, help="number of training iterations")
    parser.add_argument("--eval_num", default=100, type=int, help="evaluation frequency")
    parser.add_argument("--save_num", default=2000, type=int, help="evaluation frequency")
    parser.add_argument("--print_num", default=10, type=int, help="evaluation frequency")
    parser.add_argument("--warmup_steps", default=500, type=int, help="warmup steps")
    parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
    parser.add_argument("--feature_size", default=24, type=int, help="embedding size")
    parser.add_argument("--dropout_path_rate", default=0.0, type=float, help="drop path rate")
    parser.add_argument("--use_checkpoint", action="store_true", help="use gradient checkpointing to save memory")
    parser.add_argument("--only_test", action="store_true", help="only test the model")
    parser.add_argument("--pretrained", default="None", type=str, help="if pretrained")
    parser.add_argument("--load_part", default=None, type=str, help="if load part")
    parser.add_argument("--freeze_feaExtractor", default=-1, type=int, help="if freeze feaExtractor")
    parser.add_argument("--cut_patches", default=False, type=bool, help="if cut patches for use densenetbigtwoforkssmall")
    parser.add_argument("--label_name", default="label", type=str, help="label name in csv")
    parser.add_argument("--path_name", default="path", type=str, help="path name in csv")
    parser.add_argument("--trlabel_name", default=False, type=str, help="trlabel name in csv")
    parser.add_argument("--trpath_name", default=False, type=str, help="trpath name in csv")
    parser.add_argument("--telabel_name", default=False, type=str, help="testlabel name in csv")
    parser.add_argument("--tepath_name", default=False, type=str, help="testpath name in csv")
    parser.add_argument("--oversample", default=False, type=bool, help="if oversample")
    parser.add_argument("--train_data_rate", default=1.0, type=float, help="train data rate")
    parser.add_argument("--max_result", default=[0.,0.,0.], type=list, help="best result")
    parser.add_argument("--bind_num", default=0, type=int, help="不提升3次就要进行下一次操作")
    parser.add_argument("--bind_lr", default=False, type=bool, help="if bind lr")
    parser.add_argument("--feabindblock", default=None, type=str, help="None,first2block'")
    parser.add_argument("--local_rank", type=int, default=0, help="local rank")
    parser.add_argument("--batch_size", default=5, type=int, help="number of batch size")
    parser.add_argument("--sw_batch_size", default=2, type=int, help="number of sliding window batch size")
    parser.add_argument("--lr", default=1e-4, type=float, help="learning rate")
    parser.add_argument("--decay", default=0.0001, type=float, help="decay rate")
    parser.add_argument("--momentum", default=0.9, type=float, help="momentum")
    parser.add_argument("--lrdecay", action="store_true", help="enable learning rate decay")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="maximum gradient norm")
    parser.add_argument("--loss_type", default="SSL", type=str)
    parser.add_argument("--opt", default="adamw", type=str, help="optimization algorithm")
    parser.add_argument("--lr_schedule", default="warmup_cosine", type=str)
    parser.add_argument("--resume", default=None, type=str, help="resume training")
    parser.add_argument("--grad_clip", action="store_true", help="gradient clip")
    parser.add_argument("--noamp", action="store_true", help="do NOT use amp for training")
    parser.add_argument("--smartcache_dataset", action="store_true", help="use monai smartcache Dataset")
    parser.add_argument("--cache_dataset", action="store_true", help="use monai cache Dataset")
    parser.add_argument("--num_workers", default=4, type=int, help="number of workers")
    parser.add_argument("--device", default="0,1,2,3", type=str, help="gpu device")
    parser.add_argument("--num_classes", default=2, type=int, help="number of classes for multi-classification")
    # 新增：Cox模型时间点AUC评估参数
    parser.add_argument("--time_points", type=str, default="183,365,548,730", 
                        help="comma-separated time points for survival status prediction AUC (months)")
    args = parser.parse_args()
    
    def seed_func(seed=3407):
        random.seed(3407)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
    seed_func(args.seed)
    
    # 处理时间点参数
    args.time_points = [float(t) for t in args.time_points.split(',')] if args.time_points else []
    print("args.time_points:", args.time_points,"时间点一定要和标签单位统一，如果要自行设置，请 设置args.time_points参数")
    
    if args.img_size:
        args.img_size = [int(args.img_size.split(',')[0]), int(args.img_size.split(',')[1]), int(args.img_size.split(',')[2])]
    args.logdir = args.save_path
    print('args.logdir was set to args.save_path:', args.logdir)

    # Split testpaths into a list
    args.testpaths = args.testpaths.split(',') if args.testpaths else []

    if accelerator.is_main_process:
        os.makedirs(args.logdir, exist_ok=True)
        writer = SummaryWriter(args.logdir)
        create_args_yaml(args)

    if args.model.lower() == 'clip_cls':
        model = CLIP_CLS(classes=args.num_classes, weight_dict="/data3/log/CLIP/EXP2 - EGFR969696 - MultiGpus - 2/final_model502001.pt").to("cpu")
    elif args.model.lower() == 'm3d_cls' or args.model.lower() == 'm3d':
        config = M3DCLIPConfig()
        model = M3D_CLS(config, args).to("cpu")
        print('model:', model)
    elif args.model.lower() == 'swin3d':
        from swin3d import Swin3D
        model = Swin3D(num_classes=args.num_classes, args=args).to("cpu")
    elif args.model.lower() == 'densenet' or args.model.lower() == 'dense':
        model = DenseNet(args=args, conv0_channel=args.in_channels, num_classes=args.num_classes).to("cpu")
        print('model:', model)
    elif args.model.lower() == 'vit3d':
        

        from monai.networks.nets import ViT


        class ViT3D(nn.Module):
            def __init__(self, num_classes=2, in_channels=1):
                super().__init__()

                # 假设输入 (B, C, T, H, W) = (B, 1, 48, 256, 256)
                self.model = ViT(
                    spatial_dims=3,           # 3D ViT
                    in_channels=in_channels,  # CT 一般是 1 通道
                    img_size=(48, 256, 256),  # (T, H, W)
                    patch_size=(8, 16, 16),
                    hidden_size=768,
                    mlp_dim=3072,
                    num_layers=12,
                    num_heads=12,
                    classification=True,      # 直接输出分类 logits
                    num_classes=num_classes,
                    dropout_rate=0.1,
                )

            def forward(self, x):
                # MONAI 的 ViT 在 classification=True 时，forward 一般只返回 logits
                x,_ = self.model(x)   # 不要写 x, _ = ...
                return x


        model = ViT3D(args.num_classes,args.in_channels).to("cpu")

    else:
        print('args.model:', args.model)
        raise ValueError("Model not found")

    if args.resume:
        model_pth = args.resume
        model_dict = torch.load(model_pth, map_location="cpu",weights_only=False)
        state_dict = model_dict["state_dict"]
        new_state_dict = {}
        for key in list(state_dict.keys()):
            if "module." in key:
                new_state_dict[key[7:]] = state_dict[key]
            else:
                new_state_dict[key] = state_dict[key]
        try:
            model.load_state_dict(new_state_dict)
            accelerator.print("Model and weight fitted and loaded successfully")
        except:
            accelerator.print("Model and weight not fitted and loaded failed")
            model_module_dict = model.state_dict()
            cnt = 0
            for key, value in model_module_dict.items():
                if key in new_state_dict.keys() and value.shape == new_state_dict[key].shape:
                    model_module_dict[key] = new_state_dict[key]
                    cnt += 1
                else:
                    accelerator.print(f"{key} is not loaded")
            model.load_state_dict(model_module_dict)
            if cnt == 0:
                raise ValueError("Model and weight loaded failed")
            else:
                accelerator.print(f"Model and weight loaded successfully {cnt} keys loaded")

    # 优化器和调度器初始化需要放在模型加载之后
    if args.opt == "adam":
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    elif args.opt == "adamw":
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.decay)
    elif args.opt == "sgd":
        optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.decay)

    if args.lrdecay:
        if args.lr_schedule == "warmup_cosine":
            scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=args.num_steps)
        elif args.lr_schedule == "poly":
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: (1 - epoch / args.epochs) ** 0.9)
    else:
        scheduler = None

    if args.loss.lower() in ["ce", "crossentropy", "cross_entropy"]:
        loss_function = torch.nn.CrossEntropyLoss()
    elif args.loss.lower() in ['focal', 'focalloss']:
        num_class = [2494, 291]
        alpha = [num_class[j] / sum(num_class) for j in range(len(num_class))]
        alpha = torch.tensor(alpha, dtype=torch.float32)
        loss_function = MultiCEFocalLoss(class_num=args.num_classes, gamma=2, alpha=alpha, reduction='mean')
        print('focal loss, gamma=2, alpha=', alpha, "num_class:", num_class)
    elif args.loss.lower() in ['multilabel_ce', 'multilabelcrossentropy']:
        weight = [1.0] * args.num_classes
        loss_function = MUltilabelCELoss(class_num=args.num_classes, weight=weight, reduction='mean', ifsigmoid=True)
        print('multilabel cross entropy loss, weight:', weight)
    elif args.loss.lower() in ['cox', 'cox_log_loss','coxloss','nll']:
        loss_function = CoxLoss()
        print('cox loss')
        # 对于Cox模型，默认使用C-index作为评估指标
        if args.metric_func != 'cindex':
            accelerator.print(f"Warning: For Cox loss, metric_func is set to 'cindex' instead of '{args.metric_func}'")
            args.metric_func = 'cindex'
    elif args.loss.lower() in ["pchazard"]:
        loss_function = PCHazardLoss(num_durations=args.num_classes)
        print("pchazard loss")
        if args.metric_func != 'cindex':
            accelerator.print(f"Warning: For PCHazard loss, metric_func is set to 'cindex' instead of '{args.metric_func}'")
            args.metric_func = 'cindex'
    elif args.loss.lower() in ["nllloss","nll","nllsurv"]:
        loss_function = NLLSurvLoss(time_bins=args.time_points, alpha=0.15)
        print("nll loss")
        if args.metric_func != 'cindex':
            accelerator.print(f"Warning: For nllsurv loss, metric_func is set to 'cindex' instead of '{args.metric_func}'")
            args.metric_func = 'cindex'

        
    if args.loss.lower() in ['cox','cox_log_loss','coxloss','nll','pchazard']:
        train_dataset = dataset_clip_cox(args=args, csvpath=args.trainpath, mode='train')
        test_datasets = [dataset_clip_cox(args=args, csvpath=testpath, mode='test') for testpath in args.testpaths]
    else:
        train_dataset = dataset_clip_cls(args=args, csvpath=args.trainpath, mode='train')
        test_datasets = [dataset_clip_cls(args=args, csvpath=testpath, mode='test') for testpath in args.testpaths]

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              pin_memory=args.pin_memory, num_workers=args.num_workers)
    test_loaders = [DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                               pin_memory=args.pin_memory, num_workers=args.num_workers)
                    for test_dataset in test_datasets]

    model, optimizer, train_loader, *test_loaders = accelerator.prepare(
        model, optimizer, train_loader, *test_loaders
    )

    global_step = 1
    best_val = 1e8

    while global_step < args.num_steps:
        global_step, loss, best_val = train_cls(args, global_step, train_loader, best_val, None, scheduler)

    if accelerator.is_main_process:
        checkpoint = {
            "epoch": args.epochs,
            "state_dict": accelerator.unwrap_model(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": args
        }
        torch.save(accelerator.unwrap_model(model).state_dict(), args.logdir + "final_model.pth")
        save_ckp(checkpoint, args.logdir + "/model_final_epoch.pt")

if __name__ == "__main__":
    print('pid:', os.getpid())
    main()
