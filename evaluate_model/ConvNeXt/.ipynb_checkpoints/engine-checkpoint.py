# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import math
from typing import Iterable, Optional
import torch
from timm.data import Mixup
from timm.utils import accuracy, ModelEma
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, average_precision_score
import numpy as np
import pandas as pd

import utils

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None, log_writer=None,
                    wandb_logger=None, start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None, use_amp=False):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    optimizer.zero_grad()

    for data_iter_step, examples in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        samples = examples["features"].to(device, non_blocking=True)
        targets = examples["labels"].to(device, non_blocking=True)

        if use_amp:
            with torch.cuda.amp.autocast():
                output = model(samples)
                loss = criterion(output, targets)
        else: # full precision
            output = model(samples)
            loss = criterion(output, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value): # this could trigger if using AMP
            print("Loss is {}, stopping training".format(loss_value))
            assert math.isfinite(loss_value)

        if use_amp:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
        else: # full precision
            loss /= update_freq
            loss.backward()
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.step()
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)

        torch.cuda.synchronize()

        if mixup_fn is None:
            class_acc = (output.max(-1)[-1] == targets).float().mean()
        else:
            class_acc = None
        metric_logger.update(loss=loss_value)
        metric_logger.update(class_acc=class_acc)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        if use_amp:
            metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(class_acc=class_acc, head="loss")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            if use_amp:
                log_writer.update(grad_norm=grad_norm, head="opt")
            log_writer.set_step()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

# @torch.no_grad()
# def evaluate(data_loader, model, device, use_amp=False):
#     criterion = torch.nn.CrossEntropyLoss()

#     metric_logger = utils.MetricLogger(delimiter="  ")
#     header = 'Test:'

#     # switch to evaluation mode
#     model.eval()
#     for batch in metric_logger.log_every(data_loader, 10, header):
#         images = batch["features"]
#         target = batch["labels"]

#         images = images.to(device, non_blocking=True)
#         target = target.to(device, non_blocking=True)

#         # compute output
#         if use_amp:
#             with torch.cuda.amp.autocast():
#                 output = model(images)
#                 loss = criterion(output, target)
#         else:
#             output = model(images)
#             loss = criterion(output, target)

#         acc1, acc5 = accuracy(output, target, topk=(1, 5))

#         batch_size = images.shape[0]
#         metric_logger.update(loss=loss.item())
#         metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
#     # gather the stats from all processes
#     metric_logger.synchronize_between_processes()
#     print('* Acc@1 {top1.global_avg:.3f}  loss {losses.global_avg:.3f}'
#           .format(top1=metric_logger.acc1, losses=metric_logger.loss))

#     return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
@torch.no_grad()
def evaluate(data_loader, model, device, use_amp=False):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    all_predictions = []
    all_targets = []
    all_probabilities = []

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch["features"]
        target = batch["labels"]

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        if use_amp:
            with torch.cuda.amp.autocast():
                output = model(images)
                loss = criterion(output, target)
        else:
            output = model(images)
            loss = criterion(output, target)

        probabilities = F.softmax(output, dim=1)
        predictions = output.argmax(dim=1)

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())

        all_predictions.append(predictions.cpu().numpy())
        all_targets.append(target.cpu().numpy())
        all_probabilities.append(probabilities.cpu().numpy())

    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    all_probabilities = np.concatenate(all_probabilities, axis=0)
    print(all_targets.shape,all_probabilities.shape)

    # average='macro': 计算每个类别的召回率然后平均
    # average='micro': 全局计算
    # average='weighted': 按样本数量加权平均

    accuracy = accuracy_score(all_targets, all_predictions)
    print(f"acc的值为{accuracy}")
    recall_weighted = recall_score(all_targets, all_predictions, average='weighted', zero_division=0)
    roc_auc_ovo = roc_auc_score(all_targets, all_probabilities[:, 1])

    metric_logger.meters['accuracy'].update(accuracy, n=len(all_targets))  # 注意：这里用len(all_targets)，不是batch_size
    metric_logger.meters['recall_weighted'].update(recall_weighted, n=len(all_targets))
    metric_logger.meters['auc'].update(roc_auc_ovo, n=len(all_targets))

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    print(
        '* Acc {acc.global_avg:.3f}  Recall {recall.global_avg:.3f} Auc {auc.global_avg:.3f} Loss {loss.global_avg:.3f}'.format(
            acc=metric_logger.meters['accuracy'],
            recall=metric_logger.meters['recall_weighted'],
            auc=metric_logger.meters['auc'],
            loss=metric_logger.loss
        ))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



@torch.no_grad()
def test(data_loader, model, device):

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    results = []
    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch["features"]
        labels = batch["labels"]
        images = images.to(device, non_blocking=True)
        output = model(images)
        probs = F.softmax(model(images),dim=1)
        preds = probs.argmax(dim=1)
        confidences = probs.max(dim=1)[0]
        # 保存每个样本的结果
        for i, path in enumerate(labels):
            results.append({
                'label': labels[i],
                'prediction': preds[i].item(),
                "confidence":confidences[i].item()
            })
    df = pd.DataFrame(results)
    return df
