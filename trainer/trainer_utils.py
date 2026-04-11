# -*- coding: utf-8 -*-
# @Time    : 2026/3/25 16:11
# @Author  : yezhuohai
# @File    : trainer_utils.py
# @Software: PyCharm

"""
trainer_utils.py —— 训练工具函数集合（学习版）

本文件包含预训练（Pretrain）和有监督微调（SFT）两个训练循环的核心工具函数。
目标：让你理解"一次完整的模型训练迭代"到底发生了什么。

训练的基本流程：
    数据 → 模型前向传播 → 计算loss → 反向传播 → 梯度更新 → 循环

涉及的关键概念：
    - DataLoader：批量加载数据的工具
    - optimizer：梯度下降优化器（这里用 AdamW）
    - get_lr：学习率调度（余弦退火）
    - gradient accumulation：梯度累积（显存不够时模拟大batch）
    - gradient clipping：梯度裁剪（防止梯度爆炸）
    - mixed precision (AMP)：混合精度训练（用fp16加速，节省显存）
"""

import os
import time
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from contextlib import nullcontext
from dataset.lm_dataset import PretrainDataset, SFTDataset

# ── wandb（可选）──────────────────────────────────────────────────────────
# wandb 是目前最主流的实验管理工具，用于记录训练指标、可视化 loss 曲线、
# 对比不同实验的超参数和结果。
#
# 为什么用 wandb 而不是 tensorboard？
#   - wandb 自动同步到云端，换电脑也能看历史实验
#   - 支持超参数搜索、实验对比、团队协作
#   - 和 HuggingFace 生态集成更好
#   - 免费额度对个人研究者完全够用
#
# 设计为可选依赖：没装 wandb 也能正常训练，只是没有可视化。
try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


# ──────────────────────────────────────────────────────────────────────────────
# 1. 学习率调度：余弦退火（Cosine Annealing）
# ──────────────────────────────────────────────────────────────────────────────
# 为什么需要学习率调度？
#   - 训练初期：需要较大的学习率快速收敛
#   - 训练后期：需要较小的学习率精细调整，避免在最优点附近震荡
#
# 余弦退火公式：
#   lr(t) = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(π * t / T))
#
#   其中 t 是当前步数，T 是总步数。
#   - t=0 时：lr = max_lr（最大值）
#   - t=T 时：lr = min_lr（最小值）
#   - 中间：平滑地从 max_lr 降到 min_lr，形状像半个余弦波
# ──────────────────────────────────────────────────────────────────────────────
def get_lr(current_step: int, total_steps: int, max_lr: float, min_lr: float = 0.0) -> float:
    """
    计算当前 step 对应的余弦退火学习率。

    Args:
        current_step: 当前训练步数
        total_steps:  总训练步数
        max_lr:       初始（最大）学习率
        min_lr:       最终（最小）学习率，默认为 0

    Returns:
        当前步对应的学习率值（float）
    """
    # decay_ratio 从 0 线性增长到 1
    decay_ratio = current_step / max(total_steps, 1)
    # cos 从 1 衰减到 -1，加 1 后变成 2→0，乘 0.5 变成 1→0
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


# ──────────────────────────────────────────────────────────────────────────────
# 2. Checkpoint 存取
# ──────────────────────────────────────────────────────────────────────────────
# 为什么需要 checkpoint？
#   - 训练可能中断（断电、显存OOM等），checkpoint 让你能从中断处继续
#   - 可以保存多个 checkpoint，选择验证集上最好的那个
#
# checkpoint 里保存了什么？
#   - model state_dict：模型的所有参数（权重）
#   - optimizer state_dict：优化器状态（动量、二阶矩等），恢复训练必须有
#     （AdamW 等优化器内部维护了每个参数的一阶矩 m 和二阶矩 v，
#      如果只保存模型权重，恢复训练时优化器从零开始，前几步会不稳定）
#   - epoch / step：记录训练进度
#   - loss：记录当前 loss，方便对比不同 checkpoint
# ──────────────────────────────────────────────────────────────────────────────
def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    loss: float,
    save_dir: str,
):
    """
    保存训练 checkpoint 到文件。

    Args:
        model:     要保存的模型
        optimizer: 优化器
        epoch:     当前 epoch
        step:      当前全局步数
        loss:      当前 loss 值
        save_dir:  保存目录（不存在会自动创建）
    """
    os.makedirs(save_dir, exist_ok=True)
    checkpoint = {
        "model": model.state_dict(),        # 模型参数字典
        "optimizer": optimizer.state_dict(), # 优化器状态字典
        "epoch": epoch,
        "step": step,
        "loss": loss,
    }
    path = os.path.join(save_dir, f"ckpt_epoch{epoch}_step{step}.pt")
    torch.save(checkpoint, path)
    print(f"[Checkpoint] 已保存到 {path}，loss={loss:.4f}")


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer = None,
    device: torch.device = torch.device("cpu"),
):
    """
    加载训练 checkpoint，恢复模型和优化器状态。

    Args:
        path:      checkpoint 文件路径
        model:     要恢复参数的模型（会被原地修改）
        optimizer: 要恢复状态的优化器（推理时可传 None，只加载模型权重）
        device:    加载到的设备（map_location 确保 GPU checkpoint 能在 CPU 上加载）

    Returns:
        (epoch, step, loss) 上次保存时的训练进度
    """
    checkpoint = torch.load(path, map_location=device)
    # strict=True：checkpoint 里的 key 必须和模型完全匹配，有多有少都会报错
    model.load_state_dict(checkpoint["model"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    print(
        f"[Checkpoint] 已从 {path} 恢复，"
        f"epoch={checkpoint['epoch']}，step={checkpoint['step']}，loss={checkpoint['loss']:.4f}"
    )
    return checkpoint["epoch"], checkpoint["step"], checkpoint["loss"]


# ──────────────────────────────────────────────────────────────────────────────
# 3. 预训练训练循环（pretrain_epoch）
# ──────────────────────────────────────────────────────────────────────────────
# 预训练目标：Next-Token Prediction（下一个 token 预测）
#   - 输入：一段文本的 token 序列
#   - 输出：每个位置预测下一个 token 的概率分布
#   - loss：CrossEntropyLoss（模型预测 vs 真实下一个 token）
#
# 三个关键技术：
#
# ① 梯度累积（Gradient Accumulation）
#   显存不够时，把大 batch 拆成多个小 batch 分步计算，
#   累积梯度后再统一更新，效果等价于大 batch 训练。
#   注意：每个小 batch 的 loss 要除以累积步数，保证梯度量级不变。
#
# ② 混合精度（AMP, Automatic Mixed Precision）
#   前向传播用 fp16（半精度）节省显存、加速计算，
#   反向传播用 fp32（全精度）保证数值稳定性。
#   GradScaler 负责自动缩放梯度，防止 fp16 梯度下溢（变成 0）。
#
# ③ 梯度裁剪（Gradient Clipping）
#   将所有参数梯度的 L2 范数限制在 max_grad_norm 以内，
#   防止梯度爆炸导致参数更新过猛、训练崩溃。
# ──────────────────────────────────────────────────────────────────────────────
def pretrain_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    total_steps: int,
    current_step: int,
    max_lr: float = 5e-4,
    min_lr: float = 1e-5,
    grad_accum_steps: int = 1,
    max_grad_norm: float = 1.0,
    use_amp: bool = True,
    log_interval: int = 100,
) -> int:
    """
    执行一个 epoch 的预训练。

    Args:
        model:            模型（MokioMindForCausalLM）
        dataloader:       预训练数据加载器（PretrainDataset）
        optimizer:        优化器（AdamW）
        device:           训练设备（cuda / cpu）
        epoch:            当前 epoch 编号（仅用于日志）
        total_steps:      总训练步数（用于余弦 lr 调度）
        current_step:     当前全局步数（跨 epoch 累计）
        max_lr:           初始学习率
        min_lr:           最小学习率
        grad_accum_steps: 梯度累积步数（默认 1，即不累积）
        max_grad_norm:    梯度裁剪阈值（默认 1.0）
        use_amp:          是否使用混合精度训练（需要 CUDA）
        log_interval:     每隔多少步打印一次日志

    Returns:
        current_step: 更新后的全局步数（供下一个 epoch 继续使用）
    """
    model.train()  # 切换到训练模式（启用 Dropout 等训练专用行为）

    # GradScaler：AMP 的梯度缩放器，只在 CUDA 上有效
    scaler = GradScaler(enabled=use_amp and device.type == "cuda")

    # autocast 上下文：在其中的前向传播自动使用 fp16
    # CPU 或不用 AMP 时，用 nullcontext() 占位（什么都不做）
    amp_ctx = (
        torch.autocast(device_type=device.type, dtype=torch.float16)
        if use_amp and device.type == "cuda"
        else nullcontext()
    )

    total_loss = 0.0
    start_time = time.time()

    for batch_idx, (input_ids, labels, attention_mask) in enumerate(dataloader):
        # ── 数据移到训练设备 ────────────────────────────────────────────────
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        attention_mask = attention_mask.to(device)

        # ── 动态调整学习率 ──────────────────────────────────────────────────
        # 每一步都根据余弦调度更新 lr，直接修改 optimizer 的 param_groups
        current_lr = get_lr(current_step, total_steps, max_lr, min_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        # ── 前向传播 + loss 计算 ────────────────────────────────────────────
        with amp_ctx:
            # model 返回 CausalLMOutputWithPast，其中 .loss 是内部计算好的 CrossEntropyLoss
            output = model(
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
            )
            # 梯度累积：每个小 batch 的 loss 除以累积步数，
            # 这样 N 步累积后的梯度等价于在 N 倍大 batch 上计算的梯度
            loss = output.loss / grad_accum_steps

        # ── 反向传播 ────────────────────────────────────────────────────────
        # scaler.scale(loss)：将 loss 乘以缩放因子，防止 fp16 梯度下溢
        scaler.scale(loss).backward()

        # ── 梯度更新（每 grad_accum_steps 步执行一次）──────────────────────
        if (batch_idx + 1) % grad_accum_steps == 0:
            # unscale_：将梯度还原为真实值（去掉 scaler 的缩放因子）
            scaler.unscale_(optimizer)
            # 梯度裁剪：将所有参数梯度的 L2 范数限制在 max_grad_norm 以内
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            # scaler.step：若梯度中没有 inf/nan，则执行 optimizer.step() 更新参数
            scaler.step(optimizer)
            # scaler.update：根据本次是否出现 inf/nan 动态调整缩放因子
            scaler.update()
            # 清空梯度缓冲区，准备下一次累积
            optimizer.zero_grad()
            current_step += 1

            # ── wandb 记录（每个 step 都记录）──────────────────────────────
            # 为什么每个 step 都记录而不是每个 log_interval？
            #   wandb 会自动做平滑和下采样，原始数据越密越好。
            #   记录的开销极小（微秒级），不会影响训练速度。
            if _WANDB_AVAILABLE and wandb.run is not None:
                wandb.log({
                    "train/loss": loss.item() * grad_accum_steps,  # 还原真实 loss
                    "train/lr": current_lr,
                    "train/step": current_step,
                    "train/epoch": epoch,
                }, step=current_step)

        total_loss += loss.item() * grad_accum_steps  # 还原真实 loss 用于日志

        # ── 日志打印 ────────────────────────────────────────────────────────
        if (batch_idx + 1) % log_interval == 0:
            avg_loss = total_loss / (batch_idx + 1)
            elapsed = time.time() - start_time
            print(
                f"[Pretrain] Epoch {epoch} | Step {current_step} | "
                f"Batch {batch_idx + 1}/{len(dataloader)} | "
                f"Loss {avg_loss:.4f} | LR {current_lr:.2e} | "
                f"Elapsed {elapsed:.1f}s"
            )

    return current_step


# ──────────────────────────────────────────────────────────────────────────────
# 4. SFT 训练循环（sft_epoch）
# ──────────────────────────────────────────────────────────────────────────────
# SFT 与 Pretrain 的核心区别：
#
#   Pretrain：对整段文本每个位置都计算 loss，labels = input_ids（全部有效）
#   SFT：     只对 assistant 回复计算 loss，labels 稀疏（大量 -100）
#
# 为什么 SFT 只对 assistant 部分计算 loss？
#   - user 的输入只是"条件"，不是模型需要学习生成的内容
#   - 如果对 user 输入也计算 loss，模型会浪费容量去"学习用户怎么提问"
#   - 只学 assistant 回复，让模型专注于"如何正确回答"
#
# 训练流程与 pretrain_epoch 完全相同，区别仅在于：
#   SFTDataset 提供的 labels 是稀疏的（非 assistant 部分为 -100），
#   CrossEntropyLoss 的 ignore_index=-100 会自动跳过这些位置。
# ──────────────────────────────────────────────────────────────────────────────
def sft_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    total_steps: int,
    current_step: int,
    max_lr: float = 1e-4,
    min_lr: float = 1e-6,
    grad_accum_steps: int = 1,
    max_grad_norm: float = 1.0,
    use_amp: bool = True,
    log_interval: int = 100,
) -> int:
    """
    执行一个 epoch 的有监督微调（SFT）。
    参数含义与 pretrain_epoch 完全一致，此处不再重复。

    Returns:
        current_step: 更新后的全局步数
    """
    model.train()

    scaler = GradScaler(enabled=use_amp and device.type == "cuda")
    amp_ctx = (
        torch.autocast(device_type=device.type, dtype=torch.float16)
        if use_amp and device.type == "cuda"
        else nullcontext()
    )

    total_loss = 0.0
    start_time = time.time()

    for batch_idx, (input_ids, labels, attention_mask) in enumerate(dataloader):
        input_ids = input_ids.to(device)
        labels = labels.to(device)          # 稀疏 labels，-100 位置不计入 loss
        attention_mask = attention_mask.to(device)

        current_lr = get_lr(current_step, total_steps, max_lr, min_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        with amp_ctx:
            output = model(
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
            )
            loss = output.loss / grad_accum_steps

        scaler.scale(loss).backward()

        if (batch_idx + 1) % grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            current_step += 1

            if _WANDB_AVAILABLE and wandb.run is not None:
                wandb.log({
                    "train/loss": loss.item() * grad_accum_steps,
                    "train/lr": current_lr,
                    "train/step": current_step,
                    "train/epoch": epoch,
                }, step=current_step)

        total_loss += loss.item() * grad_accum_steps

        if (batch_idx + 1) % log_interval == 0:
            avg_loss = total_loss / (batch_idx + 1)
            elapsed = time.time() - start_time
            print(
                f"[SFT] Epoch {epoch} | Step {current_step} | "
                f"Batch {batch_idx + 1}/{len(dataloader)} | "
                f"Loss {avg_loss:.4f} | LR {current_lr:.2e} | "
                f"Elapsed {elapsed:.1f}s"
            )

    return current_step


# ──────────────────────────────────────────────────────────────────────────────
# 5. DataLoader 构建工具
# ──────────────────────────────────────────────────────────────────────────────
# DataLoader 的作用：
#   - 自动将 Dataset 的样本打包成 batch
#   - shuffle=True：每个 epoch 随机打乱顺序，防止模型记住数据顺序
#   - num_workers：数据加载的并行进程数（Windows 建议设 0，Linux 可设 4）
#   - pin_memory=True：将数据预先锁定在内存中，加速 CPU→GPU 的数据传输
#   - drop_last=True：丢弃最后一个不完整的 batch，保证每个 batch 大小一致
# ──────────────────────────────────────────────────────────────────────────────
def build_pretrain_dataloader(
    data_path: str,
    tokenizer,
    max_length: int = 512,
    batch_size: int = 32,
    num_workers: int = 0,
    shuffle: bool = True,
) -> DataLoader:
    """
    构建预训练 DataLoader。

    Args:
        data_path:   jsonl 数据文件路径
        tokenizer:   分词器
        max_length:  序列最大长度
        batch_size:  每个 batch 的样本数
        num_workers: 数据加载的并行进程数（Windows 建议设 0）
        shuffle:     是否随机打乱数据顺序（训练集 True，验证集 False）

    Returns:
        DataLoader 对象
    """
    dataset = PretrainDataset(data_path, tokenizer, max_length)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


def build_sft_dataloader(
    data_path: str,
    tokenizer,
    max_length: int = 1024,
    batch_size: int = 16,
    num_workers: int = 0,
    shuffle: bool = True,
) -> DataLoader:
    """
    构建 SFT DataLoader。（参数含义同 build_pretrain_dataloader）
    """
    dataset = SFTDataset(data_path, tokenizer, max_length)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
