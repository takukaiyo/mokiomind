# -*- coding: utf-8 -*-
# @Time    : 2026/3/25 16:11
# @Author  : yezhuohai
# @File    : train_pretrain.py
# @Software: PyCharm

"""
train_pretrain.py —— MokioMind 预训练入口脚本

使用方法：
    # 使用默认配置训练
    python -m trainer.train_pretrain

    # 自定义参数
    python -m trainer.train_pretrain \
        --data_path data/pretrain_data.jsonl \
        --epochs 5 \
        --batch_size 32 \
        --max_lr 5e-4

整体流程：
    1. 解析命令行参数（或使用默认值）
    2. 初始化 tokenizer 和模型
    3. 构建 DataLoader
    4. 创建优化器（AdamW）
    5. 逐 epoch 训练，每个 epoch 结束保存 checkpoint
    6. 训练结束后保存最终模型（HuggingFace 格式，可直接用 from_pretrained 加载）

依赖关系：
    train_pretrain.py
        ├── model.model.MokioMindConfig        # 模型配置
        ├── model.model.MokioMindForCausalLM    # 模型定义
        ├── trainer_utils.pretrain_epoch         # 单 epoch 训练循环
        ├── trainer_utils.build_pretrain_dataloader  # DataLoader 构建
        └── trainer_utils.save_checkpoint        # Checkpoint 保存
"""

import os
import argparse
import torch
from transformers import AutoTokenizer

from model.model import MokioMindConfig, MokioMindForCausalLM
from trainer.trainer_utils import (
    pretrain_epoch,
    build_pretrain_dataloader,
    save_checkpoint,
)

# wandb 可选导入，没装也能训练
try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


# ──────────────────────────────────────────────────────────────────────────────
# 1. 命令行参数定义
# ──────────────────────────────────────────────────────────────────────────────
# 为什么用 argparse 而不是硬编码？
#   - 方便在不同实验之间切换超参数，不需要改代码
#   - 命令行参数会被记录在 shell history 里，天然的实验日志
#   - 后续可以很容易地迁移到 yaml 配置文件（用 argparse 读 yaml 即可）
# ──────────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="MokioMind 预训练脚本")

    # ── 数据相关 ──
    parser.add_argument(
        "--data_path", type=str, default="/data/yezhuohai/data/minimind_dataset/pretrain_t2t_mini.jsonl",
        help="预训练数据路径，jsonl 格式，每行一个 {\"text\": \"...\"}",
    )
    parser.add_argument(
        "--tokenizer_path", type=str, default="tokenizer",
        help="tokenizer 路径（HuggingFace 格式目录，或 Hub 上的模型名）",
    )
    parser.add_argument("--max_length", type=int, default=512)

    # ── 训练超参数 ──
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_lr", type=float, default=5e-4)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--use_amp", action="store_true", default=True)

    # ── 日志与保存 ──
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--num_workers", type=int, default=0)

    # ── wandb 实验管理 ──
    # wandb 是实验管理平台，自动记录 loss 曲线、超参数、系统指标（GPU 利用率等）。
    # 首次使用需要注册 wandb.ai 账号并运行 `wandb login`。
    # 不想用 wandb 时传 --no_wandb 即可，训练功能不受影响。
    parser.add_argument(
        "--no_wandb", action="store_true", default=False,
        help="禁用 wandb 日志（默认启用）",
    )
    parser.add_argument(
        "--wandb_project", type=str, default="mokiomind",
        help="wandb 项目名（同一项目下的实验会归到一起，方便对比）",
    )
    parser.add_argument(
        "--wandb_run_name", type=str, default=None,
        help="本次实验的名称（不填则由 wandb 自动生成）",
    )

    return parser.parse_args()


def count_parameters(model):
    """统计模型参数量。fp32 下每个参数占 4 字节，26M 参数 ≈ 100MB 显存（仅权重）。"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ──────────────────────────────────────────────────────────────────────────────
# 2. 主训练函数
# ──────────────────────────────────────────────────────────────────────────────
#
# 训练一个语言模型的完整流程可以类比为"教一个人读书"：
#
#   Step 1 — 准备教材（tokenizer + dataloader）
#       tokenizer 把文字变成数字序列，dataloader 把数字序列打包成 batch。
#
#   Step 2 — 创建学生（model）
#       随机初始化一个什么都不会的模型，它的"知识"全靠训练获得。
#
#   Step 3 — 制定学习计划（optimizer + lr scheduler）
#       optimizer 决定"怎么更新参数"（AdamW），
#       lr scheduler 决定"学习速度怎么变化"（余弦退火：先快后慢）。
#
#   Step 4 — 开始上课（training loop）
#       每个 epoch 把所有数据过一遍，每个 batch：
#       前向传播（做题）→ 计算 loss（批改）→ 反向传播（找错因）→ 更新参数（纠正）
#
#   Step 5 — 定期考试（checkpoint）
#       每个 epoch 结束保存模型状态，万一训练中断可以从断点恢复。
#
#   Step 6 — 毕业（save final model）
#       训练结束后保存为 HuggingFace 格式，可以直接用 from_pretrained 加载。
# ──────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # ── Step 1：选择训练设备 ──────────────────────────────────────────────
    # CUDA（NVIDIA GPU）> CPU
    # 为什么优先用 GPU？
    #   矩阵乘法是训练的核心运算，GPU 有数千个核心并行计算，
    #   比 CPU 快 10~100 倍。没有 GPU 也能训练，只是非常慢。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Init] 训练设备: {device}")
    if device.type == "cuda":
        print(f"[Init] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[Init] 显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # ── Step 2：加载 tokenizer ────────────────────────────────────────────
    # tokenizer 的作用：文字 ↔ 数字 的双向转换器
    #   "你好世界" → [1, 234, 567, 89, 2]  （编码）
    #   [1, 234, 567, 89, 2] → "你好世界"  （解码）
    #
    # 为什么用 AutoTokenizer？
    #   它会自动检测 tokenizer 类型（BPE/SentencePiece/...），
    #   不需要手动指定，兼容性最好。
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    # 确保 pad_token 存在。有些 tokenizer（如 GPT-2）没有 pad_token，
    # 不设置的话 DataLoader 补齐时会报错。
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"[Init] Tokenizer 词表大小: {len(tokenizer)}")

    # ── Step 3：创建模型 ──────────────────────────────────────────────────
    # 用你在 model.py 里定义的 MokioMindConfig 创建配置，
    # 然后用配置初始化模型。此时模型权重是随机的，什么都不会。
    #
    # vocab_size 必须和 tokenizer 一致，否则 embedding 层维度不匹配会报错。
    config = MokioMindConfig(vocab_size=len(tokenizer))
    model = MokioMindForCausalLM(config)
    model = model.to(device)

    total_params, trainable_params = count_parameters(model)
    print(f"[Init] 模型参数量: {total_params:,} (可训练: {trainable_params:,})")
    print(f"[Init] 预估显存占用: ~{trainable_params * 4 / 1024**2:.0f} MB (fp32 权重)")

    # ── Step 3.5：初始化 wandb ────────────────────────────────────────────
    # wandb.init 做了什么？
    #   1. 在 wandb.ai 上创建一个新的 run（实验记录）
    #   2. 自动记录系统信息（GPU 型号、显存、OS 等）
    #   3. 把你传入的 config 保存下来，方便后续对比不同实验的超参数
    #   4. 开始监控 GPU 利用率、显存占用等系统指标
    #
    # 训练结束后 wandb.finish() 会上传所有数据并关闭连接。
    # 你可以在 wandb.ai 的 dashboard 上看到：
    #   - loss 曲线（自动平滑）
    #   - lr 变化曲线
    #   - 不同实验的超参数对比表
    #   - GPU 利用率时间线
    use_wandb = _WANDB_AVAILABLE and not args.no_wandb
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                # 记录所有超参数，方便在 wandb dashboard 上筛选和对比
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "effective_batch_size": args.batch_size * args.grad_accum_steps,
                "max_lr": args.max_lr,
                "min_lr": args.min_lr,
                "weight_decay": args.weight_decay,
                "grad_accum_steps": args.grad_accum_steps,
                "max_grad_norm": args.max_grad_norm,
                "max_length": args.max_length,
                "use_amp": args.use_amp,
                "total_params": total_params,
                "trainable_params": trainable_params,
                "model_config": config.to_dict(),
            },
        )
        # wandb.watch 会记录模型梯度和参数的分布直方图，
        # 帮助你诊断梯度消失/爆炸等问题。log_freq 控制记录频率。
        wandb.watch(model, log="gradients", log_freq=500)
        print(f"[Init] wandb 已启用，项目: {args.wandb_project}")
    else:
        if not _WANDB_AVAILABLE:
            print("[Init] wandb 未安装，跳过实验记录（pip install wandb）")
        else:
            print("[Init] wandb 已禁用（--no_wandb）")

    # ── Step 4：构建 DataLoader ────────────────────────────────────────────
    # DataLoader 是 Dataset 和训练循环之间的桥梁：
    #   Dataset[i] 返回第 i 条样本 → DataLoader 把多条样本打包成 batch
    #
    # 为什么需要 batch？
    #   - 单条样本计算梯度方差太大，batch 平均后梯度更稳定
    #   - GPU 擅长并行计算，一次处理 32 条比循环 32 次快得多
    dataloader = build_pretrain_dataloader(
        data_path=args.data_path,
        tokenizer=tokenizer,
        max_length=args.max_length,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
    )
    print(f"[Init] 数据集大小: {len(dataloader.dataset)} 条")
    print(f"[Init] 每 epoch batch 数: {len(dataloader)}")

    # ── Step 5：创建优化器 ────────────────────────────────────────────────
    # 为什么用 AdamW 而不是 SGD？
    #   - Adam 自适应调整每个参数的学习率（维护一阶矩 m 和二阶矩 v），
    #     收敛速度远快于 SGD，是 LLM 训练的标配。
    #   - AdamW 是 Adam 的改进版，修正了权重衰减的实现方式，
    #     在 Adam 中 weight_decay 实际上是 L2 正则化（和梯度耦合），
    #     AdamW 将 weight_decay 解耦为真正的权重衰减（直接缩小权重）。
    #
    # betas=(0.9, 0.95)：
    #   - beta1=0.9：一阶矩（梯度均值）的指数衰减率，控制"动量"
    #   - beta2=0.95：二阶矩（梯度方差）的指数衰减率，控制"自适应学习率"
    #   - LLM 训练中 beta2 通常用 0.95 而非默认的 0.999，
    #     因为语言数据的梯度分布变化较快，需要更快地适应。
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.max_lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )

    # ── Step 6：计算总训练步数 ────────────────────────────────────────────
    # total_steps 用于余弦退火 lr 调度：lr(t) 需要知道 t 占 T 的比例。
    #
    # 注意：一个"step"是一次参数更新，不是一个 batch。
    # 如果 grad_accum_steps=4，那么每 4 个 batch 才算 1 个 step。
    # 所以 total_steps = (batches_per_epoch / grad_accum_steps) * epochs
    steps_per_epoch = len(dataloader) // args.grad_accum_steps
    total_steps = steps_per_epoch * args.epochs
    print(f"[Init] 总训练步数: {total_steps} ({steps_per_epoch} steps/epoch × {args.epochs} epochs)")
    print(f"[Init] 等效 batch size: {args.batch_size * args.grad_accum_steps}")
    print("=" * 60)

    # ── Step 7：训练循环 ──────────────────────────────────────────────────
    current_step = 0
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*60}")

        # pretrain_epoch 内部完成：
        #   遍历 dataloader → 前向传播 → loss → 反向传播 → 梯度累积 → 参数更新
        # 返回更新后的 current_step，供下一个 epoch 的 lr 调度使用。
        current_step = pretrain_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            total_steps=total_steps,
            current_step=current_step,
            max_lr=args.max_lr,
            min_lr=args.min_lr,
            grad_accum_steps=args.grad_accum_steps,
            max_grad_norm=args.max_grad_norm,
            use_amp=args.use_amp,
            log_interval=args.log_interval,
        )

        # 每个 epoch 结束保存 checkpoint
        save_checkpoint(model, optimizer, epoch, current_step, 0.0, args.save_dir)

    # ── Step 8：保存最终模型（HuggingFace 格式）──────────────────────────
    # 为什么要额外保存 HuggingFace 格式？
    #   - checkpoint 包含优化器状态，文件很大，主要用于恢复训练
    #   - HuggingFace 格式只保存模型权重 + 配置，文件小，用于推理和分享
    #   - 保存后可以直接用 MokioMindForCausalLM.from_pretrained("output") 加载
    final_dir = os.path.join(args.save_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\n[Done] 最终模型已保存到 {final_dir}")
    print(f"[Done] 加载方式: MokioMindForCausalLM.from_pretrained('{final_dir}')")

    # ── Step 9：关闭 wandb ────────────────────────────────────────────────
    # wandb.finish() 会：
    #   1. 上传所有还未同步的日志数据
    #   2. 标记这个 run 为"完成"状态
    #   3. 在终端打印 wandb dashboard 的链接，点击即可查看结果
    if use_wandb:
        wandb.finish()
        print("[Done] wandb 实验记录已上传，可在 wandb.ai 查看")


if __name__ == "__main__":
    main()


