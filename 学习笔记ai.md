# Transformer核心模块学习笔记（MokioMind项目版）

> 目标：从初学者角度理解本项目 `model.py` 中的核心模块与公式，做到“看得懂代码、能解释原理、能自己复现”。

---

## 0. 模型整体流程（先建立全局认知）

在 `MokioMindForCausalLM` 里，主流程是：

1. `input_ids` → `Embedding`
2. 经过多层 `MokioMindBlock`
   - `RMSNorm + Attention + 残差`
   - `RMSNorm + FeedForward(SwiGLU) + 残差`
3. 最后 `RMSNorm` + `lm_head` 得到 `logits`
4. 训练时用 `CrossEntropyLoss` 计算 next-token loss

---

## 1. RMSNorm

### 1.1 LayerNorm vs RMSNorm

- **LayerNorm**：先减均值再除标准差（去中心化 + 缩放）
- **RMSNorm**：只做缩放，不减均值（更简洁、更高效）

### 1.2 公式

设输入向量为 $x \in \mathbb{R}^d$：

$$
\text{RMS}(x)=\sqrt{\frac{1}{d}\sum_{i=1}^{d}x_i^2+\epsilon}
$$

$$
\text{RMSNorm}(x)=\frac{x}{\text{RMS}(x)}\odot w
$$

其中 $w$ 是可学习参数，$\epsilon$ 用于数值稳定。

### 1.3 代码映射（本项目）

- `x.pow(2).mean(-1, keepdim=True)`：计算均方
- `torch.rsqrt(...)`：$\frac{1}{\sqrt{\cdot}}$
- `* self.weight`：逐维缩放

---

## 2. RoPE（Rotary Position Embedding）

### 2.1 为什么需要位置编码？

Attention 本身对 token 顺序不敏感。  
RoPE 通过“旋转”Q/K向量，把位置信息注入注意力计算中。

### 2.2 核心思想

把向量按二维对子看作旋转平面，在位置 \(m\) 处乘旋转矩阵：

$$
R_{\theta,m}^{(i)}=
\begin{bmatrix}
\cos(m\theta_i) & -\sin(m\theta_i) \\
\sin(m\theta_i) & \cos(m\theta_i)
\end{bmatrix}
$$

基础频率常写作：

$$
\theta_i = \text{base}^{-2i/d}
$$

### 2.3 直觉理解

同一个词在不同位置会“旋转到不同角度”，  
因此注意力点积能反映相对距离关系。

### 2.4 本项目实现要点

- `precompute_freqs`：预计算所有位置的 `cos/sin`
- `apply_rotary_pos_emb`：将 `q,k` 与 `cos,sin`融合
- 只对当前序列区间切片，支持缓存推理

---

## 3. YaRN（长上下文RoPE扩展）

### 3.1 为什么要YaRN？

模型训练时上下文长度有限（如2K/4K）。  
直接推理到更长长度（如32K）会导致高频震荡、注意力不稳定。

### 3.2 YaRN做什么？

在推理时对RoPE频率做分段缩放与平滑过渡：
- 高频区少缩放
- 低频区多缩放
- 中间区线性过渡（ramp）

### 3.3 本项目关键代码对应

- `low, high`：频率区间切分点
- `ramp = clamp(...)`：0→1平滑因子
- `freqs = freqs * (1 - ramp + ramp/factor)`：融合公式
- `attn_factor`：注意力温度补偿

---

## 4. Attention（含GQA、KV Cache、Causal Mask）

### 4.1 标准步骤（对应`Attention.forward`）

1. 输入 `x` 投影得到 `q,k,v`
2. 对 `q,k` 应用RoPE
3. 若有 `past_key_value`，拼接历史 `k,v`
4. GQA场景下复制KV头（`repeat_kv`）
5. 计算注意力（Flash或普通实现）
6. 输出投影 `o_proj`

### 4.2 GQA（Grouped Query Attention）

- Query头数多，KV头数少
- 通过重复KV头匹配Query头数
- 作用：降低KV Cache显存占用

### 4.3 KV Cache（推理加速关键）

自回归生成时，历史token的K/V可缓存复用，  
新token只需计算一次，避免重复算整段历史。

### 4.4 Causal Mask（防偷看未来）

训练/推理时，当前位置只能看当前及过去位置，  
不能看到未来token。

---

## 5. FeedForward：SwiGLU

本项目前馈层使用SwiGLU风格：

$$
\text{FFN}(x) = W_{down}\Big(\text{SiLU}(W_{gate}x)\odot (W_{up}x)\Big)
$$

解释：
- `gate_proj`：产生门控
- `up_proj`：产生内容
- 两者逐元素相乘后再 `down_proj` 回到 hidden size

---

## 6. Causal LM训练损失（Shift机制）

语言模型训练目标：第 $t$ 个位置预测第 $t+1$ 个token。

代码中常见写：

- `shift_logits = logits[..., :-1, :]`
- `shift_labels = labels[..., 1:]`

再做交叉熵：

$$
\mathcal{L}=\text{CrossEntropy}(\text{shift\_logits}, \text{shift\_labels})
$$

`ignore_index=-100` 表示该位置不参与loss计算（常用于padding）。

---

## 7. 常见易错点（本项目实战）

1. `flash_attn` 与 `flash_attention` 字段名不一致会报错  
2. `self.model(...)` 返回值数量必须和解包一致  
3. `attention_mask` 维度广播错误会导致shape mismatch  
4. `transpose`后直接`view`可能报非连续内存错误（先`contiguous()`或用`reshape`）

---

## 8. 一句话复习总结

- **RMSNorm**：只缩放，不去均值  
- **RoPE**：给Q/K做旋转，编码相对位置  
- **YaRN**：让RoPE更稳地支持长上下文  
- **GQA+KV Cache**：减少显存、提升推理速度  
- **Shift Loss**：用当前token预测下一个token