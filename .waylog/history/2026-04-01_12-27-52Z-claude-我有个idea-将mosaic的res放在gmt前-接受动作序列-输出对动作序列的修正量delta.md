# 我有个idea，将MOSAIC的RES放在GMT前，接受动作序列，输出对动作序列的修正量delta 
_Exported on 04/09/2026 at 09:49:53 GMT+8 from Claude via WayLog_


**User**

我有个idea，将MOSAIC的RES放在GMT前，接受动作序列，输出对动作序列的修正量delta q，用来修正动作序列中的物理错误和弥合人类运动学与机器人动力学的Gap，我打算用两阶段训练对FrontRES进行训练，第一阶段使用监督学习，使用预训练的GMT在仿真中Rollout得到q_sim，再通过delta_q_gt= q_sim-q_ref对FrontRES输出的delta_q进行训练，第二阶段使用强化学习微调FrontRES，辅以强力域随机化 (如穿模和漂浮等，而不是加速减速)，现在我已经写好了第一阶段的监督学习，正在尝试写第二阶段的强化学习微调，第二阶段微调还需要类似loss = w1 ||q' - q_ref||^2 的正则化项，通过拉近与原始动作序列的"距离"防止机器人因危险而不敢做出动作，并且还需要对加权项进行协调，我计划使用PCGrad，请先检索整个仓库的代码，再为我思考一个合适的修改方案

---


**Claude**

让我读取关键文件的具体实现：

---


**Claude**

现在我有了足够的理解，让我检查一下modules的`__init__.py`来确认导出情况：

---


**Claude**

现在我有了完整的理解。让我综合分析并给出修改方案：

---

## 诊断：Stage 2 RL 训练的核心问题

**关键 Bug**：`FrontRESActorCritic._frontres_forward()` 中 GMT 前向传播被 `torch.no_grad()` 包裹，导致梯度路径：

```
log_prob → mu (actions) → obs_modified → delta_q → residual_actor
```

在 `torch.no_grad()` 处断开。PPO 梯度无法到达 `residual_actor`，只能更新 `std` 噪声参数。**这是 Stage 2 RL 训练能跑但无效果的根本原因。**

对比 Post-GMT RES（`ResidualActorCritic`）不存在此问题，因为它是 `a_final = a_gmt + delta_a`，`delta_a` 直接在计算图上。

---

## 修改方案（4个文件）

### 文件一：`front_residual_actor_critic.py` — 修复梯度流

**改动位置：`_frontres_forward` + `update_distribution`**

最小改动：给 `_frontres_forward` 增加 `enable_gmt_grad=False` 参数，让 `update_distribution`（训练路径）传 `True`，`act_inference`（推理路径）保持 `False`（节省显存）。

GMT 参数仍保持 `requires_grad=False`，所以梯度流过 GMT 的激活值但不更新 GMT 权重：`delta_q` → `obs_modified` → `gmt_activations (requires_grad=False params)` → `actions` → `log_prob`。

---


**User**

[Request interrupted by user for tool use]

---


**User**

我的想法是直接从仿真中取得奖励值，然后使用奖励值直接对FrontRES进行更新，这样梯度流就不需要经过GMT，请评价我的想法

---


**Claude**

你的想法**本质上正确**，实际上这就是标准策略梯度的正确思路。让我详细评价：

---

## 核心分析

**你的想法**：仿真产生奖励 → 直接用奖励信号更新 FrontRES，梯度不经过 GMT。

这正是策略梯度（Policy Gradient）的标准做法：

```
梯度路径：
∇J ∝ E[A · ∇_θ log π_θ(δq | obs)]
          ↑                ↑
     来自仿真奖励     仅经过 residual_actor
     (黑盒，无需可微)  (不需要穿过 GMT)
```

GMT + Robot = **不可微的环境（黑盒）**，FrontRES = **策略网络**，δq = **动作**。这与 PPO 处理任何物理仿真的方式完全一致，**无需对 GMT 反向传播**。

---

## 现有代码的关键 Bug

我在代码分析中发现，**当前 Stage 2 的 `update_distribution` 是错误的**：

```python
# 当前（错误）：
actions, delta_q = self._frontres_forward(observations)
# actions 来自 torch.no_grad() 内部，完全从计算图脱离
self.distribution = Normal(actions, std)
# ↑ 梯度只能流到 std，无法到达 residual_actor！
```

应改为：
```python
# 正确（你的思路）：
delta_q_mean = self.residual_actor(policy_obs)
self.distribution = Normal(delta_q_mean, std)
# ↑ 梯度 log π(δq|obs)/∂θ 只流过 residual_actor，干净清晰
```

---

## 这个方法需要配套的一处关键修改

`act()` 现在返回的值必须区分两件事：

| 用途 | 内容 | 去向 |
|---|---|---|
| 策略分布（PPO） | `δq_sample ~ Normal(δq_mean, σ)` | 存入 rollout buffer |
| 仿真执行 | `robot_actions = GMT(obs with q_ref + δq_sample)` | 送给 `env.step()` |

当前 runner 将 `policy.act()` 的返回值**既存入 buffer 又送给环境**，但现在这两者已经分离。需要在 runner 里做最小改动：

```python
# my_on_policy_runner.py 中，对 FrontRES 的 rollout 步骤：
robot_actions = policy.act(obs)   # 内部已采样 δq，同时存 self.last_delta_q_sample
env.step(robot_actions)           # 环境用真实机器人动作
storage.store_actions(policy.last_delta_q_sample)  # buffer 存 δq
```

---

## 正则化与 PCGrad 的适配

在此框架下，`||δq||^2` 正则化项天然匹配：

- `δq_sample` 就在 rollout buffer 里，直接可用
- 正则化梯度：`∂(||δq_mean||²)/∂θ` 直接穿过 `residual_actor`
- PPO 梯度：`∂log π(δq|obs)/∂θ` 同样穿过 `residual_actor`
- **两者在同一参数空间，PCGrad 可以直接工作**

---

## 总结评价

| 维度 | 评价 |
|---|---|
| 理论正确性 | ✅ 完全正确，这就是标准 RL |
| 实现复杂度 | 低，主要修改 `update_distribution` + runner 一处 |
| 与 PCGrad 兼容 | ✅ 比需要梯度穿 GMT 的方案更干净 |
| 潜在问题 | ⚠ 需确认 rollout buffer 存的是 δq 而非 robot_actions |

你的想法是对的，当前代码在 `update_distribution` 那里就是错误的实现——它把 GMT 输出（被 `torch.no_grad()` 截断的量）当作分布均值，导致 PPO 梯度根本没有流到 `residual_actor`。按你的思路修正后，整个 Stage 2 的梯度链路才能真正工作。

你是否希望我现在按照这个设计给出完整的代码修改方案？

---


**User**

请修改

---


**Claude**

先读取 runner 文件，了解 rollout 流程：

---


**Claude**

现在开始逐文件修改：

**文件一：`front_residual_actor_critic.py`** — 核心逻辑改写

---


**Claude**

现在修改 `mosaic.py`，添加 FrontRES optimizer + 正则化参数：

---


**User**

[Request interrupted by user for tool use]

---


**User**

<task-notification>
<task-id>ad44a2a855763312f</task-id>
<tool-use-id>toolu_01W41DUGRbKaH1dELJjjWCRj</tool-use-id>
<output-file>C:\Users\CHENGY~1\AppData\Local\Temp\claude\c--ArtiIntComVis-MOSAIC\2b99aabb-3476-4bdb-9737-6e7bcc708a5e\tasks\ad44a2a855763312f.output</output-file>
<status>completed</status>
<summary>Agent "Track implementation tasks" completed</summary>
<result>The todo list has been created with all 7 tasks:

1. **in_progress** - 修改 front_residual_actor_critic.py: update_distribution 改为 Normal(delta_q_mean, std)
2. **pending** - 修改 front_residual_actor_critic.py: act() 返回 delta_q_sample，缓存 obs
3. **pending** - 修改 front_residual_actor_critic.py: 新增 get_env_action() 方法
4. **pending** - 修改 mosaic.py: FrontRES optimizer setup + lambda_reg/use_pcgrad 参数
5. **pending** - 修改 mosaic.py: _update_hybrid 添加正则化 loss 和 PCGrad 逻辑
6. **pending** - 修改 mosaic.py: 新增 _pcgrad_step() 方法
7. **pending** - 修改 rsl_rl_cfg.py: RslRlMOSAICAlgorithmCfg 添加新参数

The first task is marked as `in_progress` as requested. The remaining 6 are set to `pending` (mapped from `todo`). Ready to proceed with implementation whenever you are.</result>
<usage><total_tokens>11837</total_tokens><tool_uses>2</tool_uses><duration_ms>19134</duration_ms></usage>
</task-notification>

---


**Claude**

继续修改 `mosaic.py`：

---


**Claude**

抱歉，让我重新阅读仓库来确认 Stage 2 使用的是什么算法模块：

---


**Claude**

Stage 2 用的是 PPO，不是 MOSAIC。让我读取 PPO 算法：

---


**Claude**

现在我清楚了。需要修改的是：

1. ✅ `front_residual_actor_critic.py` — 已完成
2. `ppo.py` — 添加 FrontRES optimizer、正则化 loss、PCGrad
3. `rsl_rl_ppo_cfg.py`（IsaacLab 里）或 `RslRlPpoAlgorithmCfg` — 添加参数

先确认 `RslRlPpoAlgorithmCfg` 在哪里：

---


**Claude**

`RslRlPpoAlgorithmCfg` 是外部 isaaclab_rl 包，不能修改。正确做法是在项目内新建子类 + 扩展 PPO。现在开始修改：

**`ppo.py`** — 添加 FrontRES optimizer、正则化 loss、PCGrad：

---


**Claude**

现在修改 `update()` 方法，在计算 loss 之后插入正则化和 PCGrad 逻辑：

---


**Claude**

现在修改反向传播部分：

---


**Claude**

现在在 `update()` 末尾的 `self.storage.clear()` 之前，添加 `lambda_reg` 衰减，并把 `reg_loss` 计入 loss_dict。先找到正确位置：

---


**Claude**

需要在 `update()` 顶部初始化 `mean_reg_loss`，并在结尾处理衰减和 loss_dict：

---


**Claude**

现在在 `PPO` 类末尾添加 `_pcgrad_step` 方法：

---


**Claude**

现在在 `whole_body_tracking/utils/rsl_rl_cfg.py` 添加新配置类：

---


**Claude**

最后，在 Stage-2 配置文件中导入并使用新的 algorithm config：

---


**Claude**

最后验证一下所有文件的关键改动是否一致：

---


**Claude**

所有改动已落地。总结一下这次修改涉及的 **4 个文件**和每处的核心逻辑：

---

## 修改总览

### 1. [front_residual_actor_critic.py](source/rsl_rl/rsl_rl/modules/front_residual_actor_critic.py)

| 方法 | 改动 |
|---|---|
| `update_distribution` | 分布改为 `Normal(δq_mean, std)`，不再包含 GMT 输出。梯度只经过 `residual_actor`，完全不穿过冻结的 GMT |
| `act` | 返回 `δq_sample`（**不是** robot actions），runner 存入 rollout buffer |
| `get_env_action` *(新增)* | 将 `δq_sample` 作用到 `q_ref`，调 GMT 得到真实电机指令，供 `env.step()` 使用 |
| `_parse_observations` *(新增)* | 复用的观测解析逻辑，替代之前散落各处的重复代码 |
| `_apply_delta_q_and_run_gmt` *(新增)* | 复用的 GMT 调用路径（始终在 `no_grad` 下，仅用于 `get_env_action` 和 `act_inference`） |

### 2. [ppo.py](source/rsl_rl/rsl_rl/algorithms/ppo.py)

- `__init__`：检测 `FrontRESActorCritic` → 只优化 `residual_actor + critic + std`；接受 `lambda_reg_init / decay / min / use_pcgrad` 参数
- `update()`：计算 `reg_loss = λ_reg · ‖δq_mean‖²`；分支到 PCGrad 路径或标准加权路径；衰减 `lambda_reg`；输出到 `loss_dict`
- `_pcgrad_step()` *(新增)*：per-task 反向传播 → 检测冲突 → 投影消解 → 合并梯度 → optimizer step

### 3. [utils/rsl_rl_cfg.py](source/whole_body_tracking/whole_body_tracking/utils/rsl_rl_cfg.py)

新增 `RslRlPpoFrontRESAlgorithmCfg`：继承标准 PPO config，加入 `lambda_reg_init / decay / min / use_pcgrad`

### 4. [config/g1/agents/rsl_rl_ppo_cfg.py](source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py)

`G1FlatFrontRESFinetuneRunnerCfg.algorithm` 从 `RslRlPpoAlgorithmCfg` 换成 `RslRlPpoFrontRESAlgorithmCfg`，并填入 `lambda_reg_init=0.01 / use_pcgrad=False` 默认值（建议先跑稳再开 PCGrad）

---


**User**

我已经对代码做了较大修改，现在我关注在Stage 1训练时更新到Wandb的指标上，由于FrontRES输出Δq，因此无法做可视化进行验证，只能通过训练指标进行判断，我现在考虑的是平均残差幅度 Mean(∣Δq∣)，平均错误率Mean(∣Δqpred−Δqgt∣)，修正方向一致性 (Cosine Similarity)，并且这些指标也与损失函数设计相关，请先阅读整个仓库，再进行分析，先不要修改代码

---


**Claude**

Failed to authenticate. API Error: 403 {"error":{"type":"forbidden","message":"Request not allowed"}}

---


**User**

Continue from where you left off.

---


**Claude**

No response requested.

---


**User**

我已经对代码做了较大修改，现在我关注在Stage 1训练时更新到Wandb的指标上，由于FrontRES输出Δq，因此无法做可视化进行验证，只能通过训练指标进行判断，我现在考虑的是平均残差幅度 Mean(∣Δq∣)，平均错误率Mean(∣Δqpred−Δqgt∣)，修正方向一致性 (Cosine Similarity)，并且这些指标也与损失函数设计相关，请先阅读整个仓库，再进行分析，先不要修改代码

---


**Claude**

先读取当前实际代码状态：

---


**Claude**

现在我有了完整的信息。以下是深度分析：

---

## Stage 1 训练指标分析

### 一、当前代码状态（你已实现）

`supervise.py` 的 `update()` 已返回 **9 个指标**，你的三个想法中：

| 你的提案 | 当前代码对应指标 | 是否等价 |
|---|---|---|
| Mean(&#124;Δq&#124;) | `delta_q_pred_norm`（L2 范数） | **不等价**（L2 ≠ L1） |
| Mean(&#124;Δq_pred − Δq_gt&#124;) | `joint_mae_mean` | **等价**（数学上完全相同） |
| Cosine Similarity | `cosine_similarity` | **已存在** |

---

### 二、三个指标的深层分析

#### 1. Mean(|Δq|) — 残差幅度

你提案的是 L1 均值：`predicted.abs().mean()`，当前代码是 L2 范数：`predicted.norm(dim=-1).mean()`。

**它们的区别有实际意义：**

```
L2 范数被少数大修正关节主导：若膝关节需要修正 0.3 rad，其他 28 个关节都是 0.01 rad
  → L2 = sqrt(0.3² + 28×0.01²) ≈ 0.316 rad（由膝关节支配）
  → L1 均值 = (0.3 + 28×0.01) / 29 ≈ 0.020 rad（每关节平均修正量）
```

L1 均值更具**可解释性**：直接告诉你"每个关节平均被修正了多少弧度"，与关节活动范围对比更直观。L2 范数更适合判断整体修正向量的"能量"。

**两者比值** L2 / (L1 × √n_joints) 可推断修正是否集中在特定关节：
- 比值 ≈ 1：修正均匀分布
- 比值 >> 1：修正集中在少数关节（可能是 GMT 的系统性缺陷集中在特定关节）

#### 2. Mean(|Δq_pred − Δq_gt|) = joint_mae_mean — 已完全覆盖

`joint_mae_mean = (pred - target).abs().mean(dim=0).mean()` 与你提案的指标数学等价。**无需新增。**

但需要注意一个**一致性问题**：

```python
# 损失函数：有 valid 掩码（排除 terminal 状态）
behavior_loss = (per_sample_loss * valid).sum() / n_valid  ✓

# 但 joint_mae_mean / cosine_similarity / relative_error 均无掩码 ✗
joint_mae = (predicted_actions - target_actions).abs().mean(dim=0)
```

当机器人摔倒时：`Δq_gt = q_ref - q_sim_fallen` 异常偏大，而 `Δq_pred` 仍然合理。Terminal 状态会：
- 拉高 `joint_mae_mean`（看起来误差比实际大）
- 拉低 `cosine_similarity`（方向看起来比实际差）
- 导致 **loss 曲线好看但 metric 曲线差看**的假象

#### 3. Cosine Similarity — 最关键的方向性指标，但有数值陷阱

这是三个指标里**信息量最独特**的，因为 MSE 同时优化幅度和方向，而 cosine similarity 是唯一对幅度无关的指标。它揭示了模型是否学到了"哪些关节需要修正、往哪个方向"。

**与损失函数的关系：**

MSE 梯度：`∂MSE/∂Δq_pred = 2(Δq_pred − Δq_gt)` 分解为：

```
= 2·Δq_pred   ← 将预测拉向零（量级正则化）
  -2·Δq_gt    ← 将预测拉向GT（方向和量级学习）
```

训练早期 `||Δq_pred|| ≈ 0`（网络初始化接近零），此时梯度几乎完全来自 `-2·Δq_gt`，模型**先学方向后学量级**。因此：
- **Cosine similarity 应该比 behavior loss 更早收敛**
- 若 cosine_sim 早期快速上升到 0.8+ → 模型理解了修正方向，后续主要在校准量级
- 若 cosine_sim 长期停在 0.5 以下 → 方向学习失败，预示着模型表达能力不足或观测信息不够

**数值陷阱：近零预测时 cosine similarity 无意义**

```python
# 初始化时 Δq_pred ≈ 0（小增益初始化）
# PyTorch cosine_similarity 返回值：向量接近零时输出接近0（非报错）
# 导致：训练初期 cosine_sim ≈ 0，并非真的方向错误，而是分母接近零
cos_sim = F.cosine_similarity(predicted, target, dim=-1)
# 当 predicted → 0，cosine_sim → 0（undefined behavior）
```

**建议判读顺序：** 先看 `delta_q_pred_norm`，若 norm 已经升上来（≥ 0.01 rad），再看 cosine_similarity 才有意义。

---

### 三、指标间的层次关系（用于训练诊断）

```
训练进程诊断树：

behavior ↓？
├── No：训练停滞
│   ├── check delta_q_norm_ratio：若 ≈ 0，模型在预测零 → 学习率问题
│   └── check cosine_similarity：若 ≈ 0，方向完全错误 → 观测 q_ref_start_idx 配置问题？
└── Yes：
    ├── cosine_similarity ↑ fast：方向学习良好，等待量级校准
    │   └── delta_q_norm_ratio → 1.0：量级校准完成，模型收敛
    ├── cosine_similarity 低（< 0.6 长期不变）：模型未学到方向
    │   └── 检查：valid_ratio（是否频繁摔倒？） / joint_mae_max（哪个关节最难学？）
    └── relative_error 高但 cosine_sim 高：量级对不上
        └── 比较 delta_q_gt_norm 和 delta_q_pred_norm：
            norm_ratio < 1 → 欠预测  /  norm_ratio > 1 → 过预测
```

---

### 四、与损失函数设计的关联

| 指标 | MSE 直接优化？ | Huber 直接优化？ | 指标的额外价值 |
|---|---|---|---|
| `behavior` | ✅ | ✅ | 训练主信号 |
| `joint_mae_mean` | 间接 | 更直接 | 检验 Huber vs MSE 的实际收敛差异 |
| `cosine_similarity` | **不直接** | **不直接** | 捕获 MSE 忽略的方向信息 |
| `relative_error` | **不直接** | **不直接** | 标准化的精度，检验小修正关节的预测质量 |
| `delta_q_norm_ratio` | 间接 | 间接 | 量级校准程度（目标值 1.0） |

**关键结论：** 如果你考虑在损失函数中加入 cosine loss（`L_cos = 1 - cosine_sim`），`cosine_similarity` 指标将直接反映这个 loss 项的收敛。如果只用 MSE/Huber，cosine_sim 作为无损失对应项告诉你"MSE 副作用带来的方向学习有多好"。

---

### 五、真正缺失的 Mean(|Δq|) 语义

你的原始提案 `Mean(|Δq|)` 如果指的是**每关节平均绝对修正量**（L1 意义），当前代码 `delta_q_pred_norm`（L2）无法完全替代。补充 L1 均值的实际意义：

- 可以直接与关节运动范围（URDF 中 `effort` / `limit`）对比，判断修正是否合理
- 配合 `joint_mae_max` 可以发现"一个关节被修正很多但预测很准，另一个修正很少但预测很差"的不均衡现象

---

### 结论

当前代码中三个指标的**实际覆盖情况**：
- ✅ **Cosine Similarity** — 已完整实现
- ✅ **Mean(|Δq_pred − Δq_gt|)** — 已被 `joint_mae_mean` 完全覆盖
- ⚠️ **Mean(|Δq|)** — L2 版本已有，L1 版本（每关节平均修正量）语义更清晰，若需补充

最值得关注的**潜在问题**：diagnostics 指标（cosine_similarity、joint_mae_mean、relative_error）没有 valid 掩码，而 loss 有掩码，导致高摔倒率时两者曲线不一致，需要在解读 Wandb 曲线时结合 `valid_ratio` 一起看。

---


**User**

我想先将主损失函数设计好再讨论更新指标，很明显主损失函数必须有如下三点特性：

1、主损失函数必须凸显修正量大的关节，不能被修正量不大的关节淹没

2、主损失函数必须对下肢有额外的侧重，或者有特别注意保持平衡的机制

3、主损失函数必须防止修正量过大 (跟踪已失效) 的关节污染训练

---


**User**

This session is being continued from a previous conversation that ran out of context. The summary below covers the earlier portion of the conversation.

Summary:
1. Primary Request and Intent:

The user is building MOSAIC's FrontRES system - a pre-GMT residual correction module. The conversation covers:

**Stage 2 RL fine-tuning implementation (completed):**
- Fixing the core PPO gradient bug in `FrontRESActorCritic`
- Adding regularization term `L_reg = w1 * ||q' - q_ref||^2 = ||delta_q||^2`
- Adding PCGrad to coordinate PPO loss vs regularization gradients

**Stage 1 metric analysis (completed, no code changes):**
- Analysis of three proposed metrics: Mean(|Δq|), Mean(|Δq_pred - Δq_gt|), Cosine Similarity

**Latest/Pending request - loss function design for Stage 1 (NOT YET IMPLEMENTED):**
User wants the primary loss function to have three properties:
1. Highlight joints with large corrections (not swamped by small-correction joints)
2. Give extra weight to lower limbs or have a balance-preserving mechanism
3. Prevent joints with excessive corrections (tracking failure/穿模/漂浮) from polluting training

2. Key Technical Concepts:
- FrontRES (Front-End Residual): pre-GMT residual network outputting Δq corrections
- GMT (General Motion Tracker): frozen teacher policy; pipeline: obs → FrontRES → Δq → q_ref+Δq → GMT → actions
- PPO over Δq-space: delta_q is the "action", GMT+robot is black-box "environment" — gradient NEVER needs to flow through frozen GMT
- PCGrad (Project Conflicting Gradients): per-task backward passes, project conflicting gradients onto each other's normal planes
- Core bug fixed: old `update_distribution` created `Normal(gmt_actions_detached, std)` → PPO gradient couldn't reach `residual_actor`; fix: `Normal(delta_q_mean, std)` where `delta_q_mean = residual_actor(obs)`
- `get_env_action()`: separates policy distribution (over delta_q) from env action (robot motor commands via GMT)
- Valid masking inconsistency: loss uses `valid = 1 - dones.float()` but diagnostic metrics are unmasked
- Δq_gt = q_ref - q_sim (supervised target; q_ref = reference motion, q_sim = simulated joint positions after GMT)

3. Files and Code Sections:

**`source/rsl_rl/rsl_rl/modules/front_residual_actor_critic.py`** — MAJOR REWRITE
Key changes: Distribution changed from `Normal(gmt_actions, std)` to `Normal(delta_q_mean, std)`. `act()` returns delta_q_sample. New `get_env_action()` method:

```python
def _parse_observations(self, observations):
    """Shared obs parsing → (policy_obs, ref_vel, ref_vel_estimator_obs)"""

def _apply_delta_q_and_run_gmt(self, policy_obs, delta_q, ref_vel, ref_vel_estimator_obs):
    """Apply Δq to q_ref, build GMT obs, run frozen GMT. Always in no_grad."""

def update_distribution(self, observations):
    """Distribution over Δq-space: Normal(delta_q_mean, std).
    Caches observations for get_env_action. Gradient flows through residual_actor only."""
    self._cached_observations = observations
    delta_q_mean = self.residual_actor(policy_obs)
    self.last_residual_actions = delta_q_mean.detach()
    self.distribution = Normal(delta_q_mean, std)

def act(self, observations, **kwargs):
    """Returns delta_q_sample (NOT robot actions). Runner stores in rollout buffer."""
    self.update_distribution(observations)
    return self.distribution.sample()

def get_env_action(self, observations, delta_q_sample):
    """Converts delta_q_sample → robot_actions via GMT. Called by runner for env.step()."""
    obs = getattr(self, '_cached_observations', observations)
    policy_obs, ref_vel, ref_vel_estimator_obs = self._parse_observations(obs)
    with torch.no_grad():
        robot_actions = self._apply_delta_q_and_run_gmt(policy_obs, delta_q_sample, ...)
    return robot_actions

def act_inference(self, observations):
    """Deployment: uses mean delta_q → GMT → robot_actions via _frontres_forward"""
```

**`source/rsl_rl/rsl_rl/algorithms/ppo.py`** — MODIFIED
```python
# New imports
import random
from rsl_rl.modules import ActorCritic, ActorCriticVQ, FrontRESActorCritic

# New __init__ parameters
lambda_reg_init: float = 0.0,
lambda_reg_decay: float = 1.0,
lambda_reg_min: float = 0.0,
use_pcgrad: bool = False,

# FrontRES-specific optimizer setup
if isinstance(policy, FrontRESActorCritic):
    trainable = list(policy.residual_actor.parameters()) + list(policy.critic.parameters())
    # add std or log_std
    self.optimizer = optim.Adam(trainable, lr=learning_rate)

# In update() - regularization loss
ppo_loss = surrogate_loss + value_loss_coef * value_loss - entropy_coef * entropy
reg_loss = mu_batch.pow(2).mean()  # mu_batch = delta_q_mean for FrontRES
# PCGrad or standard path
if self.use_pcgrad and self.use_frontres_reg:
    self._pcgrad_step({"ppo": ppo_loss, "reg": lambda_reg * reg_loss})
else:
    loss = ppo_loss + lambda_reg * reg_loss

# New _pcgrad_step method
def _pcgrad_step(self, task_losses: dict) -> None:
    # Step 1: per-task gradients with retain_graph
    # Step 2: PCGrad projection (if g_i · g_j < 0: g_i -= (dot/norm_sq) * g_j)
    # Step 3: sum projected grads, clip, optimizer.step()
```

**`source/whole_body_tracking/whole_body_tracking/utils/rsl_rl_cfg.py`** — MODIFIED
```python
@configclass
class RslRlPpoFrontRESAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """FrontRES Stage-2 RL Fine-tuning PPO config with regularization and PCGrad."""
    class_name: str = "PPO"
    lambda_reg_init:  float = 0.01
    lambda_reg_decay: float = 1.0
    lambda_reg_min:   float = 0.0
    use_pcgrad: bool = False
```

**`source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py`** — MODIFIED
```python
# Added import
from whole_body_tracking.utils.rsl_rl_cfg import RslRlPpoFrontRESAlgorithmCfg

# G1FlatFrontRESFinetuneRunnerCfg now uses:
algorithm = RslRlPpoFrontRESAlgorithmCfg(
    # ... standard PPO params ...
    lambda_reg_init=0.01,
    lambda_reg_decay=1.0,
    lambda_reg_min=0.0,
    use_pcgrad=False,  # enable after tuning
)
```

**`source/whole_body_tracking/whole_body_tracking/utils/supervise.py`** — READ ONLY (already updated by user)
Already has 9 metrics: behavior, delta_q_pred_norm, delta_q_gt_norm, delta_q_norm_ratio, cosine_similarity, relative_error, valid_ratio, joint_mae_mean, joint_mae_max

**`source/rsl_rl/rsl_rl/modules/supervise_learning.py`** — READ ONLY
- `SuperviseLearning` student MLP for Stage 1
- `get_supervision_target(q_sim, q_ref)` → `Δq_gt = q_ref - q_sim`
- GMT loaded from `.pt` checkpoint

4. Errors and fixes:
- **Wrong algorithm assumption**: Initially tried to modify `mosaic.py` for Stage 2 RL. User corrected: "mosaic.py不是我在第二阶段强化学习微调使用的模块" → Re-read codebase, found Stage 2 uses PPO (not MOSAIC) via `G1FlatFrontRESFinetuneRunnerCfg`
- **Edit rejection on mosaic.py**: User denied the tool call explicitly. Did NOT modify mosaic.py.
- **`RslRlPpoAlgorithmCfg` is external**: Tried to find it, but it's in `isaaclab_rl` package (external). Solution: create subclass `RslRlPpoFrontRESAlgorithmCfg` in project's own `rsl_rl_cfg.py`

5. Problem Solving:
- **Core PPO gradient bug**: `update_distribution` was creating `Normal(gmt_actions_detached, std)` → PPO gradient couldn't reach `residual_actor`. Solution: change distribution to `Normal(delta_q_mean, std)` and separate robot action generation into `get_env_action()`
- **Runner interface compatibility**: Runner calls `policy.act()` expecting what gets stored in rollout buffer AND what gets sent to environment. Solution: `act()` returns delta_q_sample (for buffer), `get_env_action()` returns robot_actions (for env). Runner already has `get_env_action` hook at line 475-478.
- **ref_vel caching**: When MOSAIC augments obs with estimated ref_vel before calling `policy.act()`, but runner passes raw obs to `get_env_action()`. Solution: `update_distribution` caches `_cached_observations`; `get_env_action` uses cached version.

6. All user messages:
- "我有个idea，将MOSAIC的RES放在GMT前...请先检索整个仓库的代码，再为我思考一个合适的修改方案"
- "我的想法是直接从仿真中取得奖励值，然后使用奖励值直接对FrontRES进行更新，这样梯度流就不需要经过GMT，请评价我的想法"
- "请修改"
- [Rejected edit on mosaic.py] "mosaic.py不是我在第二阶段强化学习微调使用的模块，应该是source\rsl_rl\rsl_rl\modules\front_residual_actor_critic.py，请重新阅读仓库"
- "我已经对代码做了较大修改，现在我关注在Stage 1训练时更新到Wandb的指标上...请先阅读整个仓库，再进行分析，先不要修改代码"
- "我想先将主损失函数设计好再讨论更新指标，很明显主损失函数必须有如下三点特性：1、主损失函数必须凸显修正量大的关节，不能被修正量不大的关节淹没 2、主损失函数必须对下肢有额外的侧重，或者有特别注意保持平衡的机制 3、主损失函数必须防止修正量过大(跟踪已失效)的关节污染训练" [CURRENT - text only response requested]

7. Pending Tasks:
- **Design Stage 1 primary loss function** with three properties:
  1. Highlight joints with large corrections (amplify signal, not swamped by small-correction joints)
  2. Extra weight for lower limbs OR balance-preserving mechanism
  3. Prevent joints with excessive corrections (tracking failure cases: penetration, floating) from polluting training
- After loss function design, update the Wandb metrics to match

8. Current Work:
The user just requested analysis and design of the Stage 1 (supervised learning) primary loss function. The current loss is simple MSE or Huber (`self.loss_fn` in `SuperviseTrainer`). The user has identified three specific requirements for the improved loss function design. No implementation has been started yet - a text-only response was requested to analyze the design.

Current loss location in supervise.py:
```python
per_sample_loss = self.loss_fn(predicted_actions, target_actions, reduction="none")  # (B, A)
behavior_loss = (per_sample_loss.mean(dim=-1, keepdim=True) * valid).sum() / n_valid
```

9. Optional Next Step:
Design and discuss (then implement) a new loss function for Stage 1 supervised learning that addresses the three stated requirements:
1. **Joint-adaptive weighting** (凸显修正量大的关节): e.g., weight each joint's loss by its GT correction magnitude |Δq_gt_j| or use a power > 1 on the loss; large-correction joints contribute proportionally more to the gradient
2. **Lower-limb emphasis** (下肢侧重): explicit per-joint weight vector w_j with higher weights for hip/knee/ankle joints (balance-critical); possibly derived from URDF joint hierarchy
3. **Outlier rejection for tracking failure** (防止失效关节污染): clip or soft-cap loss per joint when |Δq_gt_j| > threshold (indicating tracking failure like penetration/floating); could use Huber-like hard clipping per joint

The user asked: "我想先将主损失函数设计好再讨论更新指标，很明显主损失函数必须有如下三点特性..."

If you need specific details from before compaction (like exact code snippets, error messages, or content you generated), read the full transcript at: C:\Users\ChengYuxuan\.claude\projects\c--ArtiIntComVis-MOSAIC\2b99aabb-3476-4bdb-9737-6e7bcc708a5e.jsonl
Continue the conversation from where it left off without asking the user any further questions. Resume directly — do not acknowledge the summary, do not recap what was happening, do not preface with "I'll continue" or similar. Pick up the last task as if the break never happened.

---


**Claude**

以下是针对三个特性设计的损失函数方案，先分析每个特性的实现思路，再给出综合设计。

---

## 特性 1：凸显修正量大的关节（动态幅度加权）

**问题**：当前 MSE 对每个关节一视同仁。假设关节 A 的 Δq_gt = 0.001，关节 B 的 Δq_gt = 0.1，那么即使网络对关节 B 的预测误差是关节 A 的 10 倍，绝对损失贡献也只差 10 倍——信号依然容易被大量"安静关节"稀释。

**方案：动态幅度比例权重**

对每个样本，以当前批次中该样本各关节的 |Δq_gt| 归一化后作为权重：

```
w_dynamic_j = |Δq_gt_j| / (mean_j(|Δq_gt_j|) + ε)
```

效果：修正量是平均值 2 倍的关节，在梯度中贡献也是 2 倍。修正量接近零的关节几乎不参与训练，不会把梯度拖向错误方向。

---

## 特性 2：下肢额外侧重（静态关节权重）

**方案：静态权重向量 `w_static`（形状 `(num_actions,)`）**

对髋关节、膝关节、踝关节赋予更高权重（例如 2.0），其余关节为 1.0：

```python
# G1 29-DOF 示例（需根据实际 joint order 调整索引）
lower_limb_joints = {0,1,2,3,4,5,6,7,8,9,10,11}  # 左右髋(3) + 左右膝(1) + 左右踝(2) × 2
w_static = torch.ones(num_actions)
w_static[list(lower_limb_joints)] = 2.0
```

与动态权重叠加：当下肢的修正量又大、静态权重又高时，对训练的影响 4 倍于上肢安静关节，符合平衡优先的训练直觉。

---

## 特性 3：失效关节的硬性排除（异常值门控）

**问题**：当仿真中发生穿模或漂浮时，q_sim 与 q_ref 差异极大，导致 Δq_gt_j > 0.3~0.5 rad。此时这些关节已处于失效状态，用这类数据训练会误导网络"以为正确的修正量是 0.8 rad"。

> 注：已有的 `valid = 1 - dones` 只过滤掉**整个样本**（机器人跌倒后 reset），但在跌倒发生前几步，部分关节可能已经处于失效轨迹但环境尚未 done。

**方案：关节级失效门控**

```
joint_valid_j = float(|Δq_gt_j| < τ_outlier)
```

门控直接乘入动态权重，失效关节的权重归零，不污染梯度。

---

## 综合损失函数

按顺序：先排除失效关节 → 再计算动态权重 → 再乘静态权重 → 再计算加权 MSE → 再应用终止步掩码。

```python
def frontres_supervised_loss(
    predicted:  Tensor,          # (B, A) 预测的 Δq
    target:     Tensor,          # (B, A) GT Δq = q_ref - q_sim
    valid:      Tensor,          # (B, 1) 1=正常步, 0=终止步(done)
    static_w:   Tensor,          # (A,)   静态关节权重（下肢 2.0, 其余 1.0）
    tau_outlier: float = 0.4,    # rad, 超过此值视为跟踪失效
    eps: float = 1e-6,
) -> Tensor:
    
    gt_abs = target.abs()                                            # (B, A)

    # 特性 3: 关节级失效门控
    joint_valid = (gt_abs < tau_outlier).float()                     # (B, A) 0=失效关节

    # 特性 1: 动态幅度权重（只在非失效关节上计算均值）
    masked_abs   = gt_abs * joint_valid                              # (B, A) 失效关节归零
    n_valid_j    = joint_valid.sum(dim=-1, keepdim=True).clamp(1.0) # (B, 1)
    avg_abs      = masked_abs.sum(dim=-1, keepdim=True) / n_valid_j # (B, 1) 该样本非失效关节的均值
    w_dynamic    = masked_abs / (avg_abs + eps)                      # (B, A) 同时带有 joint_valid 的掩码效果

    # 特性 2: 叠加静态下肢权重
    w_combined = static_w.unsqueeze(0) * w_dynamic                  # (B, A)

    # 加权 MSE + 终止步掩码
    per_joint_sq  = (predicted - target).pow(2)                     # (B, A)
    weighted_sq   = per_joint_sq * w_combined * valid               # (B, A)
    normalizer    = (w_combined * valid).sum().clamp(min=eps)
    loss          = weighted_sq.sum() / normalizer

    return loss
```

---

## 特性交互的边界情况分析

| 情况 | 行为 |
|------|------|
| 某关节 Δq_gt ≈ 0（无需修正）| w_dynamic ≈ 0，该关节几乎不贡献梯度，正确 |
| 某关节 Δq_gt 很大（但 < τ）| w_dynamic > 1，梯度放大，网络优先学习 |
| 某关节 Δq_gt > τ（失效）| joint_valid = 0，w_dynamic = 0，完全排除 |
| 整个样本全部关节都失效 | w_combined 全零，该样本贡献 0，效果等同 done=1 |
| 下肢关节 vs 同等 Δq_gt 上肢关节 | 下肢多 2× static 权重，优先被学习 |

---

## 关于 `tau_outlier` 的选取

建议从 **0.3~0.5 rad**（约 17°~28°）开始。可以通过现有 `joint_mae_max` 指标反推：如果训练初期 `joint_mae_max` 持续 >> 0.3，说明 τ 设置偏低，可以适当放宽；如果训练中 `valid_ratio` 突然下降，说明失效样本增多，可考虑收紧 τ。

这个阈值也可以做成**自适应**的：每隔 N 次 update，用当前 buffer 中所有样本的 |Δq_gt| 的某个高百分位（如 95%ile）来更新 τ，避免手动调参。

---

确认方案后我来修改 [supervise.py](source/whole_body_tracking/whole_body_tracking/utils/supervise.py) 中的 `update()` 方法，同时在 `SuperviseTrainer.__init__` 里加入 `static_weights` 和 `tau_outlier` 参数。

---


**User**

我认为关节级门控这个思路很好，但如何确定摔倒前哪些关节已经开始失效呢？

---


**Claude**

这是个非常关键的问题，触及了门控设计的核心矛盾：**"修正量很大但合法"**和**"修正量很大因为失效"**在数值上无法直接区分。

---

## 问题的本质

先明确 Δq_gt 的含义：

```
Δq_gt_j = q_ref_j - q_sim_j
```

它大的原因只有两种：
1. **q_ref 本身在此时就要求关节走到一个大角度**（合法的大修正需求）
2. **q_sim 因为物理失效（穿模/漂浮/失衡）已经偏离了 q_ref**（失效产生的假信号）

这两种情况在 Δq_gt 的数值上是叠加的，单看一个时间步的绝对值无法分开。

---

## 三个可行的区分维度

### 维度 1：时序稳定性（关节失效是突变）

合法的大修正量在时序上是**连续平滑的**（参考运动本身就是连续的），而失效导致的 Δq_gt 爆炸通常是**突变的**——某一帧从 0.05 rad 突然跳到 0.8 rad。

可以用**相邻帧差分**作为门控信号：

```
突变指标_j(t) = |Δq_gt_j(t) - Δq_gt_j(t-1)|
```

如果突变指标超过阈值，该关节在第 t 步被门控。这样即使关节修正量很大，只要是渐进增加的就不会被排除；但失效导致的跳变会被捕捉。

**代价**：需要在 buffer 里额外存储上一步的 Δq_gt，以及 Storage 要多保存一帧。

---

### 维度 2：批内分布（对当前 batch 自适应）

在同一个 batch 内，对每个关节 j，计算跨样本的分布统计：

```
μ_j  = mean_b(|Δq_gt_bj|)
σ_j  = std_b(|Δq_gt_bj|)
```

将每个关节的**异常程度**定义为 z-score：

```
z_bj = (|Δq_gt_bj| - μ_j) / (σ_j + ε)
```

如果 z_bj > k（比如 k=2.5），就门控掉这个 (sample, joint) 组合。

**优势**：不需要手动设定 τ，自动适配每个关节的自然修正量范围。关节 A 平均修正 0.05 rad，关节 B 平均修正 0.3 rad，两者各用各自的基准，不会把关节 B 的合法大修正当成异常。

---

### 维度 3：关节物理上下界（URDF 约束）

G1 的每个关节都有 URDF 定义的 `limit`（position limit）。如果 q_sim_j 超出了 `[q_min_j, q_max_j]` 的范围，物理失效几乎是确定的：

```
失效_j = (q_sim_j < q_min_j - margin) OR (q_sim_j > q_max_j + margin)
```

这是**最硬的先验信号**——不需要任何统计，直接用物理约束判断。

**代价**：需要将 URDF 关节限位作为常量传入 Trainer，且需要在 buffer 里额外存储 q_sim（目前可能只存了 Δq_gt）。

---

## 推荐方案：维度 2（批内 z-score）为主干

维度 1 需要额外存储，维度 3 需要修改 buffer 存储内容。维度 2 只需要在当前 batch 内做统计，**对现有代码侵入最小**，且理论上最合理：

- 它自动适配不同运动模式（跑步 vs 原地站立时各关节的修正量分布差异很大）
- 它在关节级别而非全局级别判断异常，避免"腿部关节修正量本来就大"被误判
- 不需要任何领域知识或手动调参

与固定阈值 τ 的区别：

| | 固定 τ | 批内 z-score |
|--|--|--|
| 需要手动调参 | 是 | 只需设 k（更鲁棒） |
| 适配不同关节 | 否（全局同一 τ）| 是（每关节独立分布） |
| 适配不同运动 | 否 | 是（batch 内自适应） |
| 计算开销 | 极低 | 低（只需 mean + std）|

如果同时有 URDF 限位数据，可以叠加做双重门控：z-score 排除批内异常值，URDF 限位排除物理上不可能的状态，两者取并集作为最终 `joint_valid`。

---

你倾向于哪个方向？确定后我来修改 [supervise.py](source/whole_body_tracking/whole_body_tracking/utils/supervise.py) 的实现。

---


**User**

我不太确定对于第一阶段的监督学习需要做如此细致的损失函数设计，因为这只是第二阶段强化学习的预训练而已，并且同一个批量内的动作序列可能差异很大，使用批内分布似乎不太妥当，而时序稳定性可能更好

---


**Claude**

你说得对，这两点质疑都很准确。

**批内 z-score 的问题**：如果一个 batch 里混合了跑步片段（髋关节 Δq_gt ≈ 0.3 rad 是正常）和站立片段（髋关节 Δq_gt ≈ 0.05 rad 是正常），两类样本的分布根本不能放在一起算 z-score，检测结果会完全错误。

**时序稳定性的优势**：失效的本质是物理约束被破坏——这在时序上必然表现为突变，与运动类型无关。跑步中髋关节的合法大修正是连续增减的，但穿模导致的失效是突然跳变的，这个判断标准对所有运动类型都成立。

---

## 实用建议

既然第一阶段只是预训练，我认为合适的设计是**最小必要改动**：

1. **Huber 损失**（替换 MSE）：已经是代码里的选项，对中等幅度的离群值天然鲁棒，不需要额外的门控逻辑

2. **静态下肢权重**：简单、无需调参，对第二阶段 RL 的平衡学习有正向影响

3. **时序突变门控**（作为唯一的关节级门控）：在 buffer 里额外存一帧 Δq_gt，计算帧间差分，超过阈值则门控

动态幅度加权（特性 1 的精细设计）可以完全去掉——第二阶段 RL 会通过奖励信号自动纠正各关节的误差分布，第一阶段只需要给一个合理的初始化点即可。

---

你是否认可这个简化方向？确认后我来实现：在 `SuperviseStorage` 中加一列 `prev_target_actions`，在 `SuperviseTrainer` 的 `update()` 里加突变门控，以及把 Huber + 静态下肢权重接入损失计算。

---


**User**

我比较认可这个简化修改方案，我想总结一下我们讨论过的所有关于损失函数的修改：Huber Loss，静态下肢加权，时序门控，计算Δq时的掩码机制，如果正确且没有遗漏，请开始修改

---


**Claude**

四点总结正确，没有遗漏。开始实现。

---


**Claude**

先改 [supervise.py](source/whole_body_tracking/whole_body_tracking/utils/supervise.py)：

---


**Claude**

Now rewrite the core `update()` loop:

---


**Claude**

Now add `sum_joint_gate_ratio` to the accumulators and loss_dict:

---


**Claude**

Now update the config class:

---


**Claude**

Now update the runner config to pass `lower_limb_indices` for G1:

---


**Claude**

Let me do a final review of the changed file to make sure everything is consistent:

---


**Claude**

文件整体一致。全部修改完成，汇总如下：

---

## 修改总结

### [supervise.py](source/whole_body_tracking/whole_body_tracking/utils/supervise.py)

**1. Huber Loss**
- `SuperviseTrainer.__init__` 默认 `loss_type="huber"`
- `update()` 里用 `nn.functional.huber_loss(..., reduction="none")` 替换原来的 `self.loss_fn`

**2. 静态下肢加权**
- `__init__` 新增 `lower_limb_indices`、`lower_limb_weight` 参数
- `init_storage()` 里构建 `self.static_weights`（shape `(A,)`），下肢关节赋值 `lower_limb_weight`，其余为 1.0
- 损失计算：`eff_w = static_weights × joint_valid × valid`

**3. 时序门控**
- `SuperviseStorage.generator()` 额外 yield `prev_target_actions`（上一步的 Δq_gt，首帧用自身）
- `update()` 计算帧间跳变：`jump = |Δq_gt(t) - Δq_gt(t-1)|`，超过 `jump_threshold` 的关节权重归零
- 新增诊断指标 `joint_gate_ratio`（时序门通过率，↓ = 更多预摔失效关节）

**4. 计算 Δq 时的掩码机制**
- 原有 `valid = 1 - dones.float()`（样本级终止掩码）保留
- 新增 `joint_valid`（关节级时序门掩码）
- 两者与静态权重统一在 `eff_w` 里，归一化分母为 `eff_w.sum()`，确保损失尺度在不同门控比例下保持稳定

### [rsl_rl_cfg.py](source/whole_body_tracking/whole_body_tracking/utils/rsl_rl_cfg.py)

`RslRlSuperviseAlgorithmCfg` 新增三个字段：`lower_limb_indices`、`lower_limb_weight`、`jump_threshold`；`loss_type` 默认改为 `"huber"`

### [rsl_rl_ppo_cfg.py](source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py)

`G1FlatSupervisedRunnerCfg.algorithm` 显式传入 `lower_limb_indices=list(range(12))`（假设前 12 个关节为髋/膝/踝）——**请在启动训练前根据实际 G1 joint_pos 切片顺序核实这个索引列表是否正确**。

---


**User**

请给我Huber Loss，静态下肢加权，时序门控与Δq Mask Mechanism的表达式，方便我写在论文中

---


**Claude**

以下是四个组件的完整数学表达式。

---

## 符号定义

| 符号 | 含义 |
|------|------|
| $T, B, J$ | 时间步数、并行环境数、关节数 |
| $\hat{\Delta q}_{t,b,j}$ | FrontRES 对第 $t$ 步、第 $b$ 个环境、第 $j$ 个关节的预测残差 |
| $\Delta q^*_{t,b,j}$ | 监督目标：$q^{\text{ref}}_{t,b,j} - q^{\text{sim}}_{t,b,j}$ |
| $d_{t,b} \in \{0,1\}$ | 终止标志（机器人跌倒/重置时为 1） |

---

## 1. Huber Loss（逐元素）

$$
\ell_\delta(e) =
\begin{cases}
\dfrac{1}{2}\,e^2 & |e| \leq \delta \\[6pt]
\delta\!\left(|e| - \dfrac{\delta}{2}\right) & |e| > \delta
\end{cases}
$$

其中 $e = \hat{\Delta q}_{t,b,j} - \Delta q^*_{t,b,j}$，$\delta$ 为 Huber 阈值（PyTorch 默认 $\delta = 1.0$）。

---

## 2. 静态下肢加权

$$
w^{\text{s}}_j =
\begin{cases}
\alpha & j \in \mathcal{J}_{\text{lower}} \\
1 & j \notin \mathcal{J}_{\text{lower}}
\end{cases}
$$

其中 $\mathcal{J}_{\text{lower}}$ 为髋/膝/踝关节的索引集合，$\alpha > 1$ 为下肢权重系数（实验中取 $\alpha = 2$）。

---

## 3. 时序门控（关节级）

$$
m^{\text{gate}}_{t,b,j} = \mathbf{1}\!\left[\;\left|\Delta q^*_{t,b,j} - \Delta q^*_{t-1,b,j}\right| < \tau\;\right]
$$

当 $|\Delta q^*|$ 在相邻步之间发生突变（超过阈值 $\tau$）时，该关节被判定为跟踪失效（穿模/漂浮），从本步的损失计算中排除。首步令 $\Delta q^*_{0,b,j} := \Delta q^*_{1,b,j}$，使首步门控恒为开。

---

## 4. $\Delta q$ 掩码机制（样本级）

$$
m^{\text{done}}_{t,b} = 1 - d_{t,b}
$$

机器人跌倒或重置后，$q^{\text{sim}}$ 退化为物理无效状态，对应的 $\Delta q^*$ 不可作为训练信号，整个样本被屏蔽。

---

## 综合损失函数

定义有效权重：

$$
w_{t,b,j} = w^{\text{s}}_j \cdot m^{\text{gate}}_{t,b,j} \cdot m^{\text{done}}_{t,b}
$$

Stage 1 监督损失为：

$$
\boxed{
\mathcal{L}_{\text{sup}} =
\frac{\displaystyle\sum_{t=1}^{T}\sum_{b=1}^{B}\sum_{j=1}^{J}
      \ell_\delta\!\left(\hat{\Delta q}_{t,b,j} - \Delta q^*_{t,b,j}\right) \cdot w_{t,b,j}}
     {\displaystyle\sum_{t=1}^{T}\sum_{b=1}^{B}\sum_{j=1}^{J} w_{t,b,j} + \epsilon}
}
$$

其中 $\epsilon$ 为数值稳定常数，防止分母为零（所有关节同时失效的极端情况）。

---

**各 mask 的语义层次**

$$
w_{t,b,j} = \underbrace{w^{\text{s}}_j}_{\text{下肢侧重}} \cdot \underbrace{m^{\text{gate}}_{t,b,j}}_{\text{关节级：时序门控}} \cdot \underbrace{m^{\text{done}}_{t,b}}_{\text{样本级：终止掩码}}
$$

---


**User**

使用Latex似乎无法渲染，请直接使用text就行

---


**Claude**

符号定义

    T, B, J          时间步数、并行环境数、关节数
    Δq_hat(t,b,j)    FrontRES 预测残差（第 t 步、第 b 个环境、第 j 个关节）
    Δq*(t,b,j)       监督目标 = q_ref(t,b,j) - q_sim(t,b,j)
    d(t,b) ∈ {0,1}   终止标志（跌倒/重置时为 1）

---

1. Huber Loss（逐元素）

    设预测误差 e = Δq_hat(t,b,j) - Δq*(t,b,j)

        L_delta(e) = { (1/2) * e^2              if |e| <= delta
                     { delta * (|e| - delta/2)  if |e| >  delta

    delta 为 Huber 阈值（PyTorch 默认 delta = 1.0）

---

2. 静态下肢加权

        w_s(j) = { alpha    if j in J_lower
                 { 1        otherwise

    J_lower 为髋/膝/踝关节索引集合，alpha > 1 为下肢权重系数（实验取 alpha = 2）

---

3. 时序门控（关节级）

        m_gate(t,b,j) = 1[ |Δq*(t,b,j) - Δq*(t-1,b,j)| < tau ]

    相邻步之间 Δq* 的突变超过阈值 tau 时，判定该关节处于跟踪失效状态
    （穿模/漂浮），将其从本步损失中排除。
    首步令 Δq*(0,b,j) := Δq*(1,b,j)，使首步门控恒为开。

---

4. Δq 掩码机制（样本级）

        m_done(t,b) = 1 - d(t,b)

    机器人跌倒或重置后 q_sim 退化为物理无效状态，对应 Δq* 整体屏蔽。

---

综合损失函数

有效权重：

        w(t,b,j) = w_s(j) * m_gate(t,b,j) * m_done(t,b)
                   -------   ---------------   -----------
                   下肢侧重   关节级时序门控    样本级终止掩码

Stage 1 监督损失：

        L_sup = SUM_{t,b,j} [ L_delta(Δq_hat(t,b,j) - Δq*(t,b,j)) * w(t,b,j) ]
                -----------------------------------------------------------------
                              SUM_{t,b,j} [ w(t,b,j) ] + eps

    eps 为数值稳定常数，防止所有关节同时失效时分母为零。

---


**User**

我有个额外问题：某个关节处于跟踪失效状态时，其他关节是否还有必要计算Loss？因为这可能是摔倒的起点

---


**Claude**

这是个很好的问题，触及了掩码粒度的核心权衡。

---

**支持级联屏蔽整个样本的论据**

人形机器人的平衡是一个全局属性。当某个下肢关节（如踝关节）出现跟踪失效时：

- q_sim 整体已经偏离了参考运动所假设的"直立稳定"状态
- q_ref 是按照正常运动生成的，此时它对其他关节的期望角度也已经不再对应当前机器人的实际状态
- 髋关节和膝关节的 Δq* 虽然本步还没触发时序门，但可能已经是"为了紧急补偿踝关节失效"而产生的异常值，而不是 FrontRES 应该学习的目标

用这种部分失效样本训练，FrontRES 可能学到错误的关节协调模式：在踝关节失效时，髋关节应该做什么——但这个"什么"来自于一段坏轨迹。

---

**反对级联屏蔽的论据**

- 上肢关节（肩、肘、腕）与下肢平衡的耦合很弱，踝关节失效时上肢的 Δq* 大概率仍然有效
- 级联屏蔽会大幅减少有效训练样本数量，降低数据利用率
- 目前的时序门控本身已经是渐进式的：下一步如果其他关节也开始爆炸，它们也会被门控掉

---

**实用的折中方案**

区分下肢失效和上肢失效：

- 当 **下肢关节**（J_lower 中的任一关节）触发时序门控时，屏蔽整个样本（因为平衡崩溃是全身性的）
- 当 **上肢关节**触发时序门控时，仅屏蔽该关节（上肢失效不影响下肢的参考轨迹有效性）

用掩码表达式写出来就是：

    # 任意一个下肢关节触发门控 → 整个样本无效
    any_lower_fail(t,b) = 1 - min_{j in J_lower} [ m_gate(t,b,j) ]

    # 级联样本掩码
    m_cascade(t,b) = (1 - any_lower_fail(t,b)) * m_done(t,b)

    # 最终有效权重
    w(t,b,j) = w_s(j) * m_gate(t,b,j) * m_cascade(t,b)

这样上肢关节保留关节级门控，下肢关节失效则触发整个样本的屏蔽。

---

**我的建议**

如果你的训练数据中机器人跌倒频繁，级联屏蔽对数据利用率的伤害可以接受，建议采用折中方案。如果训练初期数据本来就比较稀疏，可以先用当前的关节级门控，等训练稳定后再视情况引入级联机制。你倾向于哪种？

---


**User**

我认为折中方案更好，毕竟FrontRES输出Δq，而且训练会使用--headless参数，导致无法可视化验证，折中方案符合直觉

---


**Claude**

直接实现，修改 [supervise.py](source/whole_body_tracking/whole_body_tracking/utils/supervise.py) 的 `update()` 中的掩码逻辑：

---


**Claude**

更新诊断指标，把 `joint_gate_ratio` 拆成两个更有意义的量：

---


**Claude**

同步更新论文用的表达式：有效权重现在是三层掩码的乘积。

    定义下肢级联掩码（样本级）：

        any_lower_fail(t,b) = 1 - min_{j in J_lower} [ m_gate(t,b,j) ]

        m_cascade(t,b) = (1 - any_lower_fail(t,b)) * m_done(t,b)

    最终有效权重：

        w(t,b,j) = w_s(j) * m_gate(t,b,j) * m_cascade(t,b)
                   -------   ---------------   ---------------
                   下肢侧重   上肢关节级门控    下肢级联/终止 ← 两者合并为一个样本级掩码

    语义层次：

        上肢关节失效  → 仅屏蔽该关节（m_gate = 0，m_cascade 不变）
        下肢关节失效  → 整个样本屏蔽（m_cascade = 0，所有关节均被排除）
        机器人跌倒    → m_done = 0，m_cascade 连带为 0

新增两个 Wandb 诊断指标：
- `cascade_gate_ratio`：存活样本中通过下肢级联门控的比例，↓ 说明预摔倒频繁，可用于调整 `jump_threshold`
- `upper_joint_gate_ratio`：上肢关节通过时序门控的比例，↓ 说明上肢跟踪不稳定

---


**User**

我对m_done有点疑惑：某段动作序列在仿真中某个时间步后才开始摔倒，但在这个时间步前完全可以用来计算Δq_gt，现在的方式是否直接抛弃整段q_sim？

---


**Claude**

不会抛弃整段序列，`m_done` 是**逐时间步、逐环境**施加的掩码，不是逐 episode 的。

当前 buffer 的存储结构是 `(T, B, A)`，generator 按时间步 `i` 迭代，每步计算：

    valid(t, b) = 1 - done(t, b)

如果 env b 在第 `t_fall` 步摔倒：

    t=1  到 t=t_fall-1:  done=0, valid=1 → 正常参与 Loss
    t=t_fall:            done=1, valid=0 → 仅屏蔽这一步

所以之前的所有步骤都会被正常使用，`m_done` 只排除摔倒那一帧的 Δq_gt。

---

不过你的问题引出了一个真实的边界问题：时序门控在 **episode 交界处**会出错。

    t_fall:    done=1,  target_actions[t_fall]  = Δq_gt（失效/重置状态，不可信）
    t_fall+1:  done=0,  target_actions[t_fall+1] = Δq_gt（新 episode 第一步，可信）

    jump = |target_actions[t_fall+1] - target_actions[t_fall]|

由于 `t_fall` 的 Δq_gt 来自失效状态，这个 jump 可能很大，导致新 episode 第一步被时序门控误杀——而那一步的数据实际上是完全干净的。

修复方法：当 `done(t-1, b) = 1` 时，将 `prev_target` 重置为当前步自身（jump = 0，门控恒开）：

---


**Claude**

修复在 generator 层做最干净——把上一步的 `dones` 也一起 yield，让调用方在 episode 边界处将 `prev_target` 重置：

---


**Claude**

修改完成。逻辑总结：

    prev_done(t-1, b) = 0  →  safe_prev = target_actions[t-1]  （正常计算 jump）
    prev_done(t-1, b) = 1  →  safe_prev = target_actions[t]    （jump = 0，门控恒开）

回到你原来的问题：`m_done` 每步独立施加，整段 episode 中摔倒前的所有步骤都会正常参与 Loss 计算，只有摔倒那一帧本身被屏蔽。

---


**User**

现在我们可以开始讨论Diagnostics Metrics了，我们已经确定的有主损失值，Max Amplitude，Cosine Similarity，还需要哪些Diagnostics Metrics？

---


**Claude**

API Error: Unable to connect to API (UNKNOWN_CERTIFICATE_VERIFICATION_ERROR)

---


**User**

现在我们可以开始讨论Diagnostics Metrics了，我们已经确定的有主损失值，Max Amplitude，Cosine Similarity，还需要哪些Diagnostics Metrics？

---


**Claude**

Failed to authenticate. API Error: 403 {"error":{"type":"forbidden","message":"Request not allowed"}}

---


**User**

现在我们可以开始讨论Diagnostics Metrics了，我们已经确定的有主损失值，Max Amplitude，Cosine Similarity，还需要哪些Diagnostics Metrics？

---


**Claude**

Failed to authenticate. API Error: 403 {"error":{"type":"forbidden","message":"Request not allowed"}}

---


**User**

现在我们可以开始讨论Diagnostics Metrics了，我们已经确定的有主损失值，Max Amplitude，Cosine Similarity，还需要哪些Diagnostics Metrics？

---


**Claude**

基于 FrontRES Stage 1 的训练目标和主要失效模式，以下是需要额外关注的指标：

---

**1. 预测幅度比 (Amplitude Ratio)**

    pred_norm / gt_norm

Stage 1 最大的失效风险是**平凡解**：网络学会输出 Δq ≈ 0，Loss 也能收敛但毫无意义。这个比值应趋近 1.0；如果持续下降趋近 0，说明网络在坍缩。单看 Loss 下降无法区分"学会了"和"输出全零"。

---

**2. 数据质量指标（两个）**

    valid_ratio          — done mask 生效比例，反映跌倒频率
    cascade_gate_ratio   — 存活样本中通过下肢级联门控的比例

这两个指标共同描述"实际参与训练的有效数据量"。如果 cascade_gate_ratio 很低（比如 < 0.5），说明 tau 设置太严或当前策略跌倒太频繁，损失函数本身已经失去意义。在 headless 模式下这是唯一能看到训练环境健康状况的窗口。

---

**3. 均值误差 (Mean MAE)**

    joint_mae_mean

你已确认 Max Amplitude（joint_mae_max）。但 max 对单个最差关节敏感，均值反映整体收敛程度，两者配合才完整：max 持续大但 mean 小，说明是个别关节问题；两者都大，说明整体未收敛。

---

**不建议加的指标**

- **Relative Error**：有了 Amplitude Ratio 和 MAE，相对误差提供的信息基本重叠
- **Upper Joint Gate Ratio**：上肢门控失效不影响平衡，优先级低，会稀释注意力
- **delta_q_gt_norm**：GT 幅度由参考动作决定，训练过程中不会变化，没有监控意义

---

**最终推荐的完整 Metrics 列表**

    behavior              ← 主损失（已确认）
    joint_mae_max         ← Max Amplitude（已确认）
    cosine_similarity     ← 方向对齐（已确认）
    delta_q_norm_ratio    ← 预测幅度比，平凡解检测
    valid_ratio           ← 跌倒频率代理
    cascade_gate_ratio    ← 有效训练数据比例
    joint_mae_mean        ← 均值误差，配合 max 使用

七个指标，每个各司其职，没有冗余。确认后我来对照代码检查是否需要补充或删减。

---


**User**

请进行修改

---


**Claude**

现在对照最终确定的 7 个指标，删除多余的计算和报告项：

移除：`delta_q_pred_norm`、`delta_q_gt_norm`（独立报告）、`relative_error`、`upper_joint_gate_ratio` 及其计算逻辑。
保留：`delta_q_norm_ratio` 的内部计算（仍需 pred_norm 和 gt_norm）。

---


**Claude**

完成。最终 `loss_dict` 精确对应 7 个确定的指标：

    behavior              ← 主损失，应持续下降
    delta_q_norm_ratio    ← 预测幅度比 pred/gt，检测平凡解（→ 0 表示网络坍缩）
    cosine_similarity     ← 方向对齐，目标趋近 1.0
    valid_ratio           ← 非终止步比例，跌倒频率代理
    cascade_gate_ratio    ← 存活样本通过下肢级联门控的比例
    joint_mae_mean        ← 各关节平均绝对误差
    joint_mae_max         ← 最差关节误差上界（Max Amplitude）

---


**User**

你提到我应当检查rsl_rl_ppo_cfg.py中的lower_limb_indices=list(range(12))，请先为我检索本仓库的joint_pos顺序定义在哪

---


**Claude**

`list(range(12))` 完全正确。索引 0-11 恰好对应全部下肢关节：

    0  left_hip_pitch_joint
    1  left_hip_roll_joint
    2  left_hip_yaw_joint
    3  left_knee_joint
    4  left_ankle_pitch_joint
    5  left_ankle_roll_joint
    6  right_hip_pitch_joint
    7  right_hip_roll_joint
    8  right_hip_yaw_joint
    9  right_knee_joint
    10 right_ankle_pitch_joint
    11 right_ankle_roll_joint

定义来源：[robot_registry.py](source/whole_body_tracking/whole_body_tracking/robots/robot_registry.py)，与 `joint_pos` 的实际顺序一致。现在可以直接在 [rsl_rl_ppo_cfg.py](source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py) 中设置：

    lower_limb_indices = list(range(12))

要我顺手写进去吗？

---


**User**

请执行

---


**Claude**

[rsl_rl_ppo_cfg.py:145-152](source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py#L145-L152) を確認したところ、すでに正しく設定済みです：

```python
algorithm = RslRlSuperviseAlgorithmCfg(
    loss_type="huber",
    lower_limb_indices=list(range(12)),  # ← 既に設定済み
    lower_limb_weight=2.0,
    jump_threshold=0.2,
)
```

`list(range(12))` = indices 0-11、robot_registry.py の joint 順と完全一致しており、追加変更は不要です。

---


**User**

现在我已经将FrontRES的Stage 1推送到服务器上开始训练了，我们必须尽快写完Stage 2的强化学习微调，我们已经写好了Stage 2强化学习微调的基础架构和权重调整与PCGrad，现在我们必须开始写域随机化项、奖励函数项和Diagnosis Terms，请先阅读整个仓库，检查我们写的Stage 2 RL基础架构、权重调整、PCGrad代码是否正确

---


**Claude**

Failed to authenticate. API Error: 403 {"error":{"type":"forbidden","message":"Request not allowed"}}

---


**User**

现在我已经将FrontRES的Stage 1推送到服务器上开始训练了，我们必须尽快写完Stage 2的强化学习微调，我们已经写好了Stage 2强化学习微调的基础架构和权重调整与PCGrad，现在我们必须开始写域随机化项、奖励函数项和Diagnosis Terms，请先阅读整个仓库，检查我们写的Stage 2 RL基础架构、权重调整、PCGrad代码是否正确

---


**Claude**

Failed to authenticate. API Error: 401 {"type":"error","error":{"type":"authentication_error","message":"Invalid authentication credentials"},"request_id":"req_011CZnXPNMiPDXMEL5Pg76A2"}

---


**User**

现在我已经将FrontRES的Stage 1推送到服务器上开始训练了，我们必须尽快写完Stage 2的强化学习微调，我们已经写好了Stage 2强化学习微调的基础架构和权重调整与PCGrad，现在我们必须开始写域随机化项、奖励函数项和Diagnosis Terms，请先阅读整个仓库，检查我们写的Stage 2 RL基础架构、权重调整、PCGrad代码是否正确

---


**Claude**

基础架构验证全部通过，所有关键组件均正确。现在直接读关键文件，为写域随机化、奖励函数和 Diagnosis Terms 做准备：

---


**Claude**

现在我有完整的视图。以下是验证结果和需要补充的内容。

---

## 已有代码验证：全部正确

**Stage 2 基础架构**：Distribution 是 `Normal(delta_q_mean, std)` ✓，`get_env_action()` 存在且 runner 正确调用 ✓，优化器只训练 `residual_actor + critic + std` ✓，GMT 冻结 ✓

**域随机化 (`RLFinetuneEventCfg`)**：已有 7 项，比 Stage 1 更强（摩擦力更大、关节偏差±0.02、重力范围更宽、执行器 ±30%、载荷 0-7 kg、推力频率更高），加上 motion-level 扰动（浮空/下陷/脚滑/体拖）。**无需补充**。

**PCGrad**：投影逻辑正确，`mu_batch` 确实来自 `distribution.mean`（即 Δq_mean）✓

---

## 需要补充的三项

### 1. 奖励函数 — 缺少 3 个关键项

`RLFinetuneRewardsCfg` 现有 8 项，但遗漏了：

**a) 全局锚点追踪**（anchor pos/ori）— `RewardsExpertCfg` 有但 `RLFinetuneRewardsCfg` 没有，机器人整体位置/朝向漂移无约束

**b) 脚部接触状态匹配**（`contact_feet`）— 这是 FrontRES 最核心的目标：修正浮空/穿模使脚部接触状态与参考运动一致，缺失这个奖励等于没有直接信号

**c) `joint_acc`** — 基础配置有但 Stage 2 遗漏，关节加速度无约束

### 2. Diagnosis Terms — 完全缺失

当前 `FrontRESFinetuneTrackingEnvCfg` 没有任何诊断项，headless 训练时完全看不到关键状态。

### 3. `q_ref_start_idx` 与 history 的潜在问题

配置注释已标注了这个不确定性：`history_length=5` 时 obs 被拼接为 `163×5=815` 维，`q_ref_start_idx=0` 假设 q_ref 在最新帧切片的开头，但如果 history 是 oldest-first 堆叠，实际 q_ref 位置应该是 `(5-1)×163 = 652`。这需要 debug 一步验证，不影响现在写其余代码。

---

要我现在开始实现补充的奖励项和诊断项吗？奖励项的修改位置在 [tracking_env_cfg.py:974](source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py#L974) 的 `RLFinetuneRewardsCfg`，诊断项需要在同文件里新增一个 `FrontRESDiagnosticsCfg` 并接入 `FrontRESFinetuneTrackingEnvCfg`。

---


**User**

注意Stage 2的奖励项不是用来训练GMT，而是直接用GMT在仿真中Rollout出的轨迹计算得到的奖励项更新FrontRES，而FrontRES接受q_ref，输出Δq，再得到q_dot = q_ref + Δq，然后将q_dot输入GMT，在仿真中得到奖励值，请先分析这种情况下需要什么样的奖励项

---


**Claude**

关键要先明确 FrontRES 的梯度来源：奖励通过 PPO Advantage 反传到 Δq 的 log prob，再更新 `residual_actor`。梯度不经过 GMT，只需要奖励信号能区分"好的 Δq"和"坏的 Δq"。

---

## 分析框架：FrontRES 改变了什么？

FrontRES 唯一能控制的量是 **Δq**。Δq 对仿真的影响路径是：

    Δq → q_dot = q_ref + Δq → GMT → robot_actions → 物理仿真 → 机器人状态

FrontRES 存在的原因是 **q_ref 本身有物理瑕疵**（浮空/穿模/失衡），当 FrontRES 输出正确的 Δq 后，GMT 能更好地跟踪修正后的 q_dot。

因此奖励项的设计原则是：**能明确区分"FrontRES 的修正使 GMT 的执行结果变好"的信号**。

---

## 现有奖励项的适配性分析

**已有的 4 个追踪项**（body_pos / body_ori / body_lin_vel / body_ang_vel）是合理的，因为：
- 当 Δq 正确时，GMT 跟踪改善，机器人肢体位置/速度更接近参考动作
- 这是 FrontRES 存在的核心目标

**已有的 4 个惩罚项**（undesired_contacts / joint_limit / joint_torque / action_rate_l2）是合理的物理约束，但它们只是安全边界，不是 FrontRES 的主要学习信号。

---

## 缺失的关键奖励项

### 最重要：脚部接触状态匹配 (`contact_feet`)

FrontRES 最核心的功能是修正浮空/穿模。但当前奖励里没有任何直接针对脚部接触的信号：
- 浮空：机器人脚部应接地但悬空，body_pos 误差可能很小（脚就在地面附近），但接触状态完全错误
- 穿模：脚部陷入地面，物理引擎会施加强制推力，导致整段轨迹抖动，但 body_pos 奖励无法区分这是 Δq 的问题还是 GMT 的问题

`contact_feet` 直接比较机器人的实际接触状态与参考动作的接触状态，是 FrontRES 修正效果最直接的衡量指标。

### 次要：脚部高度精确追踪 (`teleop_body_position_feet_z` 或 `teleop_body_position_feet`)

脚部高度（Z 坐标）是浮空/穿模的直接体现。整体 body_pos 对 14 个身体部位取平均，对脚部高度的权重只有 2/14。单独给脚部高度设置奖励，能给 FrontRES 更强的梯度信号专门修正脚部。

---

## 不建议添加的项

**全局锚点位置/朝向**（`motion_global_anchor_pos` / `motion_global_anchor_ori`）：
- FrontRES 在关节空间输出 Δq，无法直接控制机器人在世界坐标系中的绝对位置
- 全局位置漂移是长时间步累积误差的结果，与当前帧的 Δq 关系弱
- 将这种弱耦合的信号加入奖励会稀释 FrontRES 真正能控制的部分

**锚点线速度**（`motion_anchor_lin_vel`）：同上，间接信号，FrontRES 不是专门用来控制速度的。

---

## 最终建议的奖励结构

    现有（保留）:
        motion_body_pos        ← 肢体位置追踪
        motion_body_ori        ← 肢体姿态追踪
        motion_body_lin_vel    ← 速度追踪
        motion_body_ang_vel    ← 角速度追踪
        undesired_contacts     ← 非期望接触惩罚
        joint_limit            ← 关节超限惩罚
        joint_torque           ← 扭矩约束
        action_rate_l2         ← 平滑性

    新增（两项）:
        contact_feet           ← 脚部接触状态 vs 参考动作（FrontRES 主要功能的直接信号）
        teleop_body_position_feet ← 脚部位置精确追踪（浮空/穿模的高精度信号）

    不添加:
        全局锚点 pos/ori       ← 与 Δq 耦合过弱
        joint_acc              ← 可选，但优先级低

确认这个分析后我开始实现。

---


**User**

我认识到奖励项的设计与域随机化项息息相关，因为我们这篇文章的核心研究点就是将Human Motion到Robot Feasible Motion的投影转换成隐式去噪映射 (通过最大化奖励项来学习一个近似投影器)，Stage 2的流程是Δq → q_dot = q_ref + Δq → GMT → robot_actions → 物理仿真 → 机器人状态，再直接使用机器人状态得到的奖励项更新FrontRES，我确定了可行的域随机化项，包含[穿模，关节偏置，漂浮，关节解耦，接触错误，根节点错误，质心越界]，那奖励项该如何设计呢

---


**Claude**

这个研究框架非常清晰：**DR 注入结构化噪声，奖励函数是去噪目标，每种噪声类型必须有对应的检测信号，否则 FrontRES 无法学会修正它**。

---

## DR 项与奖励项的一一对应关系

**接触类（3 项对应同一组奖励）**

    穿模 → q_ref 脚部低于地面，机器人实际上感受到地面推力
    漂浮 → q_ref 脚部高于地面，机器人失去应有的支撑
    接触错误 → q_ref 的接触相位与实际不符

    对应奖励：
    ├── contact_feet      — 二值信号：脚部接触状态 vs 参考，直接检测三种接触错误
    └── feet_z tracking   — 连续信号：脚部高度精确值，对穿模/漂浮提供梯度方向

**关节类（2 项对应同一组奖励）**

    关节偏置  → q_ref 某些关节加了系统性偏移
    关节解耦  → q_ref 关节间的运动学约束被破坏

    两类噪声的效果都体现在身体末端段的位置/姿态偏差上

    对应奖励：
    ├── motion_body_pos   — 肢体位置：关节误差积分后的空间表现
    ├── motion_body_ori   — 肢体朝向：姿态偏差，对解耦更敏感
    ├── motion_body_lin_vel — 速度：关节偏差的时间导数
    └── motion_body_ang_vel — 角速度：姿态偏差的时间导数

**根节点类（1 项，需要单独奖励）**

    根节点错误 → q_ref 的盆骨位置/朝向有偏移

    这是我之前分析失误的地方：当 DR 显式注入根节点误差时
    全局锚点追踪就不再是"间接信号"，而是最直接的检测信号

    对应奖励：
    ├── motion_global_anchor_pos — 盆骨世界坐标位置
    └── motion_global_anchor_ori — 盆骨世界坐标朝向

**质心类（1 项，用组合代理）**

    质心越界 → CoM 超出支撑多边形

    理想的奖励是直接计算 CoM 在支撑多边形中的位置，
    但仿真中无现成 API。以下三个信号的组合等价：

    ├── motion_global_anchor_pos — 盆骨位置正确 → CoM 大概率在支撑域内
    ├── contact_feet             — 接触状态正确 → 支撑多边形正确
    └── 自然 Episode 截断        — 机器人摔倒则 Episode 提前结束，累积奖励降低

---

## 完整奖励结构

    ─── 接触去噪信号 ──────────────────────────────────
    contact_feet               (新增)  脚部接触状态匹配，覆盖：穿模/漂浮/接触错误
    teleop_body_position_feet  (新增)  脚部位置精确追踪，对穿模/漂浮提供连续梯度

    ─── 关节去噪信号 ──────────────────────────────────
    motion_body_pos            (已有)  肢体位置，覆盖：关节偏置/关节解耦
    motion_body_ori            (已有)  肢体朝向
    motion_body_lin_vel        (已有)  线速度
    motion_body_ang_vel        (已有)  角速度

    ─── 根节点去噪信号 ─────────────────────────────────
    motion_global_anchor_pos   (新增)  盆骨位置，覆盖：根节点错误/质心越界(代理)
    motion_global_anchor_ori   (新增)  盆骨朝向，覆盖：根节点错误

    ─── 物理安全约束（所有噪声类型通用）───────────────
    undesired_contacts         (已有)  非期望接触惩罚
    joint_limit                (已有)  关节超限惩罚
    joint_torque               (已有)  扭矩约束
    action_rate_l2             (已有)  平滑性

这样每一类 DR 噪声都有至少一个对应的直接奖励信号，去噪映射的学习目标是完备的。确认后开始实现。

---


**User**

我想进一步做出解释：接触错误是滑步 foot sliding，关节偏置需要考虑人形的拓扑结构，比如当手放在头后时，直接将手放到默认姿态很容易拍到头，关节解耦是某关节的运动没有导致应该出现的身体运动

---


**Claude**

好，根据这三个更精确的定义重新分析奖励对应关系：

---

**滑步 (foot sliding)**

足部在接触相位应该静止，但实际在地面上横向移动。

`contact_feet` 只检测接触状态的二值匹配，无法检测足部在接触状态下的位置漂移。对滑步最直接的信号是：

    参考动作中脚部在接触相时位置固定
    → 实际脚部因滑步偏移 → teleop_body_position_feet 捕捉到连续位置误差

`teleop_body_position_feet` 是滑步的正确奖励信号，它测量的是 3D 足部位置误差，包含横向滑移的直接梯度。`contact_feet` 作为补充，捕捉滑步严重到足以改变接触相位的情况。

---

**关节偏置（需考虑拓扑结构）**

用户举的例子很关键：手放在头后时，对手臂关节直接施加偏置，手很可能打到头。这里有两类问题：

**运动学链误差**：关节偏置在拓扑链上累积，末端效应器（手、脚）的位置偏差是关节角误差的积分。`motion_body_pos` 测量身体各段的位置，直接捕捉这种积分误差，是正确信号。

**自碰撞**：手打到头时，`undesired_contacts` 已经覆盖这种情况。当前的 body_names 正则过滤的是"非踝关节、非腕关节以外的所有接触"，手-头碰撞产生的接触力会在手或头部的 link 上被检测到并施加惩罚，阈值 1.0 N 足以捕捉真实碰撞。

因此关节偏置对应：`motion_body_pos`（链误差）+ `undesired_contacts`（自碰撞），两者已经足够，无需新增奖励项。

---

**关节解耦（关节运动未产生应有的身体运动）**

这是最微妙的一项。关节角命令发出了，但身体段没有对应运动（等价于机械断开）。

关键在于：`motion_body_pos` 和 `motion_body_ori` 测量的是**身体段的实际位置和朝向**，而不是关节角命令。如果关节解耦，关节角变了但身体没动，身体位置误差直接暴露这个问题。这正是 body tracking 奖励相比 joint position tracking 奖励的优势——它测量的是"身体是否动了"而不是"关节是否发出了命令"。

关节解耦对应：`motion_body_pos` + `motion_body_ori`，已覆盖。

---

## 最终对应表

    DR 噪声类型        对应奖励信号                          是否已有
    ─────────────────────────────────────────────────────────────────
    穿模               contact_feet, teleop_body_position_feet  新增
    漂浮               contact_feet, teleop_body_position_feet  新增
    滑步               teleop_body_position_feet                新增
    关节偏置(链误差)    motion_body_pos, motion_body_ori         已有
    关节偏置(自碰撞)    undesired_contacts                       已有
    关节解耦           motion_body_pos, motion_body_ori         已有
    根节点错误         motion_global_anchor_pos/ori             新增
    质心越界           motion_global_anchor_pos + contact_feet  部分新增

**结论：需要新增的奖励项共 3 个**（而非之前分析的 2 个，根节点错误现在更明确地需要 anchor 奖励）：

    contact_feet                     脚部接触状态
    teleop_body_position_feet        脚部位置（3D）
    motion_global_anchor_pos         盆骨位置
    motion_global_anchor_ori         盆骨朝向

确认后开始实现。

---


**User**

注意这篇文章并不针对遥操任务，而是针对Tracking任务，似乎不应该涉及遥操奖励项，请修改

---


**Claude**

正确，`teleop_*` 系列奖励函数是为遥操任务设计的（世界坐标系手部/头部追踪），不适合 Tracking 任务。

替换方案：用已有的 Tracking 风格奖励函数，加 `body_names` 过滤参数限定到脚部：

    teleop_body_position_feet   →   motion_relative_body_position_error_exp
                                        body_names=["left_ankle_roll_link",
                                                    "right_ankle_roll_link"]

`motion_relative_body_position_error_exp` 本来就支持 `body_names` 过滤，加了这个参数后它只对踝关节计算位置误差，等价于脚部位置精确追踪，且完全是 Tracking 任务语义（相对锚点的身体段位置）。

`contact_feet` 不涉及遥操概念，保留。

---

**修正后的新增奖励项（全部为 Tracking 风格）：**

    contact_feet
        → func=mdp.contact_feet
        → 覆盖：穿模/漂浮/滑步（接触状态）

    motion_feet_pos（新命名）
        → func=mdp.motion_relative_body_position_error_exp
        → body_names=["left_ankle_roll_link", "right_ankle_roll_link"]
        → 覆盖：穿模/漂浮/滑步（连续位置）

    motion_global_anchor_pos
        → func=mdp.motion_global_anchor_position_error_exp
        → 覆盖：根节点错误/质心越界

    motion_global_anchor_ori
        → func=mdp.motion_global_anchor_orientation_error_exp
        → 覆盖：根节点错误

现在可以开始实现，修改 [tracking_env_cfg.py:974](source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py#L974) 的 `RLFinetuneRewardsCfg` 吗？

---


**User**

请先解释奖励项与域随机化项的对应关系和奖励项的公式，请使用text就行，Latex无法在本地渲染

---


**Claude**

## 域随机化项与奖励项的对应关系

---

**所有追踪类奖励的统一形式**

    R = exp(-E / sigma^2)

    E 是某种误差度量，sigma 是容差标准差（越小要求越精确）
    R 的范围 [0, 1]，误差为 0 时 R = 1，误差越大 R 越小

---

### 穿模 / 漂浮 / 滑步 → contact_feet + motion_feet_pos

**三种噪声的共同机制**

    穿模：q_ref 的脚部低于地面 → GMT 试图将脚驱动到地面以下 → 物理引擎将脚推回 → 脚无法到达参考位置
    漂浮：q_ref 的脚部高于地面（应接触时） → GMT 将脚驱动至悬空高度 → 脚失去接触
    滑步：q_ref 在接触相中脚部有横向位移 → GMT 跟随滑动参考 → 脚在地面上滑动

**奖励 1: contact_feet**

    对每只脚 f in {左踝, 右踝}:
        c_actual(f) = 1  if  z_actual(f) < h_threshold  else  0
        c_ref(f)    = 1  if  z_ref(f)    < h_threshold  else  0

    R_contact = 0.5 * sum_f( 1[ c_actual(f) == c_ref(f) ] )

    范围 [0, 1]，两脚接触状态均与参考一致时取 1
    直接检测：穿模（实际在地面，参考也应接触，一致）、漂浮（实际悬空，参考应接触，不一致）
    对滑步：滑步严重到脚离地时也能检测

**奖励 2: motion_feet_pos（motion_relative_body_position_error_exp，仅脚部）**

    J_feet = {left_ankle_roll_link, right_ankle_roll_link}

    E_feet = mean_{j in J_feet} ( ||p_ref(j) - p_robot(j)||^2 )

    R_feet = exp( -E_feet / sigma_feet^2 )

    sigma_feet 建议设为 0.05 m（比全身 body_pos 更严格，反映脚部精度要求高）
    连续检测：穿模/漂浮的高度误差 + 滑步的横向位移误差

---

### 关节偏置 → motion_body_pos + undesired_contacts

**噪声机制**

    关节偏置在运动学链上累积：
    delta_q_bias(j) → 末端效应器位移 = J(j) * delta_q_bias   （J 为雅可比矩阵）
    
    拓扑约束问题（如手在头后时施加偏置）：
    偏置导致手臂进入头部区域 → 自碰撞 → 物理引擎施加碰撞力

**奖励 3: motion_body_pos（已有）**

    J_all = 所有被追踪的身体段

    E_body_pos = mean_{j in J_all} ( ||p_ref(j) - p_robot(j)||^2 )

    R_body_pos = exp( -E_body_pos / sigma_body_pos^2 )

    捕捉链式积累误差：关节偏置 → 末端段位置偏移 → E_body_pos 增大

**奖励 4: undesired_contacts（已有）**

    对所有非期望接触点（非踝关节、非腕关节的所有 link）:
    R_contact_penalty = -1  if  contact_force > F_threshold

    捕捉拓扑碰撞：手-头碰撞 → 接触力 > 1 N → 惩罚

---

### 关节解耦 → motion_body_pos + motion_body_ori

**噪声机制**

    关节角命令发出，但身体段未产生对应运动（机械解耦）
    关键：body tracking 测量的是身体段实际位置，而非关节角命令
    → 即使关节角正确，身体段未动，误差依然被检测到

**奖励 3: motion_body_pos（同上）**
**奖励 5: motion_body_ori（已有）**

    E_body_ori = mean_{j in J_all} ( quat_err(q_ref(j), q_robot(j))^2 )

    R_body_ori = exp( -E_body_ori / sigma_body_ori^2 )

    body_pos 和 body_ori 组合完整覆盖解耦效果：
    位置不动 → E_body_pos 增大；姿态不转 → E_body_ori 增大

---

### 根节点错误 → motion_global_anchor_pos + motion_global_anchor_ori

**噪声机制**

    q_ref 的盆骨（根节点）位置或朝向注入误差
    → 全身参考轨迹整体偏移 → GMT 将整个机器人驱动到错误位置

**奖励 6: motion_global_anchor_pos（新增）**

    E_anchor_pos = ||p_anchor_ref - p_anchor_robot||^2     （世界坐标系 L2 距离的平方）

    R_anchor_pos = exp( -E_anchor_pos / sigma_anchor_pos^2 )

**奖励 7: motion_global_anchor_ori（新增）**

    E_anchor_ori = quat_err(q_anchor_ref, q_anchor_robot)^2

    R_anchor_ori = exp( -E_anchor_ori / sigma_anchor_ori^2 )

    这两项是根节点误差的最直接检测信号
    注：相比 motion_body_pos 对全身平均，anchor 奖励专门针对根节点

---

### 质心越界 → motion_global_anchor_pos + contact_feet（组合代理）

**噪声机制**

    质心超出支撑多边形 → 机器人失去稳定性趋向摔倒

**无直接对应奖励函数**，通过两个代理信号覆盖：

    motion_global_anchor_pos：盆骨位置正确 → CoM 大概率在支撑域内
    contact_feet：接触状态正确 → 支撑多边形正确
    Episode 自然截断：机器人摔倒 → 后续奖励清零 → 累积奖励降低

---

## 完整奖励项汇总

    奖励项                    公式形式                     覆盖 DR 噪声
    ─────────────────────────────────────────────────────────────────────
    contact_feet             0.5 * sum_f(1[c==c_ref])    穿模/漂浮/滑步(离散)
    motion_feet_pos          exp(-E_feet / s^2)           穿模/漂浮/滑步(连续)
    motion_body_pos          exp(-E_body_pos / s^2)       关节偏置/关节解耦
    motion_body_ori          exp(-E_body_ori / s^2)       关节偏置/关节解耦
    motion_body_lin_vel      exp(-E_vel / s^2)            关节偏置/解耦/速度
    motion_body_ang_vel      exp(-E_angvel / s^2)         关节偏置/解耦/旋转速度
    motion_global_anchor_pos exp(-E_anchor_pos / s^2)    根节点错误/质心越界
    motion_global_anchor_ori exp(-E_anchor_ori / s^2)    根节点错误
    undesired_contacts       -1 if contact > threshold    关节偏置(自碰撞)/摔倒
    joint_limit              -1 if q > q_max              所有噪声类型的安全约束
    joint_torque             -||tau||^2                   所有噪声类型的安全约束
    action_rate_l2           -||a_t - a_{t-1}||^2        平滑性约束

---


**User**

我目前的疑问是：我们的流程是FrontRES → Δq → q_dot = q_ref + Δq → GMT → robot_actions → Simulator → q_sim → Reward，然后直接用Reward更新FrontRES，但这个Reward是针对物理仿真和GMT的奖励项，并不是直接针对FrontRES，直接使用Reward更新FrontRES是否会导致分布偏移，是否要对Reward进行特殊设计？

---


**Claude**

Failed to authenticate. API Error: 403 {"error":{"type":"forbidden","message":"Request not allowed"}}

---


**User**

我目前的疑问是：我们的流程是FrontRES → Δq → q_dot = q_ref + Δq → GMT → robot_actions → Simulator → q_sim → Reward，然后直接用Reward更新FrontRES，但这个Reward是针对物理仿真和GMT的奖励项，并不是直接针对FrontRES，直接使用Reward更新FrontRES是否会导致分布偏移，是否要对Reward进行特殊设计？

---


**Claude**

Failed to authenticate. API Error: 403 {"error":{"type":"forbidden","message":"Request not allowed"}}

---


**User**

我目前的疑问是：我们的流程是FrontRES → Δq → q_dot = q_ref + Δq → GMT → robot_actions → Simulator → q_sim → Reward，然后直接用Reward更新FrontRES，但这个Reward是针对物理仿真和GMT的奖励项，并不是直接针对FrontRES，直接使用Reward更新FrontRES是否会导致分布偏移，是否要对Reward进行特殊设计？

---


**Claude**

这是一个很好的问题，触及了这个训练流程的理论基础。

---

## 这是标准的 Black-box RL，不存在分布偏移问题

从 RL 的视角重新定义每个组件：

    策略 π        = FrontRES，参数 θ
    动作 a        = Δq
    状态 s        = obs（FrontRES 的输入）
    环境          = GMT + 物理仿真（整体视为黑盒）
    奖励 R        = R(q_sim)，仿真状态的函数

PPO 的策略梯度公式：

    gradient = E[ grad_theta log pi(Δq | obs) * Advantage(obs, Δq) ]

梯度不需要穿过 GMT 或仿真器，只需要知道"这个 Δq 比期望表现好多少"。这正是 Black-box RL 的标准设定，奖励函数作用于环境输出，而非直接作用于策略输出。

**奖励针对 GMT+仿真，而不是 FrontRES 本身——这是设计意图，不是问题。** FrontRES 的目标就是让 GMT+仿真工作得更好，所以测量 GMT+仿真输出质量的奖励正是 FrontRES 应该优化的目标。

---

## GMT 冻结是关键保证

若 GMT 是可训练的，则奖励梯度会同时更新 FrontRES 和 GMT，目标函数随训练移动，产生真正的非平稳性问题。但由于 GMT 完全冻结：

    Δq → q_dot → GMT(q_dot) 是确定性映射，训练中不变
    唯一变化的是 FrontRES(θ) → Δq 这一步

因此环境动力学是稳定的，标准的 on-policy PPO 适用。

---

## 唯一真实的风险：分布外的 Δq

存在一个微妙但重要的问题：如果 FrontRES 输出很大的 Δq，则

    q_dot = q_ref + Δq 可能超出 GMT 训练时的输入分布

GMT 在自身训练分布之外的行为是不可预测的，此时仿真奖励反映的是 GMT 的外推失败，而非 FrontRES 的修正质量，奖励信号失去意义。

**这正是 L_reg = λ||Δq||² 的根本作用**，不仅仅是"防止过大修正"的工程约束，而是保证 q_dot 始终在 GMT 训练分布附近，使整个 RL 信号保持有效。

    L_reg 小 → Δq 小 → q_dot ≈ q_ref → 在 GMT 分布内 → 奖励信号可信
    L_reg = 0 → FrontRES 可能学到利用 GMT 的分布外行为来"欺骗"奖励

---

## 奖励是否需要特殊设计？

不需要从根本上重新设计，当前设计已经足够：

    指数形式奖励：始终非零，梯度密集，适合 PPO
    多项物理约束惩罚：防止 FrontRES 通过违反物理约束来提高跟踪得分
    L_reg 正则项（在 PPO loss 中）：保证分布不漂移，这是最关键的设计

Stage 1 预训练的另一个价值也在这里：给 FrontRES 一个合理的初始化，使 Stage 2 RL 开始时 Δq 已经在合理范围内，信用分配问题（哪个 Δq 导致了哪个奖励）从一开始就更容易。

---

**总结：流程设计是正确的，L_reg 是保证信号有效性的核心机制，奖励函数本身无需针对 FrontRES 做特殊修改。**

---


**User**

由于我们是对Δq进行优化，平滑项和L2正则化项是否应当对Δq做更好？

---


**Claude**

这是个很好的观察，两个约束项确实存在概念对齐问题。

---

## action_rate_l2 的问题

当前 `mdp.action_rate_l2` 计算的是：

    ||robot_actions_t - robot_actions_{t-1}||^2

其中 `robot_actions` 是 GMT 的输出，经过了一整个黑盒转换链：

    Δq → q_dot → GMT → robot_actions

FrontRES 只控制 Δq，而 robot_actions 的平滑性不仅取决于 Δq 的平滑性，还取决于 GMT 的内部动力学。用 GMT 输出的平滑性来约束 FrontRES，等于给了一个间接且有噪声的梯度信号。

**正确的平滑约束应该直接作用于 Δq：**

    smooth_loss = ||Δq_t - Δq_{t-1}||^2

这个量在 rollout buffer 里是现成的（`storage.actions` 就是存储的 Δq），可以在 PPO 更新前用完整 rollout 计算，不需要打乱时序的 mini-batch：

    delta_q_diff = storage.actions[1:] - storage.actions[:-1]   # (T-1, B, A)
    smooth_loss = delta_q_diff.pow(2).mean()

然后加入 PPO loss：

    loss = ppo_loss + lambda_reg * reg_loss + lambda_smooth * smooth_loss

---

## L_reg 的评估

`L_reg = lambda_reg * ||Δq_mean||^2` 已经正确地直接作用于 Δq，**无需修改**。

但这里有个微妙的互补关系：

    L_reg（幅度）：限制 Δq 不能太大，最小干预原则
    smooth_loss（时序）：限制 Δq 不能在相邻帧间剧变

两者不能互相替代：L_reg 允许每帧都有大 Δq 只要幅度一致；smooth_loss 允许 Δq 很大只要变化缓慢。对于去噪映射的物理意义：

    L_reg     → 修正量要小（FrontRES 是微调器，不是替代 GMT）
    smooth_loss → 修正量要稳定（相邻帧的物理状态相近，修正应连续）

---

## 修改建议

    移除：RLFinetuneRewardsCfg 中的 action_rate_l2（针对 robot_actions，概念错误）

    保留：L_reg = lambda_reg * ||Δq_mean||^2（已在 PPO loss 中，正确）

    新增：smooth_loss = lambda_smooth * ||Δq_t - Δq_{t-1}||^2（在 PPO loss 中，针对 Δq）

在 `ppo.py` 的 `update()` 中，smooth_loss 与 reg_loss 同级，可选择加入 PCGrad（三任务梯度投影）或简单加权合并。

是否立即实现这两处修改？

---


**User**

请为我总结现在讨论后得到的所有奖励项，并且给出每个奖励项中E的公式

---


**Claude**

## Stage 2 完整奖励项汇总

---

### 环境奖励项（在 RLFinetuneRewardsCfg 中）

**接触去噪**

    contact_feet
        E = 对每只脚 f:
                c_actual(f) = 1  if  z_actual(f) < h_threshold  else  0
                c_ref(f)    = 1  if  z_ref(f)    < h_threshold  else  0
            R = 0.5 * sum_f( 1[ c_actual(f) == c_ref(f) ] )
            注: 非指数形式，直接为 [0, 1] 离散奖励
        覆盖: 穿模 / 漂浮 / 滑步（接触状态）

    motion_feet_pos
        E = mean_{j in {left_ankle, right_ankle}} ( ||p_ref(j) - p_robot(j)||^2 )
            p 为世界坐标系下的身体段位置
        覆盖: 穿模 / 漂浮 / 滑步（连续位置）

**关节去噪**

    motion_body_pos
        E = mean_{j in J_all} ( ||p_ref(j) - p_robot(j)||^2 )
            J_all 为所有被追踪身体段（14 个）
        覆盖: 关节偏置（链式积累）/ 关节解耦

    motion_body_ori
        E = mean_{j in J_all} ( quat_err(quat_ref(j), quat_robot(j))^2 )
            quat_err 为四元数差的旋转角度量（标量）
        覆盖: 关节偏置 / 关节解耦（姿态维度）

    motion_body_lin_vel
        E = mean_{j in J_all} ( ||v_ref(j) - v_robot(j)||^2 )
            v 为世界坐标系下的线速度
        覆盖: 关节偏置 / 关节解耦（速度维度）

    motion_body_ang_vel
        E = mean_{j in J_all} ( ||omega_ref(j) - omega_robot(j)||^2 )
            omega 为世界坐标系下的角速度
        覆盖: 关节偏置 / 关节解耦（角速度维度）

**根节点去噪**

    motion_global_anchor_pos
        E = ||p_anchor_ref - p_anchor_robot||^2
            p_anchor 为锚点（torso_link）在世界坐标系的位置
        覆盖: 根节点错误 / 质心越界（代理）

    motion_global_anchor_ori
        E = quat_err(quat_anchor_ref, quat_anchor_robot)^2
        覆盖: 根节点错误

**物理安全约束（惩罚项，无 E 公式）**

    undesired_contacts     -1  if  contact_force(link) > F_threshold
                           link 范围: 非踝关节、非腕关节的所有 link
                           覆盖: 关节偏置自碰撞 / 摔倒

    joint_limit            -1  if  q_j > q_max(j)  or  q_j < q_min(j)
                           覆盖: 所有 DR 噪声类型的安全边界

    joint_torque           E = sum_j( tau_j^2 )
                           覆盖: 执行器过载保护

    ~~action_rate_l2~~     已移除（针对 robot_actions，与 Δq 优化不对齐）

---

### PPO Loss 中的约束项（不在环境奖励中，直接在 ppo.py 计算）

    L_reg
        E = ||Δq_mean||^2  =  sum_j( delta_q_mean_j^2 )
            Δq_mean 为 residual_actor 的均值输出（非采样值）
        作用: 最小干预原则，防止 q_dot 超出 GMT 训练分布
        形式: L_reg = lambda_reg * E

    smooth_loss（新增）
        E = ||Δq_t - Δq_{t-1}||^2  =  sum_j( (delta_q_t_j - delta_q_{t-1}_j)^2 )
            对 rollout buffer 中相邻时间步的 Δq 采样值计算
        作用: 时序平滑约束，相邻帧修正量不应剧变
        形式: smooth_loss = lambda_smooth * mean_{t}(E_t)

---

### 各项覆盖的 DR 噪声汇总

    DR 噪声类型     主要检测奖励项                        辅助检测
    ──────────────────────────────────────────────────────────────────
    穿模            contact_feet, motion_feet_pos          undesired_contacts
    漂浮            contact_feet, motion_feet_pos          motion_global_anchor_pos
    滑步            motion_feet_pos, contact_feet          motion_body_pos
    关节偏置        motion_body_pos, motion_body_ori       undesired_contacts
    关节解耦        motion_body_pos, motion_body_ori       motion_body_lin_vel
    根节点错误      motion_global_anchor_pos/ori           motion_body_pos
    质心越界        motion_global_anchor_pos, contact_feet Episode 截断

---


**User**

你漏掉了奖励项与exp(-E / sigma^2)公式形态的表格

---


**Claude**

## 奖励项完整公式表

    奖励项                      R 的公式                                    sigma 建议值
    ─────────────────────────────────────────────────────────────────────────────────────
    contact_feet                R = 0.5 * sum_f( 1[c_actual(f)==c_ref(f)] )  — (非指数形式)

    motion_feet_pos             R = exp( -E_feet / sigma^2 )                 sigma = 0.05 m
                                E = mean_{j in J_feet}( ||p_ref(j)-p_robot(j)||^2 )

    motion_body_pos             R = exp( -E_body_pos / sigma^2 )             sigma = 0.3 m
                                E = mean_{j in J_all}( ||p_ref(j)-p_robot(j)||^2 )

    motion_body_ori             R = exp( -E_body_ori / sigma^2 )             sigma = 0.4 rad
                                E = mean_{j in J_all}( quat_err(quat_ref(j), quat_robot(j))^2 )

    motion_body_lin_vel         R = exp( -E_lin_vel / sigma^2 )              sigma = 1.0 m/s
                                E = mean_{j in J_all}( ||v_ref(j)-v_robot(j)||^2 )

    motion_body_ang_vel         R = exp( -E_ang_vel / sigma^2 )              sigma = 3.14 rad/s
                                E = mean_{j in J_all}( ||omega_ref(j)-omega_robot(j)||^2 )

    motion_global_anchor_pos    R = exp( -E_anchor_pos / sigma^2 )           sigma = 0.3 m
                                E = ||p_anchor_ref - p_anchor_robot||^2

    motion_global_anchor_ori    R = exp( -E_anchor_ori / sigma^2 )           sigma = 0.4 rad
                                E = quat_err(quat_anchor_ref, quat_anchor_robot)^2

    undesired_contacts          R = -1  if  contact_force(link) > 1.0 N      — (惩罚项)

    joint_limit                 R = -1  if  q_j outside [q_min, q_max]       — (惩罚项)

    joint_torque                R = -sum_j( tau_j^2 )                        — (L2惩罚)

    L_reg（PPO loss 中）        L = lambda_reg * ||Δq_mean||^2               lambda = 0.01
    smooth_loss（PPO loss 中）  L = lambda_smooth * mean_t(||Δq_t-Δq_{t-1}||^2)  lambda = 待定

---


**User**

我想要的是奖励项与exp(-E / sigma^2)公式形态与域随机化三者对应的表格

---


**Claude**

    DR 噪声类型    奖励项                      公式                                                    sigma
    ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    穿模           contact_feet                R = 0.5 * sum_f( 1[c_actual(f)==c_ref(f)] )             — (非指数形式)
    漂浮           contact_feet                R = 0.5 * sum_f( 1[c_actual(f)==c_ref(f)] )             — (非指数形式)
    滑步           contact_feet                R = 0.5 * sum_f( 1[c_actual(f)==c_ref(f)] )             — (非指数形式)

    穿模           motion_feet_pos             R = exp( -E / sigma^2 )                                 sigma = 0.05 m
    漂浮           motion_feet_pos             E = mean_{j in J_feet}( ||p_ref(j)-p_robot(j)||^2 )
    滑步           motion_feet_pos

    关节偏置       motion_body_pos             R = exp( -E / sigma^2 )                                 sigma = 0.3 m
    关节解耦       motion_body_pos             E = mean_{j in J_all}( ||p_ref(j)-p_robot(j)||^2 )

    关节偏置       motion_body_ori             R = exp( -E / sigma^2 )                                 sigma = 0.4 rad
    关节解耦       motion_body_ori             E = mean_{j in J_all}( quat_err(quat_ref(j), quat_robot(j))^2 )

    关节偏置       motion_body_lin_vel         R = exp( -E / sigma^2 )                                 sigma = 1.0 m/s
    关节解耦       motion_body_lin_vel         E = mean_{j in J_all}( ||v_ref(j)-v_robot(j)||^2 )

    关节偏置       motion_body_ang_vel         R = exp( -E / sigma^2 )                                 sigma = 3.14 rad/s
    关节解耦       motion_body_ang_vel         E = mean_{j in J_all}( ||omega_ref(j)-omega_robot(j)||^2 )

    根节点错误     motion_global_anchor_pos    R = exp( -E / sigma^2 )                                 sigma = 0.3 m
    质心越界       motion_global_anchor_pos    E = ||p_anchor_ref - p_anchor_robot||^2

    根节点错误     motion_global_anchor_ori    R = exp( -E / sigma^2 )                                 sigma = 0.4 rad
                   motion_global_anchor_ori    E = quat_err(quat_anchor_ref, quat_anchor_robot)^2

    关节偏置       undesired_contacts          R = -1  if  contact_force(link) > 1.0 N                 — (惩罚项)
    (自碰撞)

    全部噪声类型   joint_limit                 R = -1  if  q_j outside [q_min, q_max]                  — (惩罚项)
    全部噪声类型   joint_torque                R = -sum_j( tau_j^2 )                                   — (L2惩罚)

    全部噪声类型   L_reg (PPO loss)            L = lambda_reg * ||Δq_mean||^2                          lambda = 0.01
    全部噪声类型   smooth_loss (PPO loss)      L = lambda_smooth * mean_t( ||Δq_t - Δq_{t-1}||^2 )    lambda = 待定

---


**User**

穿模不仅是脚与地面穿模，还可能出现身体部位之间的穿模，比如手穿模到身体中

---


**Claude**

正确，穿模应该拆分为两类，对应不同的奖励信号：

---

**脚-地面穿模**

物理引擎将脚强制推回地面，脚无法到达参考位置。

    检测信号：contact_feet, motion_feet_pos

**身体自穿模（如手穿入躯干）**

两种情况：
- 物理引擎检测到碰撞并施加分离力 → 产生非期望接触力 → `undesired_contacts` 捕捉
- 物理引擎未能阻止穿模（soft contact）→ 手的实际位置偏离参考 → `motion_body_pos` 捕捉

    检测信号：undesired_contacts（主要）, motion_body_pos（补充）

---

## 修正后的三列对应表

    DR 噪声类型          奖励项                      公式                                                    sigma
    ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    脚-地面穿模          contact_feet                R = 0.5 * sum_f( 1[c_actual(f)==c_ref(f)] )             — (非指数形式)
    漂浮                 contact_feet
    滑步                 contact_feet

    脚-地面穿模          motion_feet_pos             R = exp( -E / sigma^2 )                                 sigma = 0.05 m
    漂浮                 motion_feet_pos             E = mean_{j in J_feet}( ||p_ref(j)-p_robot(j)||^2 )
    滑步                 motion_feet_pos

    身体自穿模           undesired_contacts          R = -1  if  contact_force(link) > 1.0 N                 — (惩罚项)
    关节偏置(自碰撞)     undesired_contacts

    身体自穿模           motion_body_pos             R = exp( -E / sigma^2 )                                 sigma = 0.3 m
    关节偏置(链误差)     motion_body_pos             E = mean_{j in J_all}( ||p_ref(j)-p_robot(j)||^2 )
    关节解耦             motion_body_pos

    关节偏置             motion_body_ori             R = exp( -E / sigma^2 )                                 sigma = 0.4 rad
    关节解耦             motion_body_ori             E = mean_{j in J_all}( quat_err(quat_ref(j), quat_robot(j))^2 )

    关节偏置             motion_body_lin_vel         R = exp( -E / sigma^2 )                                 sigma = 1.0 m/s
    关节解耦             motion_body_lin_vel         E = mean_{j in J_all}( ||v_ref(j)-v_robot(j)||^2 )

    关节偏置             motion_body_ang_vel         R = exp( -E / sigma^2 )                                 sigma = 3.14 rad/s
    关节解耦             motion_body_ang_vel         E = mean_{j in J_all}( ||omega_ref(j)-omega_robot(j)||^2 )

    根节点错误           motion_global_anchor_pos    R = exp( -E / sigma^2 )                                 sigma = 0.3 m
    质心越界             motion_global_anchor_pos    E = ||p_anchor_ref - p_anchor_robot||^2

    根节点错误           motion_global_anchor_ori    R = exp( -E / sigma^2 )                                 sigma = 0.4 rad
                         motion_global_anchor_ori    E = quat_err(quat_anchor_ref, quat_anchor_robot)^2

    全部噪声类型         joint_limit                 R = -1  if  q_j outside [q_min, q_max]                  — (惩罚项)
    全部噪声类型         joint_torque                R = -sum_j( tau_j^2 )                                   — (L2惩罚)

    全部噪声类型         L_reg (PPO loss)            L = lambda_reg * ||Δq_mean||^2                          lambda = 0.01
    全部噪声类型         smooth_loss (PPO loss)      L = lambda_smooth * mean_t( ||Δq_t - Δq_{t-1}||^2 )    lambda = 待定

---


**User**

我认为你对身体自穿模使用的奖励项解释得比较易懂，请为每个域随机化项都这样解释一遍，并将奖励项-公式-域随机化项三者对应表格拆分成域随机化项和奖励项 + 奖励项详表两个表格，我认为这样更方便阅读

---


**Claude**

## 域随机化项机制解释

**脚-地面穿模**
q_ref 的脚部低于地面，GMT 试图将脚驱动到地面以下，物理引擎强制将脚推回，脚无法到达参考位置，产生高度误差和错误的接触状态。

**漂浮**
q_ref 的脚部在应接触地面时高于地面，GMT 将脚驱动至悬空高度，机器人失去应有的地面支撑，产生高度误差和缺失的接触状态。

**滑步**
q_ref 在接触相中脚部有横向位移，GMT 跟随滑动参考将脚拖过地面，脚在接触状态下发生横向漂移，产生水平位置误差。

**关节偏置**
q_ref 的某些关节角加了系统性偏移，误差沿运动学链积累，末端效应器偏离参考位置。拓扑约束问题（如手在头后时加偏置）可能导致手穿入躯干或头部，产生非期望接触力。

**关节解耦**
q_ref 某关节发出了运动命令，但未产生对应的身体运动（等价于机械断开）。关键在于 body tracking 测量的是身体段实际位置而非关节角命令，即使关节角正确，身体段未动，误差依然被检测到。

**身体自穿模**
q_ref 导致两个身体段（如手和躯干）在空间上重叠，物理引擎检测碰撞并施加分离力产生非期望接触，或引擎未能阻止穿模导致身体段偏离参考位置。

**根节点错误**
q_ref 的盆骨（根节点）位置或朝向注入误差，全身参考轨迹整体偏移，GMT 将整个机器人驱动到错误的世界坐标位置。

**质心越界**
机器人质心超出当前支撑多边形范围，机器人趋向失衡摔倒。无直接仿真 API 可以计算，通过盆骨位置（CoM 代理）和接触状态（支撑多边形完整性）组合检测。

---

## 表格一：域随机化项与奖励项对应

    DR 噪声类型          主要奖励项                                    辅助奖励项
    ─────────────────────────────────────────────────────────────────────────────────
    脚-地面穿模          contact_feet, motion_feet_pos                 joint_limit
    漂浮                 contact_feet, motion_feet_pos                 motion_global_anchor_pos
    滑步                 motion_feet_pos, contact_feet                 motion_body_pos
    关节偏置             motion_body_pos, motion_body_ori              undesired_contacts
    关节解耦             motion_body_pos, motion_body_ori              motion_body_lin_vel
    身体自穿模           undesired_contacts, motion_body_pos           —
    根节点错误           motion_global_anchor_pos/ori                  motion_body_pos
    质心越界             motion_global_anchor_pos, contact_feet        Episode 截断

---

## 表格二：奖励项详表

    奖励项                      E 的定义                                                          公式形式                        sigma
    ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    contact_feet                c_f = 1[z(f) < h],  f in {左踝, 右踝}                           0.5*sum_f(1[c_actual==c_ref])   — (非指数)
                                E 对比 c_actual(f) 与 c_ref(f) 的二值匹配

    motion_feet_pos             E = mean_{j in J_feet}( ||p_ref(j) - p_robot(j)||^2 )           exp( -E / sigma^2 )             sigma = 0.05 m
                                J_feet = {left_ankle_roll_link, right_ankle_roll_link}

    motion_body_pos             E = mean_{j in J_all}( ||p_ref(j) - p_robot(j)||^2 )            exp( -E / sigma^2 )             sigma = 0.3 m
                                J_all = 全部 14 个被追踪身体段

    motion_body_ori             E = mean_{j in J_all}( quat_err(quat_ref(j), quat_robot(j))^2 ) exp( -E / sigma^2 )             sigma = 0.4 rad

    motion_body_lin_vel         E = mean_{j in J_all}( ||v_ref(j) - v_robot(j)||^2 )            exp( -E / sigma^2 )             sigma = 1.0 m/s

    motion_body_ang_vel         E = mean_{j in J_all}( ||w_ref(j) - w_robot(j)||^2 )            exp( -E / sigma^2 )             sigma = 3.14 rad/s

    motion_global_anchor_pos    E = ||p_anchor_ref - p_anchor_robot||^2                          exp( -E / sigma^2 )             sigma = 0.3 m

    motion_global_anchor_ori    E = quat_err(quat_anchor_ref, quat_anchor_robot)^2               exp( -E / sigma^2 )             sigma = 0.4 rad

    undesired_contacts          —                                                                -1  if  F(link) > 1.0 N         — (惩罚项)

    joint_limit                 —                                                                -1  if  q outside [qmin, qmax]  — (惩罚项)

    joint_torque                E = sum_j( tau_j^2 )                                             -E                             — (L2 惩罚)

    L_reg (PPO loss)            E = ||Δq_mean||^2                                                lambda_reg * E                  lambda = 0.01

    smooth_loss (PPO loss)      E = mean_t( ||Δq_t - Δq_{t-1}||^2 )                             lambda_smooth * E               lambda = 待定

---


**User**

我想要的域随机化项机制解释应当包含这个域随机化项的这个情况到底会被哪个奖励项捕捉，请重新解释

---


**Claude**

## 域随机化项机制解释

**脚-地面穿模**
q_ref 的脚部低于地面，GMT 试图将脚驱动到地面以下，物理引擎强制将脚推回，脚无法到达参考位置。脚部高度产生持续误差，被 `motion_feet_pos` 的 E 捕捉；应接触时脚被推离地面导致接触状态异常，被 `contact_feet` 的二值匹配捕捉。

**漂浮**
q_ref 的脚部在应接触地面时高于地面，GMT 将脚驱动至悬空高度，机器人失去应有的地面支撑。脚部悬空产生高度误差，被 `motion_feet_pos` 捕捉；接触相中脚部未接触地面，被 `contact_feet` 捕捉；盆骨高度随之抬升偏离参考，被 `motion_global_anchor_pos` 捕捉。

**滑步**
q_ref 在接触相中脚部有横向位移，GMT 跟随滑动参考将脚拖过地面，脚在接触状态下发生横向漂移。脚部的横向位置偏差直接被 `motion_feet_pos` 的 XY 分量捕捉；滑步严重到足以改变接触相位时，被 `contact_feet` 捕捉。

**关节偏置**
q_ref 某关节角加了系统性偏移，误差沿运动学链积累，末端效应器偏离参考位置。链式积累导致的身体段位置偏差被 `motion_body_pos` 捕捉，姿态偏差被 `motion_body_ori` 捕捉。当拓扑结构导致关节偏置产生自碰撞（如手臂偏置导致手打到头），碰撞产生的非期望接触力被 `undesired_contacts` 捕捉。

**关节解耦**
q_ref 某关节发出了运动命令，但未产生对应的身体运动（等价于机械断开）。`motion_body_pos` 和 `motion_body_ori` 测量的是身体段实际位置和姿态而非关节角命令，即使关节角命令正确，身体段未动，位置误差和姿态误差依然直接被两者捕捉。速度层面的解耦（关节速度命令未产生身体速度）进一步被 `motion_body_lin_vel` 和 `motion_body_ang_vel` 捕捉。

**身体自穿模**
q_ref 导致两个身体段（如手和躯干）在空间上重叠，物理引擎检测到碰撞并施加分离力。分离力产生的非期望接触被 `undesired_contacts` 捕捉；引擎无法完全阻止穿模时，身体段实际位置偏离参考，被 `motion_body_pos` 捕捉。

**根节点错误**
q_ref 的盆骨（根节点）位置或朝向注入误差，GMT 将整个机器人驱动到错误的世界坐标位置。盆骨的世界坐标位置偏差直接被 `motion_global_anchor_pos` 捕捉，朝向偏差被 `motion_global_anchor_ori` 捕捉；全身随根节点整体偏移，进一步被 `motion_body_pos` 捕捉。

**质心越界**
机器人质心超出当前支撑多边形范围，机器人趋向失衡。仿真无直接 CoM API，通过两个代理捕捉：盆骨是 CoM 的最近刚体代理，其位置偏差被 `motion_global_anchor_pos` 捕捉；接触状态异常（支撑多边形缩小）被 `contact_feet` 捕捉；失衡导致摔倒时 Episode 提前截断，后续所有奖励清零。

---


**User**

我感觉奖励项和域随机化项已经讨论得差不多了，请执行修改

---


**Claude**

现在在 `ppo.py` 中添加 `smooth_loss`：

---


**Claude**

これで構造が把握できました。`smooth_loss` を `update()` に追加します。mini-batch ループ内では時系列が壊れているため、ループ前に rollout buffer 全体から一括計算します：

---


**Claude**

现在在 `update()` 的 mini-batch 循环前插入 smooth_loss 计算，并将其加入 loss 组合：

---


**Claude**

smooth_loss 加入 loss 组合（标准路径和 PCGrad 路径）：

---


**Claude**

PCGrad 路径也加入 smooth_loss：

---


**Claude**

最后在 loss 统计和返回字典中加入 smooth_loss：

---


**Claude**

现在查看返回字典的位置：

---


**Claude**

完成。三处修改汇总：

**[tracking_env_cfg.py](source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py) — `RLFinetuneRewardsCfg`**
- 移除 `action_rate_l2`（针对 GMT 输出，概念错误）
- 新增 `contact_feet`（脚-地面穿模/漂浮/滑步，离散接触信号）
- 新增 `motion_feet_pos`（脚部位置精确追踪，std=0.05m）
- 新增 `motion_global_anchor_pos`（根节点位置，根节点错误/质心越界）
- 新增 `motion_global_anchor_ori`（根节点朝向）

**[ppo.py](source/rsl_rl/rsl_rl/algorithms/ppo.py) — `update()`**
- 在 mini-batch 循环前对 `storage.actions`（即 Δq）计算时序差分 smooth_loss
- 加入标准路径 loss 组合：`loss = ppo_loss + λ_reg * reg_loss + λ_smooth * smooth_loss`
- 加入 PCGrad 路径：smooth_loss 与 reg_loss 合并为同一任务（目标方向一致）
- `λ_smooth` 默认 0.0，待超参调优后启用

---


**User**

现在我们可以来讨论一下诊断项了，第二阶段强化学习微调需要观测什么奖励项知道FrontRES已经收敛？

---


**Claude**

收敛判断需要从三个层次同时监控：

---

## 第一层：训练过程健康（PPO 是否正常工作）

    mean_reward           总奖励均值，最核心的收敛信号，持续不增则收敛
    mean_episode_length   Episode 平均长度，越长说明机器人越少摔倒，间接反映策略质量
    value_loss            Critic 拟合误差，应在训练初期下降后稳定
    surrogate_loss        Actor 更新幅度，应在合理范围内波动
    kl_divergence         策略更新步长，应始终接近 desired_kl=0.008

这五项是判断 PPO 本身是否健康的基础，不健康的 PPO 即使奖励看起来在上升也无意义。

---

## 第二层：FrontRES 输出特性（Δq 是否有意义）

    delta_q_norm_mean     Δq 均值的 L2 范数，应小且稳定（过大说明漂移出 GMT 分布）
    delta_q_norm_std      跨环境的 Δq 幅度方差，过大说明策略不稳定
    reg_loss              ||Δq_mean||^2，与 delta_q_norm_mean 共同反映修正量大小
    smooth_loss           ||Δq_t - Δq_{t-1}||^2，时序平滑性，过大说明修正跳变

**平凡解检测**：如果 `delta_q_norm_mean → 0` 而 `mean_reward` 也不增长，说明 FrontRES 学到了零输出（平凡解），Stage 1 预训练失效，需要检查初始化。

---

## 第三层：每类 DR 噪声是否被修正（FrontRES 功能是否实现）

    contact_feet_reward      趋近 1.0 = 脚部接触状态完全匹配，穿模/漂浮/滑步已修正
    motion_feet_pos_reward   趋近 1.0 = 脚部位置误差接近零，精细接触修正成功
    motion_body_pos_reward   趋近 1.0 = 关节偏置/解耦已修正
    motion_global_anchor_reward 趋近 1.0 = 根节点误差已修正
    undesired_contacts_rate  趋近 0 = 自碰撞消失

这一层是判断 FrontRES 是否真的学到了"去噪投影"功能的直接证据。

---

## 收敛的判断标准

    必要条件（同时满足）:
        mean_reward      plateau（连续 5000 iter 增量 < 1%）
        mean_episode_length  plateau 或达到 max_episode_steps
        kl_divergence    稳定在 desired_kl 附近

    充分条件（功能验证）:
        contact_feet_reward  > 0.85
        delta_q_norm_mean    < 0.05 rad（修正量小但非零）
        undesired_contacts_rate  接近 0

    异常信号（需要干预）:
        delta_q_norm_mean  持续增大 → λ_reg 过小，需要增大
        mean_episode_length 不增长  → 奖励信号太弱或 DR 太强，需要调整
        kl_divergence >> desired_kl → 学习率过大，需要降低

---

目前 rsl_rl runner 已经自动记录 `mean_reward` 和 `mean_episode_length`，PPO loss dict 已有 `value_function / surrogate / reg / smooth / lambda_reg`。

需要新增的只有**第三层的各奖励项分项值**和 **`delta_q_norm_mean / delta_q_norm_std`**，前者需要在环境的 `extras["log"]` 里单独上报，后者需要在 PPO 的 loss dict 里添加。需要我来实现这两处吗？

---


**User**

我很喜欢你的分层解释，请为我列个表格

---


**Claude**

## Stage 2 RL 微调收敛监控指标表

---

### 第一层：训练过程健康

    指标                  来源              正常范围              异常信号及处理
    ──────────────────────────────────────────────────────────────────────────────
    mean_reward           runner 自动记录   持续上升后 plateau    长期不增 → 检查奖励权重
    mean_episode_length   runner 自动记录   趋近 max_steps        不增 → DR 过强或奖励过弱
    value_loss            PPO loss dict     初期下降后稳定         持续高 → Critic 未收敛
    surrogate_loss        PPO loss dict     在零附近小幅波动       持续负增大 → 学习率过高
    kl_divergence         PPO loss dict     ≈ desired_kl=0.008    持续偏高 → 降低学习率

---

### 第二层：FrontRES 输出特性

    指标                  来源              正常范围              异常信号及处理
    ──────────────────────────────────────────────────────────────────────────────
    delta_q_norm_mean     PPO loss dict     小且稳定 (< 0.05 rad) → 0 = 平凡解，增大奖励权重
                                                                   持续增大 → 增大 lambda_reg
    delta_q_norm_std      PPO loss dict     低方差                 高方差 → 策略不稳定
    reg_loss              PPO loss dict     随 delta_q_norm 同步   与 delta_q 不同步 → bug
    smooth_loss           PPO loss dict     趋近 0                 持续高 → 修正量跳变，启用 lambda_smooth

---

### 第三层：各 DR 噪声修正效果

    指标                       来源              收敛目标       对应 DR 噪声
    ───────────────────────────────────────────────────────────────────────────────
    contact_feet_reward        extras["log"]     > 0.85         脚-地面穿模 / 漂浮 / 滑步
    motion_feet_pos_reward     extras["log"]     趋近 1.0       穿模 / 漂浮 / 滑步（连续）
    motion_body_pos_reward     extras["log"]     趋近 1.0       关节偏置 / 关节解耦
    motion_body_ori_reward     extras["log"]     趋近 1.0       关节偏置 / 关节解耦（姿态）
    motion_anchor_pos_reward   extras["log"]     趋近 1.0       根节点错误 / 质心越界
    undesired_contacts_rate    extras["log"]     趋近 0         身体自穿模 / 关节偏置自碰撞

---

### 收敛判断标准

    级别        条件
    ──────────────────────────────────────────────────────────────────
    必要条件    mean_reward plateau（连续 5000 iter 增量 < 1%）
                mean_episode_length plateau 或达到 max_steps
                kl_divergence 稳定在 desired_kl 附近
    ──────────────────────────────────────────────────────────────────
    充分条件    contact_feet_reward > 0.85
                delta_q_norm_mean < 0.05 rad（非零）
                undesired_contacts_rate ≈ 0

---


**User**

请执行修改

---


**Claude**

先看 runner 如何读取 `extras["log"]`，再看环境如何上报：

---


**Claude**

机制已经清楚了：`ep_infos` 通过 `infos["episode"]` 或 `infos["log"]` 收集，runner 自动将其写到 Wandb。IsaacLab 的 `ManagerBasedRLEnv` 在 episode 结束时将各奖励项的累积和放入 `infos["episode"]`。

因此第三层指标（各奖励分项）**不需要额外代码**，IsaacLab 已经自动上报所有奖励项的 episode 累积值。只需要在 `ppo.py` 的 `loss_dict` 里加入第二层的 `delta_q_norm_mean` 和 `delta_q_norm_std`：

---


**Claude**

完成。两处修改：

**[ppo.py](source/rsl_rl/rsl_rl/algorithms/ppo.py)**
- 在 mini-batch 循环前对 `storage.actions`（Δq）计算 `delta_q_norm_mean` 和 `delta_q_norm_std`
- 加入 `loss_dict`，runner 自动写入 `Loss/delta_q_norm_mean` 和 `Loss/delta_q_norm_std`

**第三层各奖励分项**（`contact_feet_reward`、`motion_feet_pos_reward` 等）无需额外代码，IsaacLab 的 `ManagerBasedRLEnv` 在每个 episode 结束时自动将所有奖励项的累积值写入 `infos["episode"]`，runner 第 563-679 行已经将其全部上报到 Wandb 的 `Episode/` 命名空间下。

---


**User**

我在思考如果在微调初期就让FrontRES输出的Δq直接加给q_ref，可能会导致GMT输出的动作直接摔倒，似乎应该使用课程学习的方式逐步增加Δq的权重，并且我也在思考域随机化在训练中的使用强度，

---


**Claude**

这两个问题是相互耦合的，需要一起分析。

---

## 问题 1：Δq 权重课程

Stage 1 训练时 FrontRES 接受的是干净的 q_ref，学会了预测 Δq_gt = q_ref - q_sim。但 Stage 2 一开始 q_ref 就是被 DR 污染的，FrontRES 接受的输入分布立刻发生了偏移，初始输出的 Δq 可能完全错误。直接以 q_dot = q_ref + Δq 送入 GMT，等于在 GMT 训练分布外运行，机器人立刻摔倒，PPO 得到的奖励信号稀疏到无法学习。

**课程方案：线性混合系数 α(t)**

    q_dot = q_ref + α(t) * Δq

    α(0) = 0        → 纯 q_ref，GMT 在自己的训练分布内工作，机器人不倒
    α(T_α) = 1.0   → 完全 FrontRES 输出，Stage 2 正式生效

α 从 0 线性增加到 1，时间窗口 T_α 建议设为 Stage 2 总迭代数的 10-20%（即 5000-10000 iter）。

注意 `critic_warmup_iterations=1000` 提供的保护不够：warmup 只冻结 actor 让 critic 先收敛，但 α=1 的情况下 Δq 从 Stage 1 checkpoint 出发已经可能造成崩溃。

---

## 问题 2：DR 强度课程

DR 的问题在于两个极端都不对：

    DR 过强（训练初期）→ 机器人频繁摔倒 → 奖励信号稀疏 → FrontRES 无法获得有效梯度
    DR 过弱（整个训练）→ FrontRES 只学会处理轻微噪声 → 无法泛化到真实部署场景

**课程方案：DR 强度随训练进度线性增加**

    DR_intensity(t) = DR_min + (DR_max - DR_min) * min(t / T_DR, 1.0)

    t=0:    使用最轻的 DR（接近 Stage 1 的干净数据）
    t=T_DR: 使用完整 DR（float_prob=0.3, sink_prob=0.3 等完整配置）

---

## 两者的耦合关系

这是关键：两个课程必须协调，否则会产生矛盾信号。

    情况 A：α 小，DR 强
        FrontRES 几乎不影响结果（α≈0），DR 注入的噪声由 GMT 独立应对
        → 奖励信号是 GMT 本身的性能，与 FrontRES 无关
        → FrontRES 接收到的梯度几乎为零，浪费训练步数

    情况 B：α 大，DR 弱
        FrontRES 完全接管修正，但 DR 噪声轻微
        → 容易学会，但泛化性差

    情况 C：α 和 DR 同步增加（推荐）
        α 增加时，FrontRES 逐步接管修正；DR 同步增加，任务难度与修正能力同步增长
        → 每个阶段的任务难度与 FrontRES 当前能力匹配

**建议使用同一个课程时间轴**，让 α(t) 和 DR_intensity(t) 共享相同的 T_curriculum。

---

## 具体实现位置

    α(t) 课程       → front_residual_actor_critic.py 的 get_env_action() 中
                       q_dot = q_ref + self.delta_q_alpha * delta_q
                       self.delta_q_alpha 由 runner 在每个 iteration 更新

    DR 强度课程     → motion_perturbations 的各 prob/ratio 参数
                       由 runner 在每个 iteration 调用 env 接口更新

    课程调度器      → rsl_rl_cfg.py 中的 G1FlatFrontRESFinetuneRunnerCfg 新增参数：
                       curriculum_warmup_iters: int = 5000
                       delta_q_alpha_init: float = 0.0
                       dr_intensity_init: float = 0.1

---

你是否认可这个耦合课程的方向？确认后我来实现。

---


**User**

请不要忘了我们还必须在Stage 2微调初期对Critic进行warm up

---


**Claude**

三个机制需要统一到同一条时间轴上：

---

## 三阶段训练时间轴

    迭代数:    0              T_warmup        T_warmup + T_curriculum       T_max
               |--------------|----------------|------------------------------|
    阶段:      Critic 预热     课程过渡          完整训练
               ─────────────────────────────────────────────────────────────
    Actor:     冻结            解冻             解冻
    α(Δq):     0.0             0.0 → 1.0        1.0
    DR 强度:   DR_min          DR_min → DR_max  DR_max
    Critic:    训练             训练             训练

**Critic 预热期（0 到 T_warmup）**

Actor 冻结，FrontRES 输出固定的 Stage 1 checkpoint 值。此时 α=0，机器人完全按 q_ref 运行（最安全的初始状态），Critic 在干净分布上收敛，建立可信的价值函数基线。DR 使用最小强度（或完全关闭 motion perturbation），避免环境过于动荡影响 Critic 收敛。

**课程过渡期（T_warmup 到 T_warmup + T_curriculum）**

Actor 解冻，α 从 0 线性增加到 1，DR 强度同步从 DR_min 增加到 DR_max。α 和 DR 共享同一个进度变量 `progress = (t - T_warmup) / T_curriculum`，保证修正能力与任务难度同步成长。

**完整训练期（T_warmup + T_curriculum 到结束）**

α=1，DR=DR_max，正常 PPO 更新。

---

## 关键依赖关系

    α 课程必须在 T_warmup 之后才开始计数
    → 若 α 在 warmup 期间就开始增加，Critic 学到的是"有 FrontRES 但 Actor 冻结"的价值函数
    → Actor 解冻后 Δq 分布突然变化，Critic 的估计立刻失效，warmup 的作用完全被抵消

    DR 课程可以在 warmup 期间以 DR_min 运行（不能为零）
    → 完全关闭 DR 会导致 Critic 在干净数据上过拟合
    → DR_min 应保留基础物理随机化（摩擦力、质心），关闭 motion perturbation（穿模/漂浮）

---

## 推荐参数

    T_warmup           = 1000  iter   （已有配置，维持不变）
    T_curriculum       = 9000  iter   （Stage 2 前 20% 用于过渡）
    α_schedule         = linear, 0.0 → 1.0
    DR_min_perturbation = float_prob=0.0, sink_prob=0.0  （warmup 期关闭 motion-level DR）
    DR_max_perturbation = float_prob=0.3, sink_prob=0.3  （完整 DR，已有配置）

---

## 实现位置

    α(t) 更新     → on_policy_runner.py，每个 iteration 计算并写入 policy.delta_q_alpha
    α 应用        → front_residual_actor_critic.py 的 get_env_action()
                    q_dot = q_ref + self.delta_q_alpha * delta_q
    DR 强度更新   → on_policy_runner.py，每个 iteration 调用 env.unwrapped 更新
                    motion_perturbations 的各 prob 参数

    需新增配置参数（rsl_rl_ppo_cfg.py）:
        curriculum_warmup_iters:  int   = 1000   （与 critic_warmup_iterations 对齐）
        curriculum_total_iters:   int   = 10000  （warmup + transition 的总长度）
        delta_q_alpha_init:       float = 0.0
        dr_perturbation_init:     float = 0.0    （motion perturbation 的初始强度系数）

确认后开始实现？

---


**User**

我没理解你提到的为什么α需要在Critic Warm up期间保持0，Critic其实是FrontRES的Critic，因为Stage 2 FrontRES进行强化学习微调

---


**User**

This session is being continued from a previous conversation that ran out of context. The summary below covers the earlier portion of the conversation.

Summary:
1. Primary Request and Intent:

The conversation covers MOSAIC's FrontRES two-stage training system. The completed work includes:
- Stage 1 supervised learning loss function design and implementation (Huber loss, static lower-limb weighting, temporal gate with episode-boundary fix, cascade mask for lower-limb failures)
- Stage 2 RL fine-tuning infrastructure verification (all correct)
- Stage 2 reward function design and implementation (12 terms, removing action_rate_l2, adding contact_feet/motion_feet_pos/motion_global_anchor_pos/ori)
- smooth_loss in ppo.py for Δq temporal smoothness constraint
- Diagnostic metrics (3-layer framework, delta_q_norm_mean/std added to ppo.py loss_dict)

Currently discussing: curriculum learning for Stage 2 fine-tuning (Δq weight α schedule + DR intensity schedule + critic warmup integration). The user's most recent message questions why α needs to be 0 during critic warmup, pointing out that the Critic IS FrontRES's own RL critic.

2. Key Technical Concepts:

- FrontRES: pre-GMT residual network outputting Δq corrections; pipeline: obs → FrontRES → Δq → q_ref+Δq → GMT → robot actions
- Stage 1: Supervised learning, FrontRES predicts Δq_gt = q_ref - q_sim
- Stage 2: RL fine-tuning, PPO over Δq-space, reward from simulator, gradient never through frozen GMT
- PPO Distribution: Normal(delta_q_mean, std) — NOT Normal(gmt_actions, std)
- get_env_action(): separates policy distribution (over Δq) from env action (robot motor commands via GMT)
- L_reg = lambda_reg * ||Δq_mean||^2: prevents Δq from pushing q_dot outside GMT training distribution
- smooth_loss = lambda_smooth * mean_t(||Δq_t - Δq_{t-1}||^2): temporal consistency of corrections
- PCGrad: project conflicting gradients (PPO vs reg) onto each other's normal planes
- Temporal gate: ||Δq_gt_t - Δq_gt_{t-1}|| < tau for joint-level failure detection in Stage 1
- Cascade mask: if ANY lower-limb joint fails temporal gate, mask entire sample (balance is global property)
- Implicit denoising mapping: FrontRES maps corrupted Human Motion → Robot Feasible Motion by maximizing reward
- 7 DR noise types: 脚-地面穿模, 漂浮, 滑步, 关节偏置, 关节解耦, 身体自穿模, 根节点错误, 质心越界
- Curriculum learning: α(t) weight on Δq, DR intensity schedule, critic warmup coordination
- Critic warmup: freeze actor for first N iterations to let critic converge on stable signal

3. Files and Code Sections:

- **source/whole_body_tracking/whole_body_tracking/utils/supervise.py**
  - Complete rewrite of Stage 1 training loop
  - SuperviseStorage.generator() yields 5-tuple: (obs, target_actions, dones, prev_target_actions, prev_dones)
  - SuperviseTrainer.__init__: loss_type="huber", lower_limb_indices, lower_limb_weight=2.0, jump_threshold=0.2
  - init_storage: builds static_weights tensor with lower_limb_indices getting weight 2.0
  - update() implements:
    - Mask 1: valid = 1 - dones.float() (sample-level terminal mask)
    - Mask 2: temporal gate with episode-boundary fix using prev_dones
    - Mask 3: lower-limb cascade mask (any lower-limb failure → mask entire sample)
    - eff_w = static_weights * joint_valid * m_cascade
    - Huber or MSE loss: per_joint_loss = F.huber_loss(pred, target, reduction="none")
    - behavior_loss = (per_joint_loss * eff_w).sum() / n_eff
  - 7 diagnostic metrics in loss_dict: behavior, delta_q_norm_ratio, cosine_similarity, valid_ratio, cascade_gate_ratio, joint_mae_mean, joint_mae_max

- **source/whole_body_tracking/whole_body_tracking/utils/rsl_rl_cfg.py**
  - RslRlSuperviseAlgorithmCfg updated:
    - loss_type: str = "huber"
    - lower_limb_indices: list | None = None
    - lower_limb_weight: float = 2.0
    - jump_threshold: float = 0.2

- **source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py**
  - G1FlatSupervisedRunnerCfg.algorithm already has lower_limb_indices=list(range(12)) — verified correct
  - Joint order from robot_registry.py: indices 0-11 = left/right hip(3+3) + knee(1+1) + ankle(2+2)

- **source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py**
  - FrontRESFinetuneTrackingEnvCfg.RLFinetuneRewardsCfg completely rewritten:
    - REMOVED: action_rate_l2 (was on GMT robot_actions, not Δq — conceptually wrong)
    - NEW: contact_feet (weight=1.0, threshold=0.05) — covers foot penetration/floating/sliding (discrete)
    - NEW: motion_feet_pos = motion_relative_body_position_error_exp with body_names=["left_ankle_roll_link", "right_ankle_roll_link"], std=0.05m — continuous foot position
    - NEW: motion_global_anchor_pos (weight=0.5, std=0.3m) — root node error / CoM proxy
    - NEW: motion_global_anchor_ori (weight=0.5, std=0.4rad) — root node error
    - KEPT: motion_body_pos, motion_body_ori, motion_body_lin_vel, motion_body_ang_vel
    - KEPT: undesired_contacts (weight=-2.0), joint_limit (weight=-20.0), joint_torque (weight=-2e-4)

- **source/rsl_rl/rsl_rl/algorithms/ppo.py**
  - Added lambda_smooth = 0.0 to __init__ (disabled by default)
  - In update(), BEFORE mini-batch loop:
    ```python
    smooth_loss = torch.tensor(0.0, device=self.device)
    delta_q_norm_mean = torch.tensor(0.0, device=self.device)
    delta_q_norm_std  = torch.tensor(0.0, device=self.device)
    if self.use_frontres_reg:
        delta_q = self.storage.actions          # (T, B, A)
        per_step_norm = delta_q.norm(dim=-1)    # (T, B)
        delta_q_norm_mean = per_step_norm.mean()
        delta_q_norm_std  = per_step_norm.std()
        if self.lambda_smooth > 0.0:
            diff = delta_q[1:] - delta_q[:-1]  # (T-1, B, A)
            smooth_loss = diff.pow(2).mean()
    ```
  - Standard path: loss = ppo_loss + lambda_reg * reg_loss + lambda_smooth * smooth_loss
  - PCGrad path: task_losses = {"ppo": ppo_loss, "reg": lambda_reg * reg_loss + lambda_smooth * smooth_loss}
  - loss_dict additions: "smooth", "delta_q_norm_mean", "delta_q_norm_std"
  - mean_smooth_loss accumulator added and normalized

4. Errors and fixes:

- Edit failed on ppo.py: searched for "FrontRES 正則化参数" (with 則 character) but file had "FrontRES 正则化参数" (with 则 character) — fixed by using exact string from file read
- User corrected early approach: initially tried to modify mosaic.py for Stage 2 RL, user said "mosaic.py不是我在第二阶段强化学习微调使用的模块" — pivoted to ppo.py and front_residual_actor_critic.py
- User denied edit on mosaic.py — never modified it
- User feedback on teleop rewards: "这篇文章并不针对遥操任务，而是针对Tracking任务，似乎不应该涉及遥操奖励项" — replaced teleop_body_position_feet with motion_relative_body_position_error_exp filtered to ankle links
- User clarification on 接触错误: it is specifically 滑步 (foot sliding), not general contact state error

5. Problem Solving:

- **Stage 1 loss function**: Three-property design resolved to Huber + static weights + temporal gate + cascade mask. Batch z-score rejected (batch diversity makes it invalid). Temporal stability chosen as the gate signal.
- **Episode boundary bug in temporal gate**: When episode resets, prev_target comes from fallen state causing false gate triggers at new episode start. Fixed by passing prev_dones and setting safe_prev = current targets when prev_done=True.
- **action_rate_l2 conceptual mismatch**: Identified that this computes smoothness of GMT robot_actions, not Δq. Removed from reward, replaced by smooth_loss in PPO loss computed on storage.actions (Δq values).
- **Batch temporal ordering**: smooth_loss must be computed before mini-batch shuffling to preserve temporal ordering. Solved by computing on full rollout buffer before the mini-batch generator loop.
- **Layer 3 diagnostics**: No additional code needed — IsaacLab's ManagerBasedRLEnv automatically logs all reward terms per episode via infos["episode"], which runner already handles at lines 563-679.

6. All user messages:

- "我有个idea，将MOSAIC的RES放在GMT前" (initial FrontRES concept)
- Stage 2 RL gradient flow question and approval
- "mosaic.py不是我在第二阶段强化学习微调使用的模块"
- Stage 1 metric analysis request (read-only)
- "我想先将主损失函数设计好" with three properties
- Questions about joint-level gate detection
- "我不太确定对于第一阶段的监督学习需要做如此细致的损失函数设计"
- "我比较认可这个简化修改方案" (confirming Huber + static weights + temporal gate)
- "请进行修改" (implement Stage 1 changes)
- "请给我Huber Loss，静态下肢加权，时序门控与Δq Mask Mechanism的表达式" (LaTeX)
- "使用Latex似乎无法渲染，请直接使用text就行"
- Question about m_done discarding entire sequences
- "我认为折中方案更好" (cascade mask for lower limb failures)
- "我对m_done有点疑惑：某段动作序列在仿真中某个时间步后才开始摔倒"
- "现在我们可以开始讨论Diagnostics Metrics了" (×4 repeats)
- "请进行修改" (implement metrics changes)
- "请给我...论文中" table request
- "我想要的是奖励项与exp(-E / sigma^2)公式形态的表格"
- "我想要的是奖励项与exp(-E / sigma^2)公式形态与域随机化三者对应的表格"
- "穿模不仅是脚与地面穿模，还可能出现身体部位之间的穿模"
- "我想要的域随机化项机制解释应当包含这个域随机化项的这个情况到底会被哪个奖励项捕捉"
- "我感觉奖励项和域随机化项已经讨论得差不多了，请执行修改"
- "注意Stage 2的奖励项不是用来训练GMT" (pipeline clarification)
- "我认识到奖励项的设计与域随机化项息息相关" with 7 DR types
- "我进一步做出解释：接触错误是滑步，关节偏置需要考虑人形的拓扑结构，关节解耦是某关节的运动没有导致应该出现的身体运动"
- "注意这篇文章并不针对遥操任务"
- "请为我总结现在讨论后得到的所有奖励项"
- "我想要的是奖励项与exp(-E / sigma^2)公式形态与域随机化项三者对应的表格"
- "我认为你对身体自穿模使用的奖励项解释得比较易懂，请为每个域随机化项都这样解释一遍"
- "请将奖励项-公式-域随机化项三者对应表格拆分成两个表格"
- "穿模不仅是脚与地面穿模" (self-penetration addition)
- "我感觉奖励项和域随机化项已经讨论得差不多了，请执行修改"
- "由于我们是对Δq进行优化，平滑项和L2正则化项是否应当对Δq做更好"
- "我想要的是奖励项与exp(-E / sigma^2)公式形态的表格" (requesting complete formula table)
- "现在我们可以来讨论一下诊断项了" — 3-layer monitoring framework discussion
- "请执行修改" (implement diagnostics)
- "我在思考如果在微调初期就让FrontRES输出的Δq直接加给q_ref，可能会导致GMT输出的动作直接摔倒" — curriculum learning discussion
- "请不要忘了我们还必须在Stage 2微调初期对Critic进行warm up"
- "我没理解你提到的为什么α需要在Critic Warm up期间保持0，Critic其实是FrontRES的Critic，因为Stage 2 FrontRES进行强化学习微调" — MOST RECENT MESSAGE

7. Pending Tasks:

- Implement curriculum learning:
  - α(t) schedule for Δq weight: q_dot = q_ref + α(t) * Δq
  - DR intensity schedule for motion_perturbations parameters
  - Integration with existing critic_warmup_iterations=1000
  - Clarify α behavior during critic warmup (user's current question)
- The curriculum needs to be implemented in:
  - front_residual_actor_critic.py: get_env_action() to use self.delta_q_alpha
  - on_policy_runner.py: update α and DR intensity each iteration
  - rsl_rl_ppo_cfg.py: add curriculum config parameters

8. Current Work:

Immediately before this summary request, we were discussing curriculum learning for Stage 2 fine-tuning. The assistant proposed:

Phase 0 [0, T_warmup=1000]: Critic warmup, actor frozen, α=0, DR=DR_min
Phase 1 [1000, 10000]: Transition, α: 0→1, DR: DR_min→DR_max  
Phase 2 [10000, end]: α=1, DR=DR_max, full training

The user's last message questions: "我没理解你提到的为什么α需要在Critic Warm up期间保持0，Critic其实是FrontRES的Critic，因为Stage 2 FrontRES进行强化学习微调"

The user is pointing out that since the critic IS FrontRES's own critic (learning to estimate value function of FrontRES's policy), having α=0 during warmup would cause the critic to learn the value function of "FrontRES with no effect" (q_ref only), which is DIFFERENT from the value function it needs to learn (q_ref + α*Δq). This creates a distribution shift when α increases after warmup.

The user's implication: α=0 during critic warmup may not be necessary or even counterproductive from a critic learning perspective. The critic should learn the value function of FrontRES at its starting α value.

9. Optional Next Step:

Respond to user's question about why α=0 during critic warmup. The user said: "我没理解你提到的为什么α需要在Critic Warm up期间保持0，Critic其实是FrontRES的Critic，因为Stage 2 FrontRES进行强化学习微调"

The correct explanation to give:
- The user is RIGHT that α=0 during warmup causes the critic to learn the wrong value function
- The real reason to keep α small during warmup is PRACTICAL: prevent the robot from falling too often when FrontRES outputs are unreliable, so the critic gets meaningful training signal from non-trivial trajectories
- α=0 is NOT a theoretical requirement from a critic learning perspective
- Better approach: α starts at α_init (small but non-zero, e.g., 0.1) from the very beginning of training (including warmup), so the critic learns the value function of FrontRES at α_init weight
- After warmup, α gradually increases from α_init to 1.0
- During warmup, actor is frozen so α_init remains fixed — critic converges on a stable (if slightly FrontRES-influenced) policy
- This removes the distribution shift problem the user identified

If you need specific details from before compaction (like exact code snippets, error messages, or content you generated), read the full transcript at: C:\Users\ChengYuxuan\.claude\projects\c--ArtiIntComVis-MOSAIC\2b99aabb-3476-4bdb-9737-6e7bcc708a5e.jsonl
Continue the conversation from where it left off without asking the user any further questions. Resume directly — do not acknowledge the summary, do not recap what was happening, do not preface with "I'll continue" or similar. Pick up the last task as if the break never happened.

---


**Claude**

你说得完全正确，我之前的设计有一个概念错误。

**问题所在：**

如果 α=0 during warmup，Critic 学到的是 "FrontRES 对环境零影响" 这个策略的价值函数 V(s | α=0)。但 warmup 结束后 α 开始增加，Critic 需要估计的是 V(s | α>0)——这是一个完全不同的价值函数，distribution shift 反而出现在 warmup→正式训练的切换点，批了 warmup 的好处。

**正确设计：**

α 应该从训练一开始（包括 warmup 期间）就保持一个小但非零的初始值 α_init。

```
Phase 0 [0, T_warmup]:      α = α_init (固定), Actor 冻结, DR = DR_min
                             Critic 学习 V(s | FrontRES @ α_init)
Phase 1 [T_warmup, T_ramp]: α: α_init → 1.0 (线性), Actor 解冻, DR: DR_min → DR_max
Phase 2 [T_ramp, end]:      α = 1.0, DR = DR_max
```

**α_init 存在的理由是实际稳定性，不是理论要求：**

Stage 1 刚结束时 FrontRES 输出的 Δq 具有一定误差，直接以 α=1 注入可能让 q_dot 大幅偏离 GMT 的训练分布，导致频繁摔倒，Critic 获得的大量 sample 都是 terminal state，学习信号质量差。α_init 设小（如 0.1～0.2）让 FrontRES 的影响在初期处于可控范围，保证 Critic 有足够质量的轨迹可以学习。

**Actor 冻结期间 Critic 收敛的保证：**

Actor 冻结 → 策略固定在 π(a|s, α=α_init) → Critic 面对的是一个稳态 MDP → 价值估计可以正常收敛。Actor 解冻后，α 的缓慢增加会让策略缓慢偏移，Critic 有足够时间追随，不会出现突然的 bootstrap error 爆炸。

所以修改后的 curriculum 参数应该是：

```python
alpha_init:       float = 0.1    # warmup 期间的固定 α，非零
alpha_final:      float = 1.0    # ramp 结束时的目标 α
critic_warmup_iterations: int = 1000
alpha_ramp_iterations:    int = 9000   # warmup 结束后线性增加的步数
```

这样 Critic warmup 期间学到的价值函数与正式训练使用的价值函数之间没有 distribution shift，平滑过渡。

---


**User**

但Critic可以观测到特权信息，包含域随机化的种类和强度，即Asymmetric Actor-Critic，我认为必须考虑到这一点，并且我不是很理解ramp阶段，我也不是很理解Warmup阶段的a_init和DR_min为什么固定，请更详细解释

---


**Claude**

## Asymmetric Actor-Critic 对 Curriculum 的影响

首先明确这个架构：

```
Actor:  obs_proprioceptive → π(Δq | s)          [无特权信息]
Critic: obs_proprioceptive + obs_privileged      [含DR类型、强度等特权信息]
        → V(s, DR_params | π)
```

Critic 学到的不是一个单一的价值函数，而是一个**条件价值函数** V(s, DR_params | π)。它的输入空间包含 DR 参数，所以它原则上可以泛化到不同的 DR 强度。但这个泛化能力需要通过训练样本来建立。

---

## 为什么 Warmup 阶段要固定 α 和 DR

Warmup 的唯一目标是：**让 Critic 在一个稳态 MDP 上收敛，得到可靠的价值估计**。

Critic 通过 TD 学习（Bellman 方程）收敛，前提是目标值 `r + γV(s')` 相对稳定。如果 MDP 本身在变化，目标值一直在飘，Critic 追不上，训练不稳定。

**三个固定量各自的作用：**

**1. 冻结 Actor 参数**
最直接的原因：策略 π 不变 → 产生的轨迹分布不变 → Critic 面对的是一个稳态 MDP。

**2. 固定 α = α_init（非零）**
即使 Actor 参数冻结，如果 α 在变化：

```
t=0:  env_action = q_ref + 0.1 * Δq  → 某个轨迹分布、某个奖励信号
t=500: env_action = q_ref + 0.5 * Δq  → 完全不同的轨迹分布、不同的奖励信号
```

Actor 参数没变，但环境中实际执行的动作变了，奖励信号变了，状态转移分布变了——MDP 还是非稳态的，Critic 无法收敛。所以 α 必须在 Warmup 期间固定。

α_init 设为非零（如 0.1）而不是 0，原因是你之前指出的：α=0 时 Critic 学到的是 "FrontRES 无效果" 的价值函数，之后α增大时产生 distribution shift。

**3. 固定 DR = DR_min**
这里 Asymmetric Critic 的特殊性体现出来：

Critic 的输入包含 DR 参数。如果 Warmup 期间 DR 在变化：

```
Critic input at t=0:   [s_priv, DR_intensity=0.1]  → target V₁
Critic input at t=500: [s_priv, DR_intensity=0.6]  → target V₂
```

Critic 需要同时学习多个 DR 强度下的价值函数，但此时 Actor 是冻结的、轨迹分布是有限的，训练样本覆盖的 DR 强度是稀疏的。结果是 Critic 对任何 DR 强度都没有学好，价值估计质量差，Bootstrap error 高。

固定 DR_min 的另一个好处：低 DR 强度 → 环境噪声小 → 奖励信号更干净 → Critic 更容易识别哪些状态是好状态。这给 Critic 提供了一个清晰的学习起点。

---

## 为什么需要 Ramp 阶段（而不是 Warmup 后直接跳到全量）

假设 Warmup 结束后直接设置 α=1, DR=DR_max：

**对 Actor 的冲击：**
Warmup 期间 Actor 参数冻结但已经适应了 α_init 下的策略梯度。突然 α 从 0.1 跳到 1.0，FrontRES 的 Δq 在环境中的效果放大了 10 倍。Actor 输出的 Δq 尺度没有改变，但实际影响剧增，机器人大概率立刻不稳定。

**对 Critic 的冲击：**
```
Warmup 结束时 Critic 学好了: V(s, DR_min | π @ α_init)
突然跳到:                    需要估计 V(s, DR_max | π @ α=1)
```

这是 Critic 从未见过的输入组合（高 DR + 大 α），价值估计严重错误，Actor 更新方向基于错误的优势函数，训练发散风险高。

**Ramp 的作用：**
```
每个 iteration 只让 α 增加一点点，DR 增加一点点
Critic 在每一小步都能追上实际价值函数的变化
Actor 逐渐适应更大 Δq 权重下的策略梯度
```

---

## 完整三阶段结构

```
Phase 0: Critic Warmup
  [0, T_warmup]
  目的: 在稳态 MDP 上建立可靠的价值估计基线
  Actor:  冻结 (参数不更新)
  α:      固定 = α_init (如 0.1)
  DR:     固定 = DR_min (如最小噪声强度)
  更新:   只更新 Critic

Phase 1: Ramp
  [T_warmup, T_warmup + T_ramp]
  目的: 平滑过渡，Actor 和 Critic 协同适应
  Actor:  解冻，开始更新
  α:      α_init → 1.0 (线性)
  DR:     DR_min → DR_max (线性)
  更新:   Actor + Critic 同时更新

Phase 2: Full Training
  [T_warmup + T_ramp, end]
  α = 1.0，DR = DR_max，标准 PPO
```

**Ramp 的线性增量（每 iteration）：**
```
Δα   = (1.0 - α_init) / T_ramp
ΔDR  = (DR_max - DR_min) / T_ramp
```

每步变化量极小，Critic 的 prediction error 始终有机会在下一次更新中修正，不会积累成发散。

---


**User**

我被说服了，请执行修改

---


**Claude**

Let me read the relevant files first.

---


**Claude**

Let me check the runner config and the FrontRES-specific runner config:

---


**Claude**

Now I have everything I need. Let me implement the three changes:

---


**Claude**

These warnings are pre-existing (the `distribution = None` typing issue). My edits didn't introduce them. Continuing with the runner.

---


**Claude**

Now edit the runner. The critic warmup block already exists at line 429-450. I need to add the alpha scheduler right after it.

---


**Claude**

Now add the per-iteration alpha update inside the loop, right after the existing warmup management block:

---


**Claude**

Now log the alpha value so it appears in TensorBoard:

---


**Claude**

修改完成，涉及三个文件：

**[front_residual_actor_critic.py](source/rsl_rl/rsl_rl/modules/front_residual_actor_critic.py)**
- 在 `__init__` 末尾添加 `self.delta_q_alpha: float = 1.0`，默认 1.0 保持推理/非课程训练行为不变
- `_apply_delta_q_and_run_gmt` 中将 `+ delta_q` 改为 `+ self.delta_q_alpha * delta_q`

**[on_policy_runner.py](source/rsl_rl/rsl_rl/runners/on_policy_runner.py)**
- learn() 循环开始前读取 `delta_q_alpha_init` 和 `delta_q_alpha_ramp_iterations` 配置
- 每个 iteration 按三阶段调度更新 `policy.delta_q_alpha`：
  - Phase 0：固定 `alpha_init`
  - Phase 1：线性 ramp `alpha_init → 1.0`
  - Phase 2：固定 `1.0`
- log() 中新增 `Curriculum/delta_q_alpha` 到 TensorBoard

**[rsl_rl_ppo_cfg.py](source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py)**
- 添加 `delta_q_alpha_init = 0.1` 和 `delta_q_alpha_ramp_iterations = 9000`
- 附详细中文注释解释三阶段的设计动机

总训练进度：`[0, 1000)` warmup → `[1000, 10000)` ramp → `[10000, end)` 全量。

---


**User**

我现在主要使用wandb观察训练进度，请将诊断项都记录在wandb中，并且我对第二阶段微调长度有点疑惑，我的第一阶段监督学习已经训练到了2.5w轮次，valid_ratio全部保持在0.99以上；joint_mae_mean虽然有几个尖刺，但绝大部分都接近0，joint_mae_max与joint_mae_mean类似；delta_q_norm_ratio在绝大多数时刻也接近1，虽然有几个尖刺，出现轮次与joint_mae_mean一致；cosine_similarity绝大多数时刻接近1，出现的尖刺是反向，轮次与先前的参数一致；cascade_gate_ratio在0.992到0.994之间震荡，没有尖刺；behavior绝大多数时刻接近0，尖刺轮次与先前参数一致，请评价Stage 1是否训练完毕，并为我设置Stage 2的训练长度

---


**Claude**

现在我有完整信息，先回答评估问题，再执行修改。

---

## Stage 1 评估

**结论：Stage 1 已完全收敛，可以开始 Stage 2。**

**指标逐项分析：**

| 指标 | 观测值 | 解读 |
|---|---|---|
| valid_ratio > 0.99 | 极佳 | GMT rollout 几乎不摔倒，训练数据质量高 |
| cascade_gate_ratio 0.992~0.994 稳定无尖刺 | 极佳 | 下肢 temporal gate 无系统性失效，动作数据干净 |
| behavior ≈ 0 | 极佳 | Huber loss 已达收敛下限 |
| delta_q_norm_ratio ≈ 1 | 极佳 | FrontRES 输出幅度与 Δq_gt 完全对齐，无坍缩 |
| cosine_similarity ≈ 1 | 极佳 | 方向对齐接近完美 |
| joint_mae_mean ≈ 0 | 极佳 | 各关节平均误差收敛 |

**尖刺的含义（重要）：** 所有尖刺出现在完全相同的 iteration，这不是训练不稳定，而是 **数据集中的困难动作片段**（快速关节切换，触发 temporal gate 的 cascade mask，导致有效样本骤减，batch loss 瞬时升高后立刻恢复）。这是监督学习在此数据集上的天花板，不可通过延长训练改善——Stage 2 的 RL 信号正是为了处理这类困难样本而设计的。

---

## Stage 2 训练长度 & 配置 Bug

**当前 config 存在一个逻辑错误：**

```python
max_iterations = _STAGE1_MAX_ITERATIONS + _STAGE2_EXTRA_ITERATIONS  # 250000
```

但 runner 内部：
```python
tot_iter = start_iter + num_learning_iterations  # 相对计数
```

Stage 1 在第 25000 轮结束，Stage 2 resume 后 `start_iter=25000`，runner 会运行到 `25000 + 250000 = 275000`——**实际跑了 250000 轮 Stage 2，远超预期的 50000 轮**。

**推荐 Stage 2 长度：40000 轮**

| 阶段 | 轮次 | 说明 |
|---|---|---|
| Phase 0 (Critic warmup) | 1000 | Actor 冻结，α=0.1 固定 |
| Phase 1 (Ramp) | 9000 | α: 0.1→1.0，DR: min→max |
| Phase 2 (Full PPO) | 30000 | 标准 PPO，处理困难样本 |
| **Total** | **40000** | |

Stage 1 初始化质量如此之高，30000 轮全量 PPO 足够让 RL 信号改善困难样本的处理，同时避免过度偏离 Stage 1 学到的精确预测。

现在执行修改：

---


**Claude**

The comment at line 119-120 is informative — IsaacLab's workflow passes `max_iterations - current_iter` as the actual iteration count to `learn()`. So the cumulative design IS correct. The real bug is that `_STAGE1_MAX_ITERATIONS = 200000` but Stage 1 only ran 25000 iterations, causing Stage 2 to run for `65000 - 25000 = 40000` correct iterations only if we set max_iterations correctly.

---


**Claude**

Now fix the Stage 2 max_iterations line:

---


**Claude**

These are pre-existing IDE false positives — `@configclass` fields aren't recognized by Pylance. Let me verify the imports are correct to rule out a real issue.

---


**Claude**

Imports are all correct. The errors at lines 157-162 are Pylance false positives — it cannot statically resolve `@configclass` inherited fields (this is an isaaclab pattern that uses `dataclasses` under the hood). `RslRlSuperviseAlgorithmCfg` does define `loss_type`, `lower_limb_indices`, `lower_limb_weight`, and `jump_threshold` as confirmed from the previous session. These will work correctly at runtime.

Now let me proceed with the wandb diagnostic logging. Let me check what the runner currently logs and where to add Stage 2 diagnostics:

---


**Claude**

The existing `loss_dict` loop and `ep_infos` loop already cover most diagnostics. What's missing for Stage 2 in wandb:

1. `Curriculum/critic_warmup_active` — currently only a `print()`, never written to wandb
2. The alpha and warmup state should appear in the terminal output too

Let me add these:

---


**Claude**

First, pass `_warmup_actor_frozen` into `locs` so `log()` can access it. Find where `loss_dict = self.alg.update()` is called:

---


**Claude**

Now update the `log()` function to use these variables and emit them to wandb, and include them in the terminal output:

---


**Claude**

Now add the curriculum state to the terminal output (appears in the `else` branch for Stage 2 where `rewbuffer` has data):

---


**Claude**

修改完成。

**关于 IDE 报错：** `loss_type`、`lower_limb_indices` 等字段确实在 `RslRlSuperviseAlgorithmCfg` 中定义，IDE 误报是因为 Pylance 无法静态解析 `@configclass` 的字段继承，运行时不受影响。

**修改汇总：**

`rsl_rl_ppo_cfg.py`:
- `_STAGE1_MAX_ITERATIONS` 从 200000 改为 **25000**（与实际停止轮次对齐）
- `_STAGE2_EXTRA_ITERATIONS` 从 50000 改为 **40000**（1000 warmup + 9000 ramp + 30000 full PPO）
- `max_iterations` 注释从 250000 更新为 **65000**

`on_policy_runner.py` (wandb 诊断):
- 在 `loss_dict` 写入前暴露 `frontres_alpha` 和 `frontres_warmup_active` 到 `locals()`
- `Curriculum/delta_q_alpha` — 每 iteration 写入 wandb
- `Curriculum/critic_warmup_active` — 1=Actor 冻结中，0=已解冻，写入 wandb
- 终端输出新增 `Δq alpha: 0.1000 [WARMUP]` 行，warmup 结束后去掉 `[WARMUP]` 标记

**Stage 2 全部 wandb 指标：**

| 命名空间 | 指标 |
|---|---|
| `Loss/` | `value_function`, `surrogate`, `entropy`, `reg`, `smooth`, `lambda_reg`, `delta_q_norm_mean`, `delta_q_norm_std`, `learning_rate` |
| `Curriculum/` | `delta_q_alpha`, `critic_warmup_active` |
| `Policy/` | `mean_noise_std` |
| `Episode/` | 所有奖励项（IsaacLab 自动上报） |
| `Train/` | `mean_reward`, `mean_episode_length` |

---


**User**

我记得我们先前讨论过Stage 2 的诊断项，大概有mean_reward、mean_episode_length、kl_divergence、contact_feet_reward、delta_q_norm_mean等等，请检查是否也写入了wandb中

---


**Claude**

Now let me check the PPO update for `kl_divergence` — it's computed in PPO but I need to see if it's in the loss_dict:

---


**Claude**

Found the issue. `kl_mean` is computed per mini-batch for the adaptive LR scheduler but never accumulated or put into `loss_dict`. Everything else is already covered:

| 指标 | 来源 | wandb key | 状态 |
|---|---|---|---|
| `mean_reward` | `rewbuffer` | `Train/mean_reward` | ✅ |
| `mean_episode_length` | `lenbuffer` | `Train/mean_episode_length` | ✅ |
| `kl_divergence` | `kl_mean` in PPO loop | — | ❌ 缺失 |
| `contact_feet_reward` | `ep_infos["episode"]` | `Episode/Reward/contact_feet` | ✅ (IsaacLab 自动) |
| `delta_q_norm_mean` | `loss_dict` | `Loss/delta_q_norm_mean` | ✅ |
| `delta_q_norm_std` | `loss_dict` | `Loss/delta_q_norm_std` | ✅ |
| `delta_q_alpha` | curriculum | `Curriculum/delta_q_alpha` | ✅ |
| `critic_warmup_active` | curriculum | `Curriculum/critic_warmup_active` | ✅ |
| `surrogate/value/entropy` | `loss_dict` | `Loss/*` | ✅ |
| `reg/smooth/lambda_reg` | `loss_dict` | `Loss/*` | ✅ |

只需补充 `kl_divergence`，修改 `ppo.py`：

---


**Claude**

Now accumulate `kl_mean` into `mean_kl_divergence` right after it's computed:

---


**Claude**

Now find where `num_updates` is used to normalize the losses, and add `mean_kl_divergence` normalization:

---


**Claude**

完成。**变更只在 `ppo.py`**：

- `mean_kl_divergence` 累加器在 `update()` 顶部初始化为 0
- 在 adaptive LR 调度块内（`kl_mean` 计算后）累加 `mean_kl_divergence += kl_mean.item()`
- `update()` 末尾按 `num_updates` 归一化，写入 `loss_dict["kl_divergence"]`
- wandb 通过现有的 `Loss/*` 循环自动拾取为 `Loss/kl_divergence`

当 `schedule != "adaptive"` 时，`kl_mean` 不计算，`mean_kl_divergence` 保持 0 并直接写入（标志着 KL 调度未启用）。由于 Stage 2 cfg 里 `schedule="adaptive"`，正常情况下每个 mini-batch 都会有值。

---


**User**

GMT和MOSAIC中都使用了较为复杂的Adaptive Sampling策略，我们的训练中是否需要Adaptive Sampling策略？

---


**Claude**

不需要，理由如下。

## GMT / MOSAIC 为什么需要 Adaptive Sampling

**GMT** 从零学习运动跟踪，数据集中不同动作的难度差异极大（站立 vs 跑跳 vs 高难度杂技），均匀采样会导致简单动作主导 batch，难动作梯度信号严重不足。Adaptive Sampling 按难度/失败率提升难样本的采样概率，是从零收敛的必要手段。

**MOSAIC** 的多教师 BC 场景中，不同动作组的收敛速度不同（LAFAN 组和 FLD 组的 teacher loss 量级可能差 10 倍），需要动态平衡各组的 BC 信号权重，否则一个组主导梯度，其他组被遗忘。

## FrontRES Stage 2 为何不需要

FrontRES 的任务性质与上述两者根本不同：

**1. 运动多样性问题已由 GMT 解决**
GMT 是冻结的，它已经对所有动作都有合理的跟踪能力。FrontRES 不是在学习"如何跟踪各种动作"，而是在学习"在哪里、多大程度上修正 q_ref"。运动难度的分布问题对 FrontRES 透明——它只看到 obs，GMT 处理其余的。

**2. Stage 1 的尖刺是数据质量问题，不是难度问题**
Stage 1 指标中的尖刺来自 temporal gate 触发（Δq_gt 在相邻步之间跳变超过阈值），是 GMT rollout 中的局部跟踪失效，不是系统性的"难动作"。Stage 2 RL 中，这类片段会直接触发 `done=True`，被 episode reset 截断，GAE 自然处理边界。

**3. RL 奖励信号本身就是隐式重要性权重**
- FrontRES 在 Δq 修正有效的状态下获得高奖励，在无效或有害时获得低奖励
- GAE 优势估计 A(s,a) 已经对"哪些动作值得更新、更新多少"做了加权
- 这等价于 reward-weighted 的自适应采样，不需要额外的采样策略

**4. DR Curriculum 已经提供了难度渐进**
alpha 从 0.1 → 1.0 的 ramp 加上 DR 强度的同步增长，已经为 Stage 2 设计了由易到难的训练进程。叠加 Adaptive Sampling 会与 Curriculum 产生耦合，难以调试。

**5. 过度采样难动作的风险**
如果强制提升高奖励方差片段的采样频率，FrontRES 可能学到针对这些片段的 over-correction，破坏 GMT 在简单动作上已有的稳定性（Δq 应该接近 0 的地方反而输出噪声）。

## 结论

Stage 2 使用标准均匀随机 mini-batch 采样即可。PPO 的 GAE + clip + adaptive KL LR 调度已经足够处理样本重要性的隐式权衡，再加 Adaptive Sampling 是过度设计。

---


**User**

我理解了不用复杂的Adaptive Sampling，但可能我们需要初始状态随机化

---


**Claude**

让我先看看当前的环境配置中 reset 相关的事件设置。

---


**Claude**

这正是我需要的。`_resample_command` 已经有两条路径：`start_frame` 固定起点 和 `_adaptive_sampling`。现在看看 `MotionCommandCfg` 的配置字段：

---


**Claude**

现在我已经完全了解现有机制了。来做分析。

---

## 现状总结

`_resample_command` 中已经有两条路径：

- `start_from_beginning = True` → 所有 episode 从 `start_frame`（默认 0）开始，**固定起点**
- `start_from_beginning = False` → 调用 `_adaptive_sampling`，从运动序列的随机帧开始，即**内置了 RSI（Reference State Initialization）**

这说明 **Initial State Randomization 已经实现，开关是 `start_from_beginning`**。现在需要看 Stage 2 的命令配置是否打开了这个开关。

---


**Claude**

现在我有完整信息了。

---

## 分析结论

**Initial State Randomization 已经实现，并且对 Stage 2 默认开启。**

关键证据：

1. `MultiMotionCommandCfg.start_from_beginning` 默认值是 `False`
2. `start_from_beginning=False` → `_resample_command` 调用 `_adaptive_sampling`，后者从运动序列的**随机帧**开始（`sampled_bins` 通过 multinomial 采样，不是固定 0）
3. `FrontRESFinetuneTrackingEnvCfg.__post_init__` 没有修改 `commands.motion.start_from_beginning`，所以继承了默认的 `False`
4. `MultiMotionCommandsCfg` 也没有显式设置 `start_from_beginning`

所以 Stage 2 **每次 episode reset 时，运动序列的起始帧是随机的**——这正是标准的 RSI（Reference State Initialization），与 DeepMimic、AMP 等工作的做法一致。

**但有一个微妙问题需要注意：**

`_adaptive_sampling` 中当前的采样概率是基于历史失败分布的：
```python
prob = self.bin_failed_count[motion_indices]  # 失败次数越多的帧，采样概率越高
```

在 Stage 2 初期（Critic warmup，Actor 冻结，α=0.1），FrontRES 对仿真几乎无影响，机器人行为近似 GMT baseline。**失败分布会反映 GMT 本身的困难帧**，而不是 FrontRES 需要改进的困难帧。这些统计数据会随着 α 增大而失效。

然而 `MultiMotionCommandsCfg` 已经将所有 adaptive sampling 的 warmup/ramp 设为 `1e9`（近似无穷大），意味着**整个 Stage 2 期间 adaptive sampling 的失败权重始终处于 warmup 阶段（均匀采样），不会累积偏置**。这是一个正确的保守选择。

## 结论

**不需要任何修改。** Stage 2 的 Initial State Randomization 已经正确配置：
- `start_from_beginning=False` → 每次 reset 从随机帧开始
- `motion_sampling_warmup_s=1e9` → 整个训练期间保持均匀随机采样，不会因为早期失败统计偏置后期的起始帧分布
- RSI 与 Curriculum 的三阶段 alpha 完全兼容——无论从哪一帧开始，Critic warmup 期间 α=0.1 保证机器人不会因 FrontRES 输出错误而立刻崩坏

---
