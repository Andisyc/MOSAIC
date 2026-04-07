# 我有个idea，将MOSAIC的RES放在GMT前，接受动作序列，输出对动作序列的修正量delta 
_Exported on 04/04/2026 at 19:39:03 GMT+8 from Claude via WayLog_


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
