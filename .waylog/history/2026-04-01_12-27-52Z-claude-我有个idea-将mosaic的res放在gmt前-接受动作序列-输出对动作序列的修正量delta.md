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
