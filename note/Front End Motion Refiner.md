# Front End Motion Refiner

## 本工作版本演进总结

本工作研究一个放置在冻结 GMT tracker 前端的轻量 motion refiner。FEMR 不替代 GMT，而是在参考动作进入 GMT 之前输出 task-space residual correction：

$$
a^{\mathrm{FEMR}}_t
=
[
\Delta x,
\Delta y,
\Delta z,
\Delta r,
\Delta p,
\Delta yaw,
\rho^p_t
].
$$

其中前六维是 HSL 学到的 root-level \(\Delta SE(3)\) 修复提案，最后一维 \(\rho^p_t\) 是 HRL/PPO 学到的 position rejoin rate。当前版本不再把最后两维解释为 `conf_pos` 和 `conf_rpy`；旧的 confidence gate 只保留为 legacy objective / ablation。

目标是修复视频/视觉动作提取产生的 reference-frame artifacts，使被污染的动作序列重新接近 clean motion，并保持 GMT 可执行。

### 1. 基础架构：前端残差修复

最初设计将 FEMR 放在 GMT 前端：

$$
x^{\mathrm{noisy}}
\xrightarrow{\mathrm{FEMR}}
x^{\mathrm{noisy}}+\Delta_\theta
\xrightarrow{\mathrm{GMT}}
\tau^{\mathrm{fr}}.
$$

这一设计保持 GMT 冻结，只训练 FEMR。这样可以把论文叙事固定为：

$$
\text{reference-frame error consumes robustness budget},
$$

而不是重新训练一个新的 tracker。

### 2. Task-Space Correction：从通用残差到动力学连续修复

FEMR 使用 task-space correction，而不是直接输出 joint-space residual：

$$
\Delta_\theta \in SE(3)
$$

主要原因是 reference-frame artifacts 通常表现为 root position、root orientation、yaw、roll/pitch 等可解释扰动。

早期 rp demo 分支只激活：

$$
M_{\mathrm{rp}}
=
[0,0,0,1,1,0],
$$

即只允许 FEMR 修复 roll/pitch 方向误差。该设计验证了姿态修复的有效性，但在高强度扰动下暴露出一个关键问题：将 reference 几何上强行拉回 Clean，特别是 root position，可能破坏与前一帧的动力学连续性。

因此当前 HSL+HRL 版本采用非对称分工：

$$
\Delta rpy^{\mathrm{write}}_t
=
\Delta rpy^{\mathrm{HSL}}_t,
$$

$$
\Delta p^{\mathrm{write}}_t
=
(1-\rho^p_t)\Delta p^{\mathrm{cont}}_t
+
\rho^p_t\Delta p^{\mathrm{HSL}}_t.
$$

HSL 负责回答“被污染的 root frame 应当往哪里修”，HRL/PPO 只负责回答“root position 应当以多大比例重新贴近 Clean 几何目标”。这样 PPO 不再直接改变修复方向，只学习动力学连续性与 Clean rejoin 之间的折中。

### 3. 三分支 Rollout：Clean / Noisy / FEMR

为了给 FEMR 提供稳定训练信号，代码使用三分支 rollout：

$$
\tau^{\mathrm{clean}}
=
\mathrm{GMT}(x^{\mathrm{clean}}),
$$

$$
\tau^{\mathrm{noisy}}
=
\mathrm{GMT}(x^{\mathrm{noisy}}),
$$

$$
\tau^{\mathrm{fr}}
=
\mathrm{GMT}(x^{\mathrm{noisy}}+\Delta_\theta).
$$

Clean 分支表示原始动作序列在仿真中的执行结果，是最终目标。Noisy 分支表示不修复时 GMT 的 baseline。FEMR 分支表示修复后的执行结果。

### 4. Oracle 分支：从修复上限改为可信度估计

早期版本引入 feasible oracle：

$$
\tau^{\mathrm{oracle}}
=
\mathrm{GMT}
\left(
x^{\mathrm{noisy}}
+
\Pi_{\mathcal{A}}(\Delta^\star)
\right),
$$

其中 \(\Pi_{\mathcal{A}}\) 将 clean repair target 投影到当前 active action cone。

这个 oracle 原本用于估计可修上限：

$$
R_{\mathrm{oracle}} - R_{\mathrm{noisy}}.
$$

但后续发现，如果 oracle 因动作锥、clip、projection 或奇异约束低于 Clean，那么它会错误限制 FEMR 的修复幅度。因此当前版本改为：

$$
\text{Clean is the target, oracle is trust.}
$$

Oracle 只用于判断样本是否可信：

$$
g_{\mathrm{oc}}
=
\max(R_{\mathrm{clean}}-R_{\mathrm{oracle}},0),
$$

$$
c_{\mathrm{oracle}}
=
\exp
\left(
-
\frac{g_{\mathrm{oc}}}{\tau_{\mathrm{oc}}}
\right).
$$

### 5. Reward 目标：从 executability gain 转向 Clean restoration

早期强化学习目标主要基于 executable gain：

$$
R_{\mathrm{fr}} - R_{\mathrm{noisy}}.
$$

这个目标能鼓励 FEMR 提高可执行性，但存在 reward hacking 风险：FEMR 可能学到“更容易被 GMT 执行”的 reference，而不是“更接近原始 Clean motion”的 reference。

当前版本将主目标改为 Clean restoration：

$$
r_{\mathrm{restore}}
=
e_{\mathrm{raw}} - e_{\mathrm{fr}},
$$

其中：

$$
e_{\mathrm{raw}}
=
d(x^{\mathrm{noisy}},x^{\mathrm{clean}}),
$$

$$
e_{\mathrm{fr}}
=
d(x^{\mathrm{noisy}}+\Delta_\theta,x^{\mathrm{clean}}).
$$

也就是说，FEMR 获得正收益当且仅当它让 corrupted reference 更接近 Clean。

### 6. Sample Selection：用 Clean Damage 选择可修样本

样本难度不再由 oracle gap 决定，而由 clean damage 决定：

$$
d_{\mathrm{clean}}
=
\max(R_{\mathrm{clean}}-R_{\mathrm{noisy}},0).
$$

然后通过 double-sigmoid repairability window 得到：

$$
\mu_{\mathrm{repair}}.
$$

PPO actor 的样本权重为：

$$
w_{\mathrm{actor}}
=
c_{\mathrm{oracle}}\mu_{\mathrm{repair}}
+
\alpha_{\mathrm{side}}
\left(
\mu_{\mathrm{safe}}+\mu_{\mathrm{broken}}
\right).
$$

这表示：真正主导 PPO 更新的是 clean damage 明显、处于可修区间、且 oracle 不明显低于 Clean 的样本。

### 7. Action Cost：从最小动作改为 Clean-Bounded 修正

旧 action cost 惩罚修正幅度：

$$
P_{\mathrm{mag}}
=
\sum_i w_i
\left(
\frac{\Delta_i}{\Delta_i^{\max}}
\right)^2.
$$

这会压制必要修正，导致 FEMR 输出很小，不适合 demo。

当前版本默认关闭普通 magnitude cost：

$$
w_i=0.
$$

新的约束是 Clean-bounded action cost。它不惩罚“修得大”，只惩罚：

$$
\text{超过 Clean 目标}
\quad\text{和}\quad
\text{偏离 Clean 方向}.
$$

因此：

$$
P_{\mathrm{clean\text{-}bound}}
=
P_{\mathrm{side}}
+
P_{\mathrm{over}}.
$$

这一机制同时解决两个问题：safe 区域不允许乱修，fragile 区域不压制必要修正。

### 8. Under-Repair Penalty：让 Demo 修复更明显

仅有 restoration reward 时，FEMR 可能只做小幅改善。为了让修复在 demo 中更明显，当前版本加入 under-repair penalty：

$$
\rho_{\mathrm{restore}}
=
\frac{
e_{\mathrm{raw}}-e_{\mathrm{fr}}
}{
e_{\mathrm{raw}}+\epsilon
},
$$

$$
P_{\mathrm{under}}
=
\lambda_{\mathrm{under}}
\mu_{\mathrm{repair}}
\left[
\max
(\rho_{\min}-\rho_{\mathrm{restore}},0)
\right]^2.
$$

它只在 repairable 区域要求 FEMR 达到最低恢复比例。

### 9. Harmful Penalty：防止动态上修坏

虽然主目标是 Clean restoration，但 FEMR 仍然不能输出会破坏 GMT 执行的修正。因此保留 harmful repair penalty：

$$
P_{\mathrm{harm}}
\propto
\max(R_{\mathrm{noisy}}-R_{\mathrm{fr}}-\epsilon_{\mathrm{harm}},0).
$$

该项通过 action gate 过滤 no-op 噪声。只有 FEMR 确实输出了修正，并且让 executability 变差时，才认为是 harmful repair。

### 10. 训练策略：HSL 提供修复方向，HRL 过滤位置连续性

训练分为三阶段：

1. HSL warmup：学习 clean-oriented \(\Delta SE(3)\) 修复提案。
2. Actor takeover：逐步提高 PPO 对 \(\rho^p_t\) 的控制权。
3. PPO fine-tuning：在 HSL anchor 下学习 root position rejoin rate。

总优化目标为：

$$
L_{\mathrm{total}}
=
w_{\mathrm{ppo}}L_{\mathrm{PPO}}
+
\lambda_VL_V
-
\lambda_HH(\pi_\theta)
+
\lambda_{\mathrm{sup}}L_{\mathrm{sup,total}}.
$$

这里 supervised loss 提供 clean restoration 方向，PPO reward 提供可执行性和物理约束。

当前混合版本中，PPO 的 actor gradient 只允许进入 \(\rho^p_t\) 输出维度，不能更新 \(\Delta SE(3)\) proposal direction。这样保留 HSL 的 clean-restoration 上限，同时限制 HRL 的作用域，降低 reward hacking 风险。

### 11. 当前代码中的核心模块

当前代码实际包含以下机制：

1. FrontRES task-space actor。
2. HSL \(\Delta SE(3)\) proposal + PPO \(\rho^p_t\) position rejoin。
3. Clean / Noisy / FEMR / Oracle 分支。
4. Clean restoration reward。
5. Clean damage based repairability window。
6. Oracle-Clean trust gate。
7. Clean-bounded action cost。
8. Under-repair penalty。
9. Harmful repair penalty。
10. PPO actor sample weighting。
11. Supervised anchor and rho-only PPO restriction。
12. Checkpoint probe。
13. TensorBoard / console diagnostics。

这些机制共同服务于一个目标：

$$
\boxed{
\text{learn to restore corrupted references toward Clean while remaining executable by GMT.}
}
$$

## 预期修改后的公式与流程

### 1. 三分支 Rollout

对同一段 motion，构造三条主分支：

$$
\tau^{\mathrm{clean}}
=
\mathrm{GMT}(x^{\mathrm{clean}}),
$$

$$
\tau^{\mathrm{noisy}}
=
\mathrm{GMT}(x^{\mathrm{noisy}}),
$$

$$
\tau^{\mathrm{fr}}
=
\mathrm{GMT}(x^{\mathrm{noisy}}+\Delta_\theta).
$$

FEMR 输出 task-space residual correction：

$$
\Delta_\theta
=
f_\theta(o).
$$

当前 task-space action layout 为：

$$
\Delta_\theta
=
[
\Delta x_\theta,
\Delta y_\theta,
\Delta z_\theta,
\Delta r_\theta,
\Delta p_\theta,
\Delta yaw_\theta
].
$$

rp-only 分支使用 action mask：

$$
M_{\mathrm{rp}}
=
[0,0,0,1,1,0].
$$

因此有效修正为：

$$
\bar{\Delta}_\theta
=
M_{\mathrm{rp}}\odot\Delta_\theta.
$$

### 2. Clean Restoration Target

Clean 是最终目标。Noisy 到 Clean 的真实修复目标为：

$$
\Delta^\star
=
x^{\mathrm{clean}} - x^{\mathrm{noisy}}.
$$

在 rp-only 分支中：

$$
\bar{\Delta}^{\star}
=
M_{\mathrm{rp}}\odot\Delta^\star.
$$

FEMR 的核心目标是：

$$
\bar{\Delta}_\theta
\approx
\bar{\Delta}^{\star}.
$$

### 3. Restoration Reward

rp 误差使用局部旋转切空间计算。

Noisy 到 Clean 的 rp error：

$$
e_{\mathrm{raw}}^{\mathrm{rp}}
=
\left\|
\log\left(
(q^{\mathrm{noisy}})^{-1}q^{\mathrm{clean}}
\right)_{\mathrm{rp}}
\right\|_2.
$$

FEMR 修正后到 Clean 的 rp error：

$$
e_{\mathrm{fr}}^{\mathrm{rp}}
=
\left\|
\log\left(
(q^{\mathrm{fr}})^{-1}q^{\mathrm{clean}}
\right)_{\mathrm{rp}}
\right\|_2.
$$

rp restoration reward：

$$
r_{\mathrm{restore}}^{\mathrm{rp}}
=
e_{\mathrm{raw}}^{\mathrm{rp}}
-
e_{\mathrm{fr}}^{\mathrm{rp}}.
$$

一般形式为：

$$
r_{\mathrm{restore}}
=
\sum_{k\in\{xy,z,rp,yaw\}}
w_k
\left(
e_{\mathrm{raw}}^k
-
e_{\mathrm{fr}}^k
\right).
$$

当前 rp-only 配置为：

$$
w_{\mathrm{rp}}=1,
\qquad
w_{xy}=w_z=w_{yaw}=0.
$$

因此：

$$
r_{\mathrm{restore}}
=
r_{\mathrm{restore}}^{\mathrm{rp}}.
$$

### 4. Executability Scores

定义执行分数：

$$
R_{\mathrm{clean}}
=
E(\tau^{\mathrm{clean}}),
$$

$$
R_{\mathrm{noisy}}
=
E(\tau^{\mathrm{noisy}}),
$$

$$
R_{\mathrm{fr}}
=
E(\tau^{\mathrm{fr}}).
$$

Oracle 分支为：

$$
\tau^{\mathrm{oracle}}
=
\mathrm{GMT}
\left(
x^{\mathrm{noisy}}
+
\Pi_{\mathcal{A}}(\Delta^\star)
\right),
$$

$$
R_{\mathrm{oracle}}
=
E(\tau^{\mathrm{oracle}}).
$$

\(\Pi_{\mathcal{A}}\) 表示将 Clean 修复目标投影到当前 active action cone。

本版本不再使用：

$$
R_{\mathrm{oracle}}
\quad
\text{as the repair upper bound}.
$$

Oracle 只用于判断样本可信度。

### 5. Clean Gap 与 Oracle Trust

Clean damage：

$$
d_{\mathrm{clean}}
=
\max(R_{\mathrm{clean}} - R_{\mathrm{noisy}},0).
$$

FEMR repair gain：

$$
g_{\mathrm{fr}}
=
R_{\mathrm{fr}} - R_{\mathrm{noisy}}.
$$

Repair ratio：

$$
\rho_{\mathrm{exec}}
=
\frac{
g_{\mathrm{fr}}
}{
\max(d_{\mathrm{clean}}, d_{\min})
}.
$$

Oracle-Clean gap：

$$
g_{\mathrm{oc}}
=
\max(R_{\mathrm{clean}} - R_{\mathrm{oracle}},0).
$$

Oracle trust：

$$
c_{\mathrm{oracle}}
=
\exp
\left(
-
\frac{g_{\mathrm{oc}}}{\tau_{\mathrm{oc}}}
\right).
$$

如果：

$$
g_{\mathrm{oc}} \gg 0,
$$

说明 oracle 明显低于 Clean。此时 oracle 不能限制 FEMR 的修复目标，只能降低该样本作为 PPO repair 样本的可信度。

### 6. Repairability Window

用 clean damage 构造 repairability window：

$$
\mu_{\mathrm{enter}}
=
\sigma
\left(
\frac{
d_{\mathrm{clean}} - d_{\mathrm{safe}}
}{T}
\right),
$$

$$
\mu_{\mathrm{exit}}
=
\sigma
\left(
\frac{
d_{\mathrm{broken}} - d_{\mathrm{clean}}
}{T}
\right).
$$

未归一化窗口：

$$
\mu_{\mathrm{raw}}
=
\mu_{\mathrm{enter}}
\mu_{\mathrm{exit}}.
$$

归一化 repair gate：

$$
\mu_{\mathrm{repair}}
=
\mathrm{clip}
\left(
\frac{
\mu_{\mathrm{raw}}
}{
\mu_{\mathrm{peak}}
},
0,1
\right).
$$

Safe gate：

$$
\mu_{\mathrm{safe}}
=
1-\mu_{\mathrm{enter}}.
$$

Broken gate：

$$
\mu_{\mathrm{broken}}
=
1-\mu_{\mathrm{exit}}.
$$

### 7. PPO Actor Sample Weight

PPO actor sample weight 为：

$$
w_{\mathrm{actor}}
=
c_{\mathrm{oracle}}\mu_{\mathrm{repair}}
+
\alpha_{\mathrm{side}}
\left(
\mu_{\mathrm{safe}}
+
\mu_{\mathrm{broken}}
\right).
$$

含义是：

$$
\text{repairable and oracle-trusted samples dominate PPO actor updates}.
$$

Safe 和 broken 样本只保留小权重，用于防止完全忽略边界行为。

### 8. Clean-Bounded Action Cost

旧 action cost：

$$
P_{\mathrm{mag}}
=
\sum_i
w_i
\left(
\frac{\Delta_{\theta,i}}{\Delta_i^{\max}}
\right)^2
$$

会压制必要修正。本版本默认：

$$
w_i=0.
$$

新 cost 使用 Clean target。

先归一化有效修正和目标：

$$
\tilde{\Delta}_\theta
=
\frac{\bar{\Delta}_\theta}{\Delta^{\max}},
\qquad
\tilde{\Delta}^{\star}
=
\frac{\bar{\Delta}^{\star}}{\Delta^{\max}}.
$$

Clean 方向：

$$
u
=
\frac{
\tilde{\Delta}^{\star}
}{
\|\tilde{\Delta}^{\star}\|_2+\epsilon
}.
$$

FEMR 修正沿 Clean 方向的分量：

$$
\tilde{\Delta}_{\parallel}
=
\left(
\tilde{\Delta}_\theta^\top u
\right)u.
$$

FEMR 修正偏离 Clean 方向的分量：

$$
\tilde{\Delta}_{\perp}
=
\tilde{\Delta}_\theta
-
\tilde{\Delta}_{\parallel}.
$$

Side cost：

$$
P_{\mathrm{side}}
=
\lambda_{\mathrm{side}}
\left\|
\tilde{\Delta}_{\perp}
\right\|_2^2.
$$

Over-correction cost：

$$
P_{\mathrm{over}}
=
\lambda_{\mathrm{over}}
\left[
\max
\left(
\tilde{\Delta}_\theta^\top u
-
\|\tilde{\Delta}^{\star}\|_2
-
m_{\mathrm{over}},
0
\right)
\right]^2.
$$

Clean-bounded action cost：

$$
P_{\mathrm{clean\text{-}bound}}
=
P_{\mathrm{side}}
+
P_{\mathrm{over}}.
$$

### 9. Under-Repair Penalty

Restoration ratio：

$$
\rho_{\mathrm{restore}}
=
\frac{
e_{\mathrm{raw}} - e_{\mathrm{fr}}
}{
e_{\mathrm{raw}}+\epsilon
}.
$$

Under-repair penalty：

$$
P_{\mathrm{under}}
=
\lambda_{\mathrm{under}}
\mu_{\mathrm{repair}}
\left[
\max
\left(
\rho_{\min}
-
\rho_{\mathrm{restore}},
0
\right)
\right]^2.
$$

这个项只在 repairable 区域鼓励 FEMR 修得足够明显。

### 10. Harmful Repair Penalty

Executable gain：

$$
g_{\mathrm{fr}}
=
R_{\mathrm{fr}} - R_{\mathrm{noisy}}.
$$

Raw harmful magnitude：

$$
h_{\mathrm{raw}}
=
\max(-g_{\mathrm{fr}}-\epsilon_{\mathrm{harm}},0).
$$

Action activity：

$$
a
=
\frac{
\|\bar{\Delta}_\theta\|_2^2
}{
\|\Delta^{\max}\|_2^2+\epsilon
}.
$$

Action-conditioned harm gate：

$$
w_{\mathrm{harm\text{-}action}}
=
\mathrm{clip}
\left(
\frac{
a-a_{\min}
}{
a_{\max}-a_{\min}
},
0,1
\right).
$$

Harm penalty：

$$
P_{\mathrm{harm}}
=
\lambda_{\mathrm{harm}}
w_{\mathrm{region}}
w_{\mathrm{harm\text{-}action}}
h_{\mathrm{raw}}.
$$

其中：

$$
w_{\mathrm{region}}
=
\mu_{\mathrm{repair}}
+
\beta_{\mathrm{broken}}\mu_{\mathrm{broken}}
+
\beta_{\mathrm{safe}}\mu_{\mathrm{safe}}.
$$

### 11. Progress Scaling

Reward progress：

$$
p_{\mathrm{DR}}
=
\mathrm{clip}
\left(
\frac{
\mathrm{DR\ scale}
}{
\mathrm{DR}_{\mathrm{ref}}
},
0,1
\right),
$$

$$
p_{\mathrm{actor}}
=
\mathrm{clip}
\left(
\mathrm{PPO\ actor\ weight},
0,1
\right),
$$

$$
p
=
p_{\mathrm{DR}}p_{\mathrm{actor}}.
$$

Constraint progress：

$$
p_c
=
p^{\gamma}.
$$

### 12. Final Intrinsic Reward

完整形式：

$$
r_\Delta
=
p
\left(
w_{\mathrm{restore}}r_{\mathrm{restore}}
+
w_{\mathrm{exec}}r_{\mathrm{exec}}
+
w_{\mathrm{rescue}}r_{\mathrm{rescue}}
\right)
-
p_c
\left(
P_{\mathrm{harm}}
+
P_{\mathrm{clean\text{-}bound}}
+
P_{\mathrm{under}}
+
P_{\mathrm{mag}}
\right).
$$

当前 rp-only demo-oriented 配置近似为：

$$
w_{\mathrm{restore}}=1,
\qquad
w_{\mathrm{exec}}=0,
\qquad
w_{\mathrm{rescue}}=0,
\qquad
P_{\mathrm{mag}}=0.
$$

因此：

$$
r_\Delta^{\mathrm{rp}}
=
p
\left(
e_{\mathrm{raw}}^{\mathrm{rp}}
-
e_{\mathrm{fr}}^{\mathrm{rp}}
\right)
-
p_c
\left(
P_{\mathrm{harm}}
+
P_{\mathrm{side}}
+
P_{\mathrm{over}}
+
P_{\mathrm{under}}
\right).
$$

### 13. PPO Loss

PPO ratio：

$$
r_t(\theta)
=
\frac{
\pi_\theta(a_t|s_t)
}{
\pi_{\theta_{\mathrm{old}}}(a_t|s_t)
}.
$$

Clipped PPO actor loss：

$$
L_{\mathrm{PPO}}
=
\mathbb{E}
\left[
w_{\mathrm{actor}}
\max
\left(
-r_t(\theta)A_t,
-
\mathrm{clip}
\left(
r_t(\theta),
1-\epsilon,
1+\epsilon
\right)A_t
\right)
\right].
$$

Value loss：

$$
L_V
=
\mathbb{E}
\left[
\left(
V_\theta(s_t)-\hat{R}_t
\right)^2
\right].
$$

### 14. Supervised Anchor

Supervised target：

$$
\bar{\Delta}^{\star}
=
M\odot\Delta^\star.
$$

Supervised regression loss：

$$
L_{\mathrm{sup}}
=
\mathrm{Huber}
\left(
\bar{\Delta}_\theta,
\bar{\Delta}^{\star}
\right).
$$

Direction loss：

$$
L_{\mathrm{dir}}
=
1
-
\cos
\left(
\bar{\Delta}_\theta,
\bar{\Delta}^{\star}
\right).
$$

Total supervised loss：

$$
L_{\mathrm{sup,total}}
=
L_{\mathrm{sup}}
+
\lambda_{\mathrm{dir}}L_{\mathrm{dir}}.
$$

### 15. Total Optimization Objective

总 loss：

$$
L_{\mathrm{total}}
=
w_{\mathrm{ppo}}L_{\mathrm{PPO}}
+
\lambda_VL_V
-
\lambda_HH(\pi_\theta)
+
\lambda_{\mathrm{sup}}L_{\mathrm{sup,total}}.
$$

其中：

$$
w_{\mathrm{ppo}}
=
\mathrm{PPO\ actor\ weight}.
$$

### 16. Training Flow

监督 warmup：

$$
\theta
\leftarrow
\arg\min_\theta
L_{\mathrm{sup,total}}.
$$

Actor takeover：

$$
w_{\mathrm{ppo}}:0\rightarrow1,
$$

$$
p_{\mathrm{actor}}:0\rightarrow1.
$$

PPO fine-tuning：

$$
\theta
\leftarrow
\arg\min_\theta
L_{\mathrm{total}}.
$$

最终约束为：

$$
\boxed{
\text{Clean is the target; oracle is trust; action cost is Clean-bounded.}
}
$$
