# Layer 0 Implementation Details：二维多智能体 Flocking 仿真环境

本文档用于指导课程 project 的 **Layer 0：基础 Python 仿真环境** 实现。Layer 0 的目标不是实现完整的 Olfati-Saber flocking 算法，而是先搭建一个清晰、可扩展、可调试的二维多智能体仿真框架，为后续 Layer 1–4 的 α-agent、γ-agent、β-agent、动态障碍物和改进人工势场模块打基础。

---

## 1. Layer 0 的目标

Layer 0 需要完成以下基础能力：

1. 搭建二维连续空间中的多智能体仿真环境；
2. 实现二阶点质量动力学；
3. 初始化 agent 的位置和速度；
4. 每个仿真步根据控制输入更新状态；
5. 支持速度上限、加速度上限、边界处理；
6. 支持圆形静态/动态障碍物的数据结构；
7. 实现基础可视化，包括轨迹图、动画和状态曲线；
8. 实现基础指标记录，例如平均速度、目标距离、最小 agent-agent 距离；
9. 预留接口，方便后续加入 α-agent flocking、γ-agent 导航、β-agent 避障和 dynamic IAPF。

Layer 0 可以理解为整个 project 的“物理仿真底座”。后续所有控制算法都只需要向环境提供控制输入：

\[
u_i(t)\in \mathbb{R}^2
\]

环境负责根据动力学方程更新：

\[
q_i(t),\quad p_i(t)
\]

---

## 2. 推荐开发环境

建议使用普通 Python 科学计算栈，不建议一开始使用 PettingZoo、VMAS、PyBullet 等大型多智能体或强化学习环境。原因是本 project 的核心是复现显式控制律，而不是训练 policy。自己写轻量级 simulator 更容易调试、解释和做 ablation。

### 2.1 Conda 环境

```bash
conda create -n mas_flocking python=3.10
conda activate mas_flocking
```

### 2.2 必需依赖

```bash
pip install numpy scipy matplotlib networkx imageio tqdm pyyaml jupyter
```

### 2.3 可选依赖

```bash
pip install numba shapely
```

各依赖的用途如下：

| 包 | 用途 |
|---|---|
| `numpy` | 位置、速度、加速度等矩阵运算 |
| `scipy.spatial.KDTree` | 后续用于加速邻居搜索 |
| `matplotlib` | 轨迹图、动画、指标曲线 |
| `networkx` | 后续用于邻近图、连通分量、图拉普拉斯和 \(\lambda_2\) |
| `imageio` | 保存 GIF 或视频 |
| `tqdm` | 显示仿真进度条 |
| `pyyaml` | 读取实验配置 |
| `jupyter` | 交互式调试和可视化 |
| `numba` | agent 数量较多时加速循环 |
| `shapely` | 后续处理复杂障碍物几何 |

---

## 3. 为什么 Layer 0 不建议直接用成熟多智能体框架？

常见多智能体环境包括：

- PettingZoo / MPE；
- VMAS；
- Mesa；
- PyBullet / gym-pybullet-drones。

这些框架适合多智能体强化学习、agent-based modeling 或真实机器人动力学，但对于本 project 的 Layer 0 并不是最合适。

本 project 需要完全控制下面这些细节：

1. 邻居集合 \(N_i\) 如何构建；
2. agent 状态 \(q_i,p_i\) 如何更新；
3. 控制律 \(u_i^\alpha,u_i^\beta,u_i^\gamma\) 如何叠加；
4. \(\sigma\)-norm、bump function、势函数如何实现；
5. β-agent 投影点如何生成；
6. 动态障碍物的相对速度和预测风险如何进入避障项；
7. 每个仿真步记录哪些指标。

大型框架虽然功能强，但会引入额外 API 和物理规则，不利于准确复现论文公式。因此建议：

> Layer 0 使用自己编写的 `numpy + matplotlib` 轻量 simulator；成熟 codebase 只作为参考，不作为主框架。

---

## 4. 推荐项目结构

建议先从简单结构开始，不要一开始过度工程化。

### 4.1 初始版本结构

```text
mas_flocking/
  main.py
  simulator.py
  obstacles.py
  controllers.py
  metrics.py
  visualize.py
  utils.py
  outputs/
    figures/
    animations/
    logs/
```

### 4.2 后续扩展版本结构

当 Layer 1–4 都加入后，可以扩展为：

```text
mas_flocking/
  configs/
    free_flocking.yaml
    static_obstacle.yaml
    dynamic_obstacle.yaml

  src/
    simulator.py
    agents.py
    obstacles.py
    controllers/
      __init__.py
      alpha_flocking.py
      gamma_navigation.py
      beta_obstacle.py
      dynamic_iapf.py
    metrics.py
    visualize.py
    utils.py

  experiments/
    run_free_flocking.py
    run_static_obstacle.py
    run_dynamic_obstacle.py
    run_ablation.py

  outputs/
    figures/
    animations/
    logs/
```

Layer 0 阶段建议只写第一种简单结构。等控制器模块增多后再拆分。

---

## 5. 核心状态变量设计

每个 agent 在二维平面中有：

\[
q_i \in \mathbb{R}^2
\]

\[
p_i \in \mathbb{R}^2
\]

\[
u_i \in \mathbb{R}^2
\]

其中：

- \(q_i\)：位置；
- \(p_i\)：速度；
- \(u_i\)：控制输入，即加速度。

所有 agent 的状态可以堆叠成矩阵：

\[
Q =
\begin{bmatrix}
q_1^T \\
q_2^T \\
\vdots \\
q_N^T
\end{bmatrix}
\in \mathbb{R}^{N\times 2}
\]

\[
P =
\begin{bmatrix}
p_1^T \\
p_2^T \\
\vdots \\
p_N^T
\end{bmatrix}
\in \mathbb{R}^{N\times 2}
\]

\[
U =
\begin{bmatrix}
u_1^T \\
u_2^T \\
\vdots \\
u_N^T
\end{bmatrix}
\in \mathbb{R}^{N\times 2}
\]

在代码里建议命名为：

```python
q: np.ndarray  # shape [N, 2], positions
p: np.ndarray  # shape [N, 2], velocities
u: np.ndarray  # shape [N, 2], accelerations / control inputs
```

---

## 6. 二阶点质量动力学

Olfati-Saber 框架中的基础 agent 模型是二阶系统：

\[
\dot q_i = p_i
\]

\[
\dot p_i = u_i
\]

离散仿真中可以使用 semi-implicit Euler：

\[
p_i(t+\Delta t)=p_i(t)+u_i(t)\Delta t
\]

\[
q_i(t+\Delta t)=q_i(t)+p_i(t+\Delta t)\Delta t
\]

相比直接用旧速度更新位置，semi-implicit Euler 在简单动力学中更稳定一些。

### 6.1 加速度裁剪

为了避免数值爆炸，需要限制控制输入：

\[
\|u_i\|\le u_{\max}
\]

### 6.2 速度裁剪

同样需要限制速度：

\[
\|p_i\|\le v_{\max}
\]

### 6.3 工具函数：向量范数裁剪

```python
import numpy as np

def clip_by_norm(x: np.ndarray, max_norm: float, eps: float = 1e-8) -> np.ndarray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    scale = np.minimum(1.0, max_norm / (norms + eps))
    return x * scale
```

---

## 7. `FlockingEnv` 类设计

Layer 0 的核心是环境类 `FlockingEnv`。它只负责状态初始化和动力学更新，不负责具体控制算法。

### 7.1 类接口

```python
class FlockingEnv:
    def __init__(
        self,
        n_agents: int,
        dt: float = 0.02,
        world_size: tuple[float, float] = (20.0, 12.0),
        v_max: float = 3.0,
        u_max: float = 8.0,
        seed: int = 0,
    ):
        ...

    def reset(self, init_mode: str = "random_left"):
        ...

    def step(self, u: np.ndarray):
        ...

    def get_state(self):
        ...
```

### 7.2 主要属性

```python
self.n              # agent 数量
self.dt             # 仿真步长
self.world_size     # 世界边界，例如 (width, height)
self.v_max          # 最大速度
self.u_max          # 最大加速度
self.q              # positions, shape [N, 2]
self.p              # velocities, shape [N, 2]
self.t              # 当前仿真时间
self.rng            # numpy random generator
```

### 7.3 推荐实现

```python
import numpy as np

class FlockingEnv:
    def __init__(
        self,
        n_agents: int,
        dt: float = 0.02,
        world_size: tuple[float, float] = (20.0, 12.0),
        v_max: float = 3.0,
        u_max: float = 8.0,
        seed: int = 0,
    ):
        self.n = n_agents
        self.dt = dt
        self.world_size = np.array(world_size, dtype=float)
        self.v_max = v_max
        self.u_max = u_max
        self.rng = np.random.default_rng(seed)

        self.q = None
        self.p = None
        self.t = 0.0

    def reset(self, init_mode: str = "random_left"):
        if init_mode == "random_left":
            # 初始化在世界左侧区域
            x = self.rng.uniform(1.0, 5.0, size=(self.n, 1))
            y = self.rng.uniform(1.0, self.world_size[1] - 1.0, size=(self.n, 1))
            self.q = np.hstack([x, y])
        elif init_mode == "random_center":
            self.q = self.rng.uniform(
                low=np.array([self.world_size[0] * 0.35, self.world_size[1] * 0.25]),
                high=np.array([self.world_size[0] * 0.65, self.world_size[1] * 0.75]),
                size=(self.n, 2),
            )
        else:
            raise ValueError(f"Unknown init_mode: {init_mode}")

        # 初始速度可以小随机，也可以全部朝右
        self.p = self.rng.normal(loc=0.0, scale=0.2, size=(self.n, 2))
        self.t = 0.0
        return self.get_state()

    def step(self, u: np.ndarray):
        if u.shape != (self.n, 2):
            raise ValueError(f"u should have shape {(self.n, 2)}, got {u.shape}")

        u = clip_by_norm(u, self.u_max)

        # semi-implicit Euler
        self.p = self.p + u * self.dt
        self.p = clip_by_norm(self.p, self.v_max)

        self.q = self.q + self.p * self.dt

        # Layer 0 先使用简单边界裁剪
        self.q = np.clip(self.q, [0.0, 0.0], self.world_size)

        self.t += self.dt
        return self.get_state()

    def get_state(self):
        return {
            "q": self.q.copy(),
            "p": self.p.copy(),
            "t": self.t,
        }
```

---

## 8. 边界处理策略

Layer 0 可以先用最简单的边界裁剪：

```python
self.q = np.clip(self.q, [0.0, 0.0], self.world_size)
```

但这会导致 agent 贴在边界时出现速度不合理的问题。后续可以改成以下两种方式。

### 8.1 反弹边界

如果 agent 撞到边界，位置裁剪，同时对应方向速度反向：

```python
for dim in range(2):
    low_hit = self.q[:, dim] < 0.0
    high_hit = self.q[:, dim] > self.world_size[dim]

    self.q[low_hit, dim] = 0.0
    self.q[high_hit, dim] = self.world_size[dim]

    self.p[low_hit | high_hit, dim] *= -0.5
```

### 8.2 软边界势场

后续可以把边界也当成墙壁障碍物，用 wall β-agent 或 repulsive potential 处理。这更贴近 Olfati-Saber 的思想，但 Layer 0 不需要实现。

---

## 9. 障碍物数据结构

Layer 0 只需要支持圆形障碍物。每个障碍物有：

- center；
- radius；
- velocity；
- 是否动态。

### 9.1 圆形障碍物类

```python
class CircleObstacle:
    def __init__(
        self,
        center: tuple[float, float],
        radius: float,
        velocity: tuple[float, float] = (0.0, 0.0),
        name: str = "obstacle",
    ):
        self.center = np.array(center, dtype=float)
        self.radius = float(radius)
        self.velocity = np.array(velocity, dtype=float)
        self.name = name

    def step(self, dt: float):
        self.center = self.center + self.velocity * dt
```

### 9.2 动态障碍物边界处理

如果动态障碍物超出边界，可以设置反弹：

```python
def step(self, dt: float, world_size=None):
    self.center = self.center + self.velocity * dt

    if world_size is not None:
        for dim in range(2):
            if self.center[dim] - self.radius < 0:
                self.center[dim] = self.radius
                self.velocity[dim] *= -1
            elif self.center[dim] + self.radius > world_size[dim]:
                self.center[dim] = world_size[dim] - self.radius
                self.velocity[dim] *= -1
```

---

## 10. 初始控制器：零控制与目标导航

Layer 0 不需要马上实现 flocking，可以先实现两个最简单的控制器，用于测试环境是否正确。

### 10.1 零控制

```python
def zero_control(q: np.ndarray, p: np.ndarray) -> np.ndarray:
    return np.zeros_like(q)
```

用零控制时，如果初始速度不为 0，agent 应该匀速运动。

### 10.2 简单目标导航控制

假设目标点为 \(q_g\)，设计一个 PD 控制器：

\[
u_i = k_p(q_g-q_i)-k_d p_i
\]

代码：

```python
def goal_pd_control(
    q: np.ndarray,
    p: np.ndarray,
    goal: np.ndarray,
    k_p: float = 1.0,
    k_d: float = 1.0,
) -> np.ndarray:
    return k_p * (goal[None, :] - q) - k_d * p
```

这个控制器不考虑 agent-agent 碰撞，只用于验证动力学和可视化。

---

## 11. 邻近图构建

后续 flocking 需要邻居集合：

\[
N_i=\{j:\|q_j-q_i\|<r,\ i\ne j\}
\]

Layer 0 可以先实现距离矩阵版本。

### 11.1 距离矩阵

```python
def pairwise_distances(q: np.ndarray) -> np.ndarray:
    diff = q[:, None, :] - q[None, :, :]
    return np.linalg.norm(diff, axis=-1)
```

### 11.2 邻接矩阵

```python
def adjacency_matrix(q: np.ndarray, r: float) -> np.ndarray:
    D = pairwise_distances(q)
    A = (D < r).astype(float)
    np.fill_diagonal(A, 0.0)
    return A
```

### 11.3 邻居列表

```python
def neighbor_lists(q: np.ndarray, r: float) -> list[list[int]]:
    A = adjacency_matrix(q, r)
    return [list(np.where(A[i] > 0)[0]) for i in range(q.shape[0])]
```

### 11.4 KDTree 版本

当 \(N\) 较大时，可以用 KDTree：

```python
from scipy.spatial import KDTree

def neighbor_lists_kdtree(q: np.ndarray, r: float) -> list[list[int]]:
    tree = KDTree(q)
    all_neighbors = tree.query_ball_point(q, r)
    return [[j for j in neigh if j != i] for i, neigh in enumerate(all_neighbors)]
```

Layer 0 用距离矩阵即可。\(N\leq 100\) 时完全够用。

---

## 12. 基础指标记录

Layer 0 至少实现以下指标。

### 12.1 平均速度

\[
\bar p(t)=\frac{1}{N}\sum_i p_i(t)
\]

```python
def mean_velocity(p: np.ndarray) -> np.ndarray:
    return np.mean(p, axis=0)
```

### 12.2 速度一致性误差

\[
E_v(t)=\frac{1}{N}\sum_i \|p_i-\bar p\|
\]

```python
def velocity_consensus_error(p: np.ndarray) -> float:
    p_bar = np.mean(p, axis=0, keepdims=True)
    return float(np.mean(np.linalg.norm(p - p_bar, axis=1)))
```

### 12.3 最小 agent-agent 距离

\[
d_{aa}^{min}(t)=\min_{i\ne j}\|q_i-q_j\|
\]

```python
def min_agent_distance(q: np.ndarray) -> float:
    D = pairwise_distances(q)
    np.fill_diagonal(D, np.inf)
    return float(np.min(D))
```

### 12.4 平均目标距离

\[
D_g(t)=\frac{1}{N}\sum_i \|q_i-q_g\|
\]

```python
def mean_goal_distance(q: np.ndarray, goal: np.ndarray) -> float:
    return float(np.mean(np.linalg.norm(q - goal[None, :], axis=1)))
```

### 12.5 图连通分量数量

```python
import networkx as nx

def num_connected_components(q: np.ndarray, r: float) -> int:
    A = adjacency_matrix(q, r)
    G = nx.from_numpy_array(A)
    return nx.number_connected_components(G)
```

### 12.6 代数连通度 \(\lambda_2\)

图拉普拉斯：

\[
L=D-A
\]

第二小特征值：

\[
\lambda_2(L)
\]

```python
def algebraic_connectivity(q: np.ndarray, r: float) -> float:
    A = adjacency_matrix(q, r)
    degree = np.diag(A.sum(axis=1))
    L = degree - A
    eigvals = np.linalg.eigvalsh(L)
    if len(eigvals) < 2:
        return 0.0
    return float(max(eigvals[1], 0.0))
```

注意：如果图不连通，\(\lambda_2=0\)。数值上可能出现很小的负数，可以做裁剪。

---

## 13. 可视化设计

Layer 0 至少要有三类可视化。

### 13.1 轨迹图

保存所有 agent 的历史位置：

```python
traj = []  # each element shape [N, 2]
```

仿真结束后画轨迹：

```python
def plot_trajectories(traj, goal=None, obstacles=None, save_path=None):
    import matplotlib.pyplot as plt

    traj = np.array(traj)  # [T, N, 2]
    plt.figure(figsize=(8, 5))

    for i in range(traj.shape[1]):
        plt.plot(traj[:, i, 0], traj[:, i, 1], linewidth=1.0)
        plt.scatter(traj[0, i, 0], traj[0, i, 1], s=10, marker="o")
        plt.scatter(traj[-1, i, 0], traj[-1, i, 1], s=10, marker="x")

    if goal is not None:
        plt.scatter(goal[0], goal[1], s=100, marker="*", label="Goal")

    if obstacles is not None:
        ax = plt.gca()
        for obs in obstacles:
            circle = plt.Circle(obs.center, obs.radius, fill=False, linewidth=2)
            ax.add_patch(circle)

    plt.axis("equal")
    plt.grid(True)
    plt.legend()

    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()
```

### 13.2 动画

可以用 `matplotlib.animation.FuncAnimation`。Layer 0 的动画中建议展示：

- agent 当前位置；
- agent 速度箭头；
- 目标点；
- 圆形障碍物；
- 世界边界。

### 13.3 指标曲线

例如：

```python
def plot_metrics(logs: dict, save_dir=None):
    import os
    import matplotlib.pyplot as plt

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    for key, values in logs.items():
        plt.figure(figsize=(6, 4))
        plt.plot(values)
        plt.title(key)
        plt.xlabel("step")
        plt.ylabel(key)
        plt.grid(True)

        if save_dir is not None:
            plt.savefig(f"{save_dir}/{key}.png", dpi=200, bbox_inches="tight")
        plt.show()
```

建议每个指标一张图，不要把太多曲线挤在一起。

---

## 14. Layer 0 主程序模板

下面是一个最小可运行主程序逻辑。

```python
import numpy as np
from simulator import FlockingEnv
from controllers import goal_pd_control
from metrics import (
    velocity_consensus_error,
    min_agent_distance,
    mean_goal_distance,
)

def main():
    env = FlockingEnv(
        n_agents=30,
        dt=0.02,
        world_size=(20.0, 12.0),
        v_max=3.0,
        u_max=8.0,
        seed=0,
    )

    state = env.reset(init_mode="random_left")
    goal = np.array([18.0, 6.0])

    n_steps = 1000
    traj = []
    logs = {
        "velocity_consensus_error": [],
        "min_agent_distance": [],
        "mean_goal_distance": [],
    }

    for step in range(n_steps):
        q = state["q"]
        p = state["p"]

        u = goal_pd_control(q, p, goal, k_p=0.8, k_d=1.2)

        state = env.step(u)

        traj.append(state["q"].copy())
        logs["velocity_consensus_error"].append(velocity_consensus_error(state["p"]))
        logs["min_agent_distance"].append(min_agent_distance(state["q"]))
        logs["mean_goal_distance"].append(mean_goal_distance(state["q"], goal))

    print("Simulation finished.")
    print("Final mean goal distance:", logs["mean_goal_distance"][-1])
    print("Final min agent distance:", logs["min_agent_distance"][-1])

if __name__ == "__main__":
    main()
```

这个程序只做目标导航，不做 flocking。它的作用是验证 Layer 0 环境、状态更新、日志记录和可视化流程是否正常。

---

## 15. Layer 0 验证 checklist

完成 Layer 0 后，应该逐项检查：

### 15.1 状态维度

- `q.shape == (N, 2)`
- `p.shape == (N, 2)`
- `u.shape == (N, 2)`

### 15.2 数值稳定性

- 没有 `NaN`；
- 速度不超过 `v_max`；
- 加速度不超过 `u_max`；
- 位置没有跑出世界边界太多。

### 15.3 零控制测试

当 \(u=0\) 时：

- 如果初始速度为 0，agent 应该静止；
- 如果初始速度不为 0，agent 应该匀速运动。

### 15.4 目标导航测试

使用 PD goal control 时：

- agent 应该整体朝目标点移动；
- 平均目标距离应该下降；
- 如果 damping 合理，agent 不应该在目标附近剧烈震荡。

### 15.5 邻近图测试

给定感知半径 \(r\)：

- 邻接矩阵对角线为 0；
- 邻接矩阵对称；
- 距离小于 \(r\) 的 agent 成为邻居；
- 连通分量数量符合直觉。

### 15.6 可视化测试

- 轨迹图能正常保存；
- agent 起点、终点能区分；
- 目标点能显示；
- 障碍物能显示；
- 动画能正常播放或保存。

---

## 16. 推荐参数初值

Layer 0 可以从下面参数开始：

```yaml
n_agents: 30
dt: 0.02
n_steps: 1000
world_size: [20.0, 12.0]
v_max: 3.0
u_max: 8.0
seed: 0

goal: [18.0, 6.0]

goal_control:
  k_p: 0.8
  k_d: 1.2

neighbor_radius: 3.0
agent_radius: 0.12
```

这些参数只是初始值，后续加入 flocking 和障碍物后需要重新调参。

---

## 17. 后续 Layer 的接口预留

Layer 0 最重要的是让接口干净。后续控制器应该统一输入和输出：

```python
def controller(q, p, obstacles, goal, params) -> np.ndarray:
    # Args:
    #   q: positions, shape [N, 2]
    #   p: velocities, shape [N, 2]
    #   obstacles: list of CircleObstacle
    #   goal: target position, shape [2]
    #   params: dict
    #
    # Returns:
    #   u: control input, shape [N, 2]
    pass
```

后续可以逐步叠加：

```python
u_alpha = alpha_flocking_control(q, p, params)
u_gamma = gamma_navigation_control(q, p, goal, params)
u_beta = beta_obstacle_control(q, p, obstacles, params)
u_dyn = dynamic_iapf_control(q, p, obstacles, goal, params)

u = u_alpha + u_gamma + u_beta + u_dyn
```

这样做 ablation 非常方便：

```python
# Only migration
u = u_gamma

# Free flocking with migration
u = u_alpha + u_gamma

# Static obstacle avoidance
u = u_alpha + u_beta + u_gamma

# Dynamic obstacle extension
u = u_alpha + u_beta + u_gamma + u_dyn
```

---

## 18. 第一周建议实现目标

第一周不要实现完整 Olfati-Saber。建议完成：

1. `FlockingEnv`；
2. `CircleObstacle`；
3. `zero_control`；
4. `goal_pd_control`；
5. `pairwise_distances`；
6. `adjacency_matrix`；
7. `velocity_consensus_error`；
8. `min_agent_distance`；
9. `mean_goal_distance`；
10. 轨迹图；
11. 简单动画；
12. 保存实验日志。

完成后应该能跑出：

- 随机 agent 朝目标移动的动画；
- agent 轨迹图；
- 平均目标距离下降曲线；
- 最小 agent-agent 距离曲线；
- 速度一致性误差曲线。

这时 Layer 0 就算完成，可以进入 Layer 1：Olfati-Saber α-agent free flocking。

---

## 19. 与后续 Layer 的关系

Layer 0 完成后，后续实现路线为：

| Layer | 内容 | 依赖 Layer 0 的部分 |
|---|---|---|
| Layer 1 | α-agent free flocking | 邻近图、状态更新、指标记录 |
| Layer 2 | γ-agent navigation | 目标点、PD tracking、轨迹图 |
| Layer 3 | β-agent static obstacle avoidance | 圆形障碍物、最小障碍距离、动画 |
| Layer 4 | dynamic IAPF | 动态障碍物、相对速度、避障指标 |

Layer 0 的代码质量会直接影响后续调试效率。建议先把环境写得简单、稳定、可视化清楚，再加入复杂控制律。

---

## 20. 最终建议

Layer 0 的核心原则是：

1. 不使用复杂框架；
2. 用 `numpy` 清晰实现状态更新；
3. 把环境和控制器分开；
4. 每一步都记录指标；
5. 每个新模块都先单独测试；
6. 可视化优先，先看现象，再调公式。

如果 Layer 0 实现清楚，后续 α-agent、β-agent 和 dynamic IAPF 都只是往 `compute_control()` 里增加控制项。
