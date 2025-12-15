# quantum-testing

面向 NISQ 芯片拓扑的量子编译器（布局/路由阶段）**效率评估**蜕变测试（Metamorphic Testing, MT）实验仓库。  
核心对比指标包括：**SWAP 开销、双比特门数（2Q gate count）、电路深度（depth）、编译时间（time）**。


---

## 1. 项目概述（Overview）

本仓库提供两套蜕变测试流程，用于评估不同路由器/编译策略在映射到具体芯片耦合图时引入的额外开销：

- **方法一（MT-1：按芯片拓扑组织的对比流程）**  
  主目录下以芯片命名的三个文件夹分别对应不同耦合图：`Aspen-4/`、`Sydney/`、`Tokyo/`。  
  你可以在每个芯片目录内运行数据集生成脚本，然后运行对应的路由测试脚本；当前仓库提供 **SABRE** 与 **QMAP** 的测试代码入口。

- **方法二（MT-2：组合/插件式测试流程）**  
  统一放在 `test2/` 中：先生成随机电路，再生成动态/组合后的测试集，最后运行不同路由器的测试脚本（当前提供 heuristic、sabre、fidls 三类入口）。

关于 **FiDLS**：本仓库不直接复刻 FiDLS 的完整执行管线，建议按 FiDLS 官方仓库的流程运行，并将数据集与芯片图替换为本项目的实验设置：  
- https://github.com/ebony72/FiDLS

---

## 2. 目录结构（Repository Layout）

- `Aspen-4/`：方法一（MT-1）在 Aspen-4 拓扑上的实验（含 SABRE/QMAP 测试入口）
- `Sydney/`：方法一（MT-1）在 Sydney 拓扑上的实验（含 SABRE/QMAP 测试入口）
- `Tokyo/`：方法一（MT-1）在 Tokyo 拓扑上的实验（含 SABRE/QMAP 测试入口）
- `test2/`：方法二（MT-2）整体代码（随机电路 + 动态/组合电路 + 路由测试）
- `requirements.txt`：Python 依赖（版本已固定，建议按该文件安装）

---

## 3. 环境与依赖（Requirements）

- Python：建议 3.10+（3.9+ 也可）
- 依赖安装：见下文
- 主要依赖（摘录）：Qiskit、mqt.qmap、networkx、numpy/pandas、matplotlib 等（详见 `requirements.txt`）

---

## 4. 安装（Installation）

推荐使用虚拟环境：

```bash
git clone https://github.com/hengbinyang166/quantum-testing.git
cd quantum-testing

python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate

pip install -r requirements.txt

```


5.1 方法一（MT-1）：以某个芯片目录为例运行

在任一芯片目录中（例如 Aspen-4/ 或 Sydney/ / Tokyo/），按如下顺序运行：

生成随机电路数据集
```bash

python generate_random_circuits.py
```

运行路由测试（示例入口）

SABRE 相关对比（示例脚本）：
```bash

python test1_compare_sabre_isomorphism.py
```

QMAP 路由（示例脚本）：
```bash

python test1_qmap.py
```

如需测试其他路由器
你可以在现有测试脚本中替换“路由/编译”模块为目标算法的调用逻辑，并保持输出指标一致，即可纳入对比。

FiDLS：请参考 FiDLS 官方仓库流程运行，并替换为你的数据集与芯片耦合图。
