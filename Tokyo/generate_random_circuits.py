import os  # 用于文件和目录操作
import random  # 随机数生成
import shutil  # 目录删除
import csv  # CSV 文件读写
from qiskit import QuantumCircuit  # 构造量子电路
from qiskit.qasm2 import dumps  # 电路序列化为 QASM 格式


def init_rng(reproducible: bool = False, seed: int = 42):
    """
    初始化随机数生成器:
    - reproducible=False: 使用系统熵源播种，每次结果不同
    - reproducible=True: 使用固定种子，保证结果可重复
    """
    if reproducible:
        random.seed(seed)
        print(f"[RNG] Using fixed seed = {seed} (reproducible)")
    else:
        random.seed()  # 无参数时使用系统时间或操作系统熵源
        print("[RNG] Seeded from system entropy (non-reproducible)")


def build_random_circuit(n_qubits: int, n_layers: int) -> QuantumCircuit:
    """
    在 n_qubits 上生成一个随机量子电路，共 n_layers 层。
    每层随机添加单比特门或双比特 CX 门。
    移除 barrier 操作，保证电路简洁。
    返回 QuantumCircuit 对象。
    """
    qc = QuantumCircuit(n_qubits, name="random_circuit")

    # 遍历每一层
    for layer in range(n_layers):
        available = set(range(n_qubits))  # 当前可放门的 qubit 列表
        placed = 0  # 本层已放置门的数量

        # 直到所有 qubit 都被尝试放置
        while available:
            # 50% 概率放置双比特 CX 门（需至少 2 个 qubit 可用）
            if random.random() < 0.5 and len(available) >= 2:
                q1, q2 = random.sample(available, 2)
                qc.cx(q1, q2)  # 添加 CX 门
                # 使用过的 qubit 从 available 中移除
                available.remove(q1)
                available.remove(q2)
            else:
                # 否则放置单比特门，随机选择 x/y/z
                q = random.choice(list(available))
                gate = random.choice(['x', 'y', 'z'])
                getattr(qc, gate)(q)  # 动态调用 qc.x(q) 或 qc.y(q) 等
                available.remove(q)
                placed += 1

        # 如果本层未放置任何门，至少插入一个 X 门保证层非空
        if placed == 0 and n_qubits > 0:
            q = random.randint(0, n_qubits - 1)
            qc.x(q)

        # barrier 操作已移除：不再插入 qc.barrier()

    return qc


if __name__ == "__main__":
    # —— 参数配置 —— #
    init_rng(reproducible=True, seed=42)  # 设置为 True 可重复

    max_qubits = 20      # 最大 qubit 数量
    max_layers = 50      # 最大层数
    num_samples = 20     # 每种 qubit+层数组合生成的电路数

    out_dir = "quantum_dataset_random"

    # —— 创建或清空输出目录 —— #
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)  # 删除已有目录及内容
    os.makedirs(out_dir, exist_ok=True)  # 新建目录

    # —— 初始化 mapping.csv，用于记录文件名与 qubit 数量映射 —— #
    mapping_path = os.path.join(out_dir, "mapping.csv")
    with open(mapping_path, "w", newline="") as csvf:
        writer = csv.writer(csvf)
        writer.writerow(["filename", "n_qubits"])  # CSV 表头

    # —— 生成随机量子电路 —— #
    for n_qubits in range(1, max_qubits + 1):
        for n_layers in range(1, max_layers + 1):
            for idx in range(1, num_samples + 1):
                # 构建电路
                qc = build_random_circuit(n_qubits, n_layers)

                # 构造文件名，如 q3_l10_s1.qasm
                fname = f"q{n_qubits}_l{n_layers}_s{idx}.qasm"
                path = os.path.join(out_dir, fname)

                # 写入 QASM 文件
                with open(path, "w") as f:
                    f.write(dumps(qc))

                # 在 mapping.csv 中追加对应关系
                with open(mapping_path, "a", newline="") as csvf:
                    writer = csv.writer(csvf)
                    writer.writerow([fname, n_qubits])

            # 每完成一个 qubit+层数组合，输出进度
            print(f"Done: n_qubits={n_qubits}, n_layers={n_layers} → {num_samples} files")

    print("全部随机电路生成完毕，保存在", out_dir)