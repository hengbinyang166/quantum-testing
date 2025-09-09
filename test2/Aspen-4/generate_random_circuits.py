# generate_random_circuits.py
# ------------------------------------------------
# 脚本功能：在给定 qubit 数量范围内，生成固定10层的随机量子电路，并保存为 QASM 文件。
# 对于每个 qubit 数（5 至 10），生成10个样本，并将所有文件保存在名为“随机的10层”的文件夹中。

import os
import random
import shutil
import csv
from qiskit import QuantumCircuit
from qiskit.qasm2 import dumps


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
        random.seed()
        print("[RNG] Seeded from system entropy (non-reproducible)")


def build_random_circuit(n_qubits: int, n_layers: int) -> QuantumCircuit:
    """
    在 n_qubits 上生成一个随机量子电路，共 n_layers 层。
    每层随机添加单比特门或双比特 CX 门。
    返回 QuantumCircuit 对象。
    """
    qc = QuantumCircuit(n_qubits, name="random_circuit")

    for _ in range(n_layers):
        available = set(range(n_qubits))  # 当前可放门的 qubit 列表
        placed = 0  # 本层已放置单比特门的标志

        while available:
            # 50% 概率尝试放置双比特 CX 门（需至少 2 个 qubit 可用）
            if random.random() < 0.5 and len(available) >= 2:
                q1, q2 = random.sample(list(available), 2)  # 转换为列表
                qc.cx(q1, q2)
                available.remove(q1)
                available.remove(q2)
            else:
                # 否则放置单比特门，随机选择 x/y/z
                q = random.choice(list(available))
                gate = random.choice(['x', 'y', 'z'])
                getattr(qc, gate)(q)
                available.remove(q)
                placed += 1

        # 如果本层没有放置任何单比特门，则强制在随机 qubit 上插入一个 X 门
        if placed == 0 and n_qubits > 0:
            q = random.randint(0, n_qubits - 1)
            qc.x(q)

    return qc


def main():
    # 1. 初始化随机数生成器
    init_rng(reproducible=True, seed=42)

    # 2. 参数配置：qubit 数范围 5..10，固定层数 10，样本数 10
    qubit_min = 3
    qubit_max = 5
    fixed_layers = 10
    num_samples = 10

    # 3. 创建输出目录：名称为“随机的10层”
    out_dir = "quantum_dataset_random 10"
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # 4. 初始化 mapping.csv，用于记录文件名与 qubit 数映射
    mapping_path = os.path.join(out_dir, "mapping.csv")
    with open(mapping_path, "w", newline="") as csvf:
        writer = csv.writer(csvf)
        writer.writerow(["filename", "n_qubits"])  # CSV 表头

    # 5. 双层循环：n_qubits (5..10) -> idx (1..10)
    for n_qubits in range(qubit_min, qubit_max + 1):
        n_layers = fixed_layers
        for idx in range(1, num_samples + 1):
            # 5.1 生成随机电路
            qc = build_random_circuit(n_qubits, n_layers)

            # 5.2 构造文件名并写 QASM 文件
            fname = f"q{n_qubits}_l{n_layers}_s{idx}.qasm"
            path = os.path.join(out_dir, fname)
            with open(path, "w") as f:
                f.write(dumps(qc))

            # 5.3 在 mapping.csv 中记录映射关系
            with open(mapping_path, "a", newline="") as csvf:
                writer = csv.writer(csvf)
                writer.writerow([fname, n_qubits])

        # 每个 n_qubits 完成后打印提示
        print(f"Done: n_qubits={n_qubits}, n_layers={n_layers} → {num_samples} files")

    print(f"全部随机电路生成完毕，保存在文件夹：{out_dir}")


if __name__ == '__main__':
    main()