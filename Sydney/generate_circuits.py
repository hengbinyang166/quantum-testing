# generate_circuits_remap.py
# -----------------------------------
# 脚本功能：在随机选择的连通子图上生成量子电路，
#           并将 qubit 重映射为连续编号 0..k-1 写入 QASM，
#           同时在 mapping.csv 中保留原始物理节点索引。

import os
import random
import shutil
import csv
import networkx as nx
from qiskit import QuantumCircuit
from qiskit.qasm2 import dumps


def init_rng(reproducible: bool = False, seed: int = 42):
    """
    初始化随机数生成器。
    :param reproducible: 是否使用固定种子以保证可复现。
    :param seed: 可复现模式下使用的种子值。
    """
    if reproducible:
        random.seed(seed)
        print(f"[RNG] Using fixed seed = {seed} (reproducible)")
    else:
        random.seed()  # 系统熵源播种
        print("[RNG] Seeded from system entropy (non-reproducible)")


def sample_connected_subgraph(G: nx.Graph, k: int, max_attempts=100) -> list:
    """
    从图 G 中随机采样一个包含 k 个节点的连通子图。
    :param G: 输入的 networkx 图（全芯片拓扑）。
    :param k: 期望的子图节点数。
    :param max_attempts: 最多重试次数。
    :return: 采样到的节点列表（使用原始标签）。
    """
    nodes = list(G.nodes())
    for _ in range(max_attempts):
        start = random.choice(nodes)    # 随机选一个起点
        visited = {start}
        frontier = [start]
        # 广度优先扩展直到收集到 k 个节点或前沿耗尽
        while frontier and len(visited) < k:
            cur = frontier.pop(0)
            for nbr in G.neighbors(cur):
                if nbr not in visited:
                    visited.add(nbr)
                    frontier.append(nbr)
                if len(visited) == k:
                    break
        if len(visited) == k:
            return list(visited)
    # 若尝试多次仍失败，则抛出异常
    raise RuntimeError(f"无法从图中采样到 {k} 个连通节点。")


def build_subgraph_random_circuit(G: nx.Graph, sub_nodes: list, n_layers: int) -> tuple:
    """
    生成两种版本的量子电路：
    1. 重映射版本(0..k-1)
    2. 原始物理节点版本
    """
    k = len(sub_nodes)
    # 重映射版本
    qc_remap = QuantumCircuit(k)
    # 原始编号版本(使用全芯片量子比特数)
    qc_original = QuantumCircuit(len(G.nodes()))
    
    mapping = {orig: idx for idx, orig in enumerate(sub_nodes)}
    allowed_edges = list(G.subgraph(sub_nodes).edges())

    for _ in range(n_layers):
        max_gates = k
        num_gates = random.randint(1, max_gates)
        available = set(sub_nodes)
        placed = 0

        for _ in range(num_gates):
            if not available:
                break

            if random.random() < 0.5 and len(available) >= 2 and allowed_edges:
                valids = [(u, v) for u, v in allowed_edges if u in available and v in available]
                if valids:
                    u, v = random.choice(valids)
                    # 在重映射版本上应用
                    qc_remap.cx(mapping[u], mapping[v])
                    # 在原始版本上应用
                    qc_original.cx(u, v)
                    available.remove(u)
                    available.remove(v)
                    continue

            u = random.choice(list(available))
            gate = random.choice(['x', 'y', 'z'])
            # 在重映射版本上应用
            getattr(qc_remap, gate)(mapping[u])
            # 在原始版本上应用
            getattr(qc_original, gate)(u)
            available.remove(u)
            placed += 1

        if placed == 0 and sub_nodes:
            u = random.choice(sub_nodes)
            qc_remap.x(mapping[u])
            qc_original.x(u)

    return qc_remap, qc_original


def main():
    # —— 1. 读取芯片拓扑 & 初始化 —— #
    init_rng(reproducible=True, seed=42)

    # 2. 从文件读取芯片拓扑（修改部分）
    if os.path.exists("chip_topology.edgelist"):
        G = nx.read_edgelist("chip_topology.edgelist", nodetype=int)
        print("[INFO] 从chip_topology.edgelist加载芯片拓扑")
    else:
        # 如果文件不存在则创建默认拓扑
        G = nx.erdos_renyi_graph(27, 0.15)
        nx.write_edgelist(G, "chip_topology.edgelist", data=False)
        print("[INFO] 生成新芯片拓扑并保存到chip_topology.edgelist")

    # 3. 初始化参数
    N = G.number_of_nodes()
    max_layers = 50
    num_samples = 20
    
    # 4. 创建输出目录
    out_dir_remap = "quantum_dataset_remap"
    out_dir_original = "quantum_dataset_original"
    os.makedirs(out_dir_remap, exist_ok=True)
    os.makedirs(out_dir_original, exist_ok=True)

    mapping_path_remap = os.path.join(out_dir_remap, "mapping.csv")
    mapping_path_original = os.path.join(out_dir_original, "mapping.csv")

    for k in range(1, N+1):
        for n_layers in range(1, max_layers+1):
            for idx in range(1, num_samples+1):
                sub_nodes = sample_connected_subgraph(G, k)
                
                # 生成两种版本的电路
                qc_remap, qc_original = build_subgraph_random_circuit(G, sub_nodes, n_layers)
                
                # 保存两种版本的电路
                fname = f"q{k}_l{n_layers}_s{idx}.qasm"
                
                # 保存重映射版本
                with open(os.path.join(out_dir_remap, fname), "w") as f:
                    f.write(dumps(qc_remap))
                with open(mapping_path_remap, "a", newline="") as csvf:
                    writer = csv.writer(csvf)
                    writer.writerow([fname, sub_nodes])
                
                # 保存原始编号版本
                with open(os.path.join(out_dir_original, fname), "w") as f:
                    f.write(dumps(qc_original))
                with open(mapping_path_original, "a", newline="") as csvf:
                    writer = csv.writer(csvf)
                    writer.writerow([fname, sub_nodes])
    
        print(f"Done: q={k}, layers={n_layers} → {num_samples} files")
    
    print(f"完成电路生成：\n- 重映射版本保存在 {out_dir_remap}\n- 原始编号版本保存在 {out_dir_original}")

if __name__ == '__main__':
    main()
