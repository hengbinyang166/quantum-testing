# -*- coding: utf-8 -*-
"""
==================================================
目标：
- 每个插件层插入的都是“原始小电路（random 10 层）”，不包含 QMAP 产生的 swap；
- QMAP-exact 仅用于求：
    1) 初始布局 L: 逻辑比特 j -> 本地物理索引 ℓ（0..q_large-1）
    2) 本地 SWAP 序列（按出现顺序的 (a,b) 列表，a/b 为本地索引）
- 不把 swap 写进电路；而是将 swap 的效果吸收到 `current_active` 的置换中；
- 15 层随机段（前/中/后）均在“当时的 current_active 顺序”上生成；
- 最终导出：
    - 物理索引版（全芯片宽度 N）
    - 重索引紧致版（仅活跃区宽度 q_large；重排为 0..q_large-1，默认按“初始活跃区顺序”）
"""

import os
import json
import random
from typing import List, Tuple, Dict, Any

import networkx as nx
from qiskit import QuantumCircuit
from qiskit.qasm2 import dumps as qasm2_dumps
from qiskit.circuit.library import CXGate, HGate, XGate, RZGate
from qiskit.providers.fake_provider import GenericBackendV2
from mqt import qmap

# ===================== 配置区（可按需修改） =====================

RANDOM_CIRCUIT_DIR = 'quantum_dataset_random 10'   # 随机小电路目录（你已生成）
EDGE_LIST_PATH     = 'chip_topology.edgelist'      # 物理芯片边表（如 Aspen-4）

NUM_INSTANCES_PER_SIZE = 10                        # 每个 q_large 生成多少个实例
MIN_LARGE_QUBITS = 3                               # 活跃区最小大小
MAX_LARGE_QUBITS = 16                               # 活跃区最大大小

FRONT_LAYERS   = 15    # 前 15 层
MID_LAYERS     = 15    # 插件之间的 15 层
PLUGIN_LAYERS  = 10    # 插件小电路固定 10 层（与随机数据集一致）
N_PLUGINS      = 3     # 三个中间层

# 输出目录
OUTPUT_DIR_PHYSICAL  = 'combined3_circuits_physical'    # 大电路（物理索引版）
OUTPUT_DIR_REINDEXED = 'combined3_circuits_reindexed'   # 大电路（重索引紧致版）
QMAP_LOCAL_QASM_DIR  = 'qmap_exact_debug_local'          # 调试：保存每个插件层的“本地路由电路”（可选）
STATS_JSON           = 'dynamic_compilation_stats_3plugins_orig.json'

random.seed(42)  # 固定随机种子，保证复现


# ===================== 工具函数 =====================

def load_chip_graph(path: str) -> nx.Graph:
    """读取芯片边表，返回无向图（点为物理比特编号）。"""
    return nx.read_edgelist(path, nodetype=int)

def sample_connected_k_nodes(G: nx.Graph, k: int, attempts: int = 200) -> List[int]:
    """在图 G 中随机采样一个大小为 k 的连通子图（返回物理比特顺序列表）。"""
    nodes = list(G.nodes())
    for _ in range(attempts):
        start = random.choice(nodes)
        sub = {start}
        frontier = list(G.neighbors(start))
        random.shuffle(frontier)
        while frontier and len(sub) < k:
            v = frontier.pop()
            if v not in sub:
                sub.add(v)
                for nb in G.neighbors(v):
                    if nb not in sub and nb not in frontier:
                        frontier.append(nb)
        if len(sub) == k and nx.is_connected(G.subgraph(sub)):
            return list(sub)
    raise RuntimeError(f"采样连通子图失败：重试 {attempts} 次仍未找到大小为 {k} 的连通子图")

def local_backend_for_nodes(G: nx.Graph, nodes: List[int]) -> GenericBackendV2:
    """为给定活跃区节点构造本地后端（节点重编号为 0..k-1）。"""
    idx = {n:i for i,n in enumerate(nodes)}
    cmap = [(idx[u], idx[v]) for (u,v) in G.subgraph(nodes).edges()]
    return GenericBackendV2(num_qubits=len(nodes), coupling_map=cmap)

def read_random_small_circuit(max_qubits: int) -> Tuple[QuantumCircuit, int, str]:
    """从随机数据集中挑选一个 qubits <= max_qubits 且层数=10 的小电路。"""
    files = [f for f in os.listdir(RANDOM_CIRCUIT_DIR) if f.endswith('.qasm')]
    candidates = []
    for f in files:
        name = os.path.splitext(f)[0]  # 形如 q{n}_l{L}_s{idx}
        try:
            parts = name.split('_')
            n = int(parts[0][1:])
            L = int(parts[1][1:])
        except Exception:
            continue
        if L == PLUGIN_LAYERS and n <= max_qubits:
            candidates.append((f, n))
    if not candidates:
        raise FileNotFoundError(
            f"未在 {RANDOM_CIRCUIT_DIR} 找到“层数=10 且 qubits<= {max_qubits}”的小电路 .qasm"
        )
    fname, n = random.choice(candidates)
    qc = QuantumCircuit.from_qasm_file(os.path.join(RANDOM_CIRCUIT_DIR, fname))
    return qc, n, fname

def generate_partial(G: nx.Graph, num_chip_qubits: int, active_nodes: List[int], layers: int) -> QuantumCircuit:
    """在给定活跃区上生成 `layers` 层随机电路（1q 随机 + 活跃区边上的若干 CX）。"""
    qc = QuantumCircuit(num_chip_qubits)
    edges = list(G.subgraph(active_nodes).edges())
    for _ in range(layers):
        # 1 比特门层
        for qb in active_nodes:
            r = random.random()
            if r < 0.33: qc.append(HGate(), [qb])
            elif r < 0.66: qc.append(XGate(), [qb])
            else: qc.append(RZGate(random.random()*3.14159), [qb])
        # 2 比特门层：从活跃区边中选若干条做 CX
        random.shuffle(edges)
        k = max(1, len(active_nodes)//2)   # 简单控制密度
        for (u,v) in edges[:k]:
            qc.append(CXGate(), [u,v])
    return qc

def apply_local_swaps(order: List[int], local_swaps: List[Tuple[int,int]]) -> List[int]:
    """按本地 SWAP 序列对“活跃区物理顺序列表”执行置换，返回更新后的顺序。"""
    order = order[:]
    for a,b in local_swaps:
        order[a], order[b] = order[b], order[a]
    return order

# ---------- 关键：从 qmap.compile 返回的 mapping_results 中“提取初始布局 L” ----------

def _as_list_from_mapping(mapping_like: Any) -> List[Tuple[int,int]]:
    """
    把各种“可能形式”的映射对象尽量解析为 [(logical, physical_local), ...] 的列表。
    兼容几种常见写法：dict、list/tuple of pair、对象带 items()/keys() 等。
    """
    pairs = []
    if mapping_like is None:
        return pairs
    # dict: {logical: physical}
    if isinstance(mapping_like, dict):
        for k, v in mapping_like.items():
            try:
                lk = int(k)
                lv = int(v)
                pairs.append((lk, lv))
            except Exception:
                continue
        return pairs
    # list/tuple of pair
    if isinstance(mapping_like, (list, tuple)):
        for it in mapping_like:
            if isinstance(it, (list, tuple)) and len(it) == 2:
                try:
                    lk = int(it[0]); lv = int(it[1])
                    pairs.append((lk, lv))
                except Exception:
                    continue
        if pairs:
            return pairs
    # 带 items()
    if hasattr(mapping_like, 'items'):
        try:
            for k, v in mapping_like.items():
                lk = int(k); lv = int(v)
                pairs.append((lk, lv))
            return pairs
        except Exception:
            pass
    # 其他对象：尽力解析（常见：可能有 .logical_to_physical / .initial_layout 等）
    return pairs

def extract_initial_layout(mapping_results: Any, n_small: int, k_local: int) -> Tuple[List[int], str]:
    """
    从 mapping_results 中尽可能提取“初始布局 L（长度 n_small 的列表，元素在 0..k_local-1）”。
    返回：(L, how) 其中 how 描述从哪个字段提取的（便于打印溯源）。
    若未能提到，回退为 L = [0,1,...,n_small-1] 并标注 'fallback_identity'。
    """
    candidates = []
    # 常见字段名穷举（不同版本命名可能不同）：
    for name in [
        'initial_layout', 'initial_mapping', 'initial_layout_dict',
        'initial_qubit_map', 'init_qubit_map', 'init_layout',
        'layout_initial', 'layout_init'
    ]:
        if hasattr(mapping_results, name):
            candidates.append((name, getattr(mapping_results, name)))

    # 某些版本可能把“初始布局”放在更深层：mapping_results.layout 或 mapping_results.layouts 等
    for name in ['layout', 'layouts', 'info', 'data']:
        if hasattr(mapping_results, name):
            obj = getattr(mapping_results, name)
            for sub in [
                'initial_layout', 'initial_mapping', 'initial_layout_dict',
                'initial_qubit_map', 'init_qubit_map', 'init_layout'
            ]:
                if hasattr(obj, sub):
                    candidates.append((f'{name}.{sub}', getattr(obj, sub)))

    # 逐个候选尝试解析
    for tag, obj in candidates:
        pairs = _as_list_from_mapping(obj)
        if pairs:
            # 组装 L
            L = [None] * n_small
            ok = True
            for (lg, ph) in pairs:
                if 0 <= lg < n_small and 0 <= ph < k_local:
                    if L[lg] is None:
                        L[lg] = ph
            # 用“自然补全”的办法填空（防御式，正常不该出现空位）
            next_phys = 0
            for j in range(n_small):
                if L[j] is None:
                    # 找到未占用的本地物理索引
                    while next_phys in L:
                        next_phys += 1
                    if next_phys >= k_local:
                        ok = False
                        break
                    L[j] = next_phys
            if ok:
                return L, f'from:{tag}'

    # 兜底：如果 mapping_results 里没有我们能识别的字段，就用身份映射
    L = list(range(n_small))
    return L, 'fallback_identity'

# ---------- QMAP exact：仅“取 L + SWAP 序列”，不使用路由后电路 ----------

def compile_for_layout_and_swaps(qc_small: QuantumCircuit,
                                 G: nx.Graph,
                                 active_nodes: List[int]) -> Tuple[List[int], List[Tuple[int,int]], str, QuantumCircuit]:
    """
    在“当前活跃区”上为小电路做 exact 编译，但仅提取：
      - 初始布局 L（逻辑 -> 本地物理索引）
      - 本地 SWAP 序列（按出现顺序的 (a,b)）
    另外返回 how（说明 L 的来源）和 mapped_local（仅用于调试保存，不参与拼接）。
    """
    backend = local_backend_for_nodes(G, active_nodes)

    # qmap.compile 返回 (mapped_local, mapping_results)
    mapped_local, mapping_results = qmap.compile(
        qc_small,
        arch=backend,
        method="exact"
    )

    # 1) 提取本地 SWAP 序列（用新 API，避免旧式解包）
    local_index = {q: i for i, q in enumerate(mapped_local.qubits)}
    local_swaps = []
    for inst in mapped_local.data:
        op = inst.operation
        if op.name == "swap":
            a = local_index[inst.qubits[0]]
            b = local_index[inst.qubits[1]]
            local_swaps.append((a, b))

    # 2) 提取初始布局 L（兼容不同版本的字段名；失败则回退 identity）
    L, how = extract_initial_layout(mapping_results, n_small=len(qc_small.qubits), k_local=len(active_nodes))
    return L, local_swaps, how, mapped_local

# ---------- 把“原始小电路”按 L 和 current_active 抬升到“全芯片物理索引空间” ----------

def lift_original_by_layout(qc_small: QuantumCircuit,
                            L: List[int],
                            current_active: List[int],
                            num_chip_qubits: int) -> QuantumCircuit:
    """
    将原始小电路（逻辑位 j）映到“全芯片物理位 current_active[L[j]]”上，返回全芯片宽度的一段电路。
    注意：此处不涉及 swap；L 是“逻辑 -> 本地物理索引”，current_active 是“本地物理索引 -> 物理比特号”。
    """
    qc_full = QuantumCircuit(num_chip_qubits)
    small_index = {qobj: i for i, qobj in enumerate(qc_small.qubits)}

    for inst in qc_small.data:
        op = inst.operation
        # 只应有量子门（random 生成的小电路无测量/经典位）
        mapped_qargs = []
        for qobj in inst.qubits:
            j_logical = small_index[qobj]         # 逻辑索引 j
            local_ph = L[j_logical]               # 本地物理索引 ℓ
            phys = current_active[local_ph]       # 全芯片物理比特号
            mapped_qargs.append(qc_full.qubits[phys])
        qc_full.append(op, mapped_qargs, [])
    return qc_full

# ---------- 重索引（紧致化）为 0..(q_large-1) ----------

def reindex_compact(qc: QuantumCircuit, phys_nodes_in_order: List[int]) -> QuantumCircuit:
    """
    按给定的物理比特次序做紧致重索引（仅量子门，不处理经典位）。
    例如 phys_nodes_in_order=[6,9,10,13] -> 映射为 6->0, 9->1, 10->2, 13->3。
    """
    phys_to_compact = {p: i for i, p in enumerate(phys_nodes_in_order)}
    src_obj_to_phys = {qobj: i for i, qobj in enumerate(qc.qubits)}

    rq = QuantumCircuit(len(phys_nodes_in_order))
    for inst in qc.data:
        op = inst.operation
        compact_qargs = []
        for qobj in inst.qubits:
            phys = src_obj_to_phys[qobj]           # 源电路中的“物理位编号”
            comp = phys_to_compact[phys]           # 紧致位编号
            compact_qargs.append(rq.qubits[comp])
        rq.append(op, compact_qargs, [])
    return rq


# ===================== 构建一个实例（核心流程） =====================

def build_instance(G: nx.Graph, q_large: int, sidx: int) -> dict:
    """
    对于给定活跃区大小 q_large 和实例编号 sidx，构建“前15 → 插件1 → 15 → 插件2 → 15 → 插件3 → 15”的大电路。
    插件层插入“原始小电路（按 L 映射到 current_active）”；SWAP 只用于更新 current_active。
    """
    num_chip_qubits = len(G.nodes())
    active0 = sample_connected_k_nodes(G, q_large)   # 初始活跃区
    print(f"\n=== 实例 q{q_large}_s{sidx} ===")
    print(f"[步骤1] 选取活跃区（物理比特顺序）：{active0}")

    segments = []   # 各段电路
    stats = {
        "instance": f"q{q_large}_s{sidx}",
        "q_large": q_large,
        "active_start": active0,
        "plugins": []
    }

    # 前 15 层
    seg_front = generate_partial(G, num_chip_qubits, active0, FRONT_LAYERS)
    segments.append(seg_front)
    print(f"[步骤2] 生成前 {FRONT_LAYERS} 层随机电路（仅在活跃区上放门）")

    current_active = active0[:]  # 当前活跃区（随 SWAP 更新）

    for p in range(1, N_PLUGINS+1):
        print(f"\n[插件层 {p}] 当前活跃区顺序（物理比特）：{current_active}")

        # 选择一个小电路（逻辑比特数 <= q_large，层数=10）
        qc_small, n_small, fname = read_random_small_circuit(q_large)
        print(f"[插件层 {p}] 选择小电路：{fname}（逻辑比特数={n_small}, 层数={PLUGIN_LAYERS}）")

        # 在 current_active 上 exact：只取 L + 本地 SWAP 序列
        L, local_swaps, how, mapped_local = compile_for_layout_and_swaps(qc_small, G, current_active)

        # （可选）保存“本地路由电路”供调试，但不用于拼接
        os.makedirs(QMAP_LOCAL_QASM_DIR, exist_ok=True)
        local_path = os.path.join(QMAP_LOCAL_QASM_DIR, f"q{q_large}_s{sidx}_p{p}_local.qasm")
        with open(local_path, 'w') as f:
            f.write(qasm2_dumps(mapped_local))
        print(f"[插件层 {p}] 本地路由（仅用于调试）已保存：{local_path}")
        print(f"[插件层 {p}] 初始布局 L 来源：{how}，L={L}")
        print(f"[插件层 {p}] 本地 SWAP 序列（按出现顺序，本地索引）: {local_swaps}")

        # 打印“映射为物理”的 SWAP，便于核对
        phys_swaps = [(current_active[a], current_active[b]) for (a,b) in local_swaps]
        print(f"[插件层 {p}] 映射到物理比特的 SWAP 序列：{phys_swaps}")

        # 插入“原始小电路” ： 逻辑 j -> 全芯片物理 current_active[L[j]]
        seg_plugin = lift_original_by_layout(qc_small, L, current_active, num_chip_qubits)
        segments.append(seg_plugin)
        print(f"[插件层 {p}] 插入原始小电路（按 L 映射到当前活跃区物理线）")

        # SWAP 生效：更新活跃区顺序
        current_active = apply_local_swaps(current_active, local_swaps)
        print(f"[插件层 {p}] 执行 SWAP 后的活跃区新顺序：{current_active}")

        # 统计信息记录
        stats["plugins"].append({
            "file": fname,
            "n_qubits": n_small,
            "initial_layout_L": L,
            "L_source": how,
            "local_swaps": local_swaps,
            "phys_swaps": phys_swaps
        })

        # 若不是最后一个插件层，则在“更新后的顺序”上生成中间 15 层
        if p != N_PLUGINS:
            seg_mid = generate_partial(G, num_chip_qubits, current_active, MID_LAYERS)
            segments.append(seg_mid)
            print(f"[插件层 {p}] 生成 {MID_LAYERS} 层随机电路（已按新顺序在活跃区上放门）")

    # 最后一段 15 层（第三个插件之后）
    seg_last = generate_partial(G, num_chip_qubits, current_active, MID_LAYERS)
    segments.append(seg_last)
    print(f"\n[步骤3] 生成最后 {MID_LAYERS} 层随机电路（使用最终活跃区顺序）")

    # 拼接所有段（全芯片宽度）
    full = QuantumCircuit(num_chip_qubits)
    for seg in segments:
        full = full.compose(seg)

    # 写出物理版（不包含任何 swap，因为我们从未插入过）
    os.makedirs(OUTPUT_DIR_PHYSICAL, exist_ok=True)
    os.makedirs(OUTPUT_DIR_REINDEXED, exist_ok=True)
    base = f"q{q_large}_s{sidx}"

    phys_path = os.path.join(OUTPUT_DIR_PHYSICAL, f"{base}_combined3_phys.qasm")
    with open(phys_path, 'w') as f:
        f.write(qasm2_dumps(full))
    print(f"[输出] 物理索引版 QASM：{phys_path}")

    # 写出重索引版（默认按“初始活跃区顺序”重排；如需按“最终活跃区”，把 active0 换成 current_active）
    used_nodes_in_order = active0[:]        # 或者改成：current_active[:]
    reidx_qc  = reindex_compact(full, used_nodes_in_order)
    reidx_path = os.path.join(OUTPUT_DIR_REINDEXED, f"{base}_combined3_reidx.qasm")
    with open(reidx_path, 'w') as f:
        f.write(qasm2_dumps(reidx_qc))
    print(f"[输出] 重索引（紧致）版 QASM：{reidx_path}")

    stats["paths"] = {"physical_qasm": phys_path, "reindexed_qasm": reidx_path}
    return stats


# ===================== 主函数 =====================

def main():
    print("加载芯片拓扑 ...")
    G = load_chip_graph(EDGE_LIST_PATH)
    print(f"物理比特总数：{len(G.nodes())}，边数：{len(G.edges())}")

    all_stats = []
    for q in range(MIN_LARGE_QUBITS, MAX_LARGE_QUBITS+1):
        for s in range(1, NUM_INSTANCES_PER_SIZE+1):
            all_stats.append(build_instance(G, q, s))

    with open(STATS_JSON, 'w', encoding='utf-8') as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=2)
    print(f"\n完成：共写入 {len(all_stats)} 条实例统计 -> {STATS_JSON}")


if __name__ == "__main__":
    main()
