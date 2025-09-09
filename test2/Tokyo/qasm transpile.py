import os
import random
import json
import networkx as nx
from qiskit import QuantumCircuit
from qiskit.qasm2 import dumps as qasm2_dumps
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.circuit.library import Barrier
from mqt import qmap

# --- 1. 配置区 (Configuration) ---
RANDOM_CIRCUIT_DIR = 'quantum_dataset_random 10'
EDGE_LIST_PATH     = 'chip_topology.edgelist'
OUTPUT_DIR_PHYSICAL = 'combined_circuits_physical'
OUTPUT_DIR_REINDEXED = 'combined_circuits_reindexed'
STATS_OUTPUT_JSON = 'dynamic_compilation_stats.json'
QUBIT_MAPPING = {4: 9, 5: 10}
TOTAL_LAYERS = 30

# --- 2. 辅助函数 (Helper Functions) ---

def load_coupling_graph(path):
    """从 edgelist 文件加载 NetworkX 图"""
    return nx.read_edgelist(path, nodetype=int)

def sample_connected_subgraph(G: nx.Graph, k: int, max_attempts: int = 100) -> list:
    """从图中采样一个 k 节点的连通子图"""
    nodes = list(G.nodes())
    for _ in range(max_attempts):
        start_node = random.choice(nodes)
        subgraph_nodes = {start_node}
        frontier = list(G.neighbors(start_node))
        random.shuffle(frontier)
        while frontier and len(subgraph_nodes) < k:
            new_node = frontier.pop(0)
            if new_node not in subgraph_nodes:
                subgraph_nodes.add(new_node)
                new_neighbors = list(G.neighbors(new_node))
                random.shuffle(new_neighbors)
                frontier.extend([neighbor for neighbor in new_neighbors if neighbor not in subgraph_nodes])
        if len(subgraph_nodes) == k:
            return sorted(list(subgraph_nodes))
    raise RuntimeError(f"无法从图中采样到 {k} 个连通节点。")

def generate_partial_circuit(num_total_qubits: int, sub_nodes: list, num_layers: int, G: nx.Graph) -> QuantumCircuit:
    """根据给定的物理子图节点生成部分随机电路"""
    qc = QuantumCircuit(num_total_qubits)
    subgraph_edges = list(G.subgraph(sub_nodes).edges())
    for _ in range(num_layers):
        if random.random() < 0.5 and subgraph_edges:
            u, v = random.choice(subgraph_edges)
            qc.cx(u, v)
        else:
            q = random.choice(sub_nodes)
            gate = random.choice([qc.x, qc.y, qc.z, qc.h, qc.s])
            gate(q)
        qc.barrier(sub_nodes)
    return qc

def get_layouts_from_compilation(qc_small: QuantumCircuit, arch: GenericBackendV2) -> tuple:
    """运行 MQT qmap 的精确编译方法，获取布局和置换信息。"""
    mapped_qc, res = qmap.compile(qc_small, arch, method='exact')
    initial_layout = {vq._index: ip for vq, ip in mapped_qc.layout.initial_layout._v2p.items()}
    
    # 关键修正：确保final_permutation只包含在初始布局中使用的物理比特
    final_permutation = {}
    initial_physical_qubits = set(initial_layout.values())

    for v_qubit, initial_phys_idx in mapped_qc.layout.initial_layout._v2p.items():
        # 确保我们只处理那些参与了小电路映射的物理比特
        if initial_phys_idx in initial_physical_qubits:
            final_phys_idx = mapped_qc.layout.final_layout[v_qubit]
            final_permutation[initial_phys_idx] = final_phys_idx
            
    return initial_layout, final_permutation, res.output.swaps

def extract_swaps_from_permutation(permutation: dict) -> list:
    """
    从置换字典中提取出实际的SWAP对。
    例如, {8: 9, 9: 8, 12: 12} -> [(8, 9)]
    """
    swaps = []
    processed_qubits = set()
    for initial_q, final_q in permutation.items():
        if initial_q in processed_qubits:
            continue
        # 寻找一个长度为2的循环，即一个SWAP
        if permutation.get(final_q) == initial_q and initial_q != final_q:
            # 添加排序后的元组以避免重复，如(8,9)和(9,8)
            swaps.append(tuple(sorted((initial_q, final_q))))
            processed_qubits.add(initial_q)
            processed_qubits.add(final_q)
    return swaps

def remap_original_circuit(qc_orig: QuantumCircuit, initial_layout: dict, num_target_qubits: int) -> QuantumCircuit:
    """根据最优初始布局，将原始小电路的门应用到大电路的物理比特上 (不含SWAP)。"""
    remapped_qc = QuantumCircuit(num_target_qubits)
    original_qubits = list(qc_orig.qubits)
    for instruction in qc_orig.data:
        op, qargs, cargs = instruction.operation, instruction.qubits, instruction.clbits
        new_qargs = [remapped_qc.qubits[initial_layout[original_qubits.index(q)]] for q in qargs]
        new_cargs = [remapped_qc.clbits[c._index] for c in cargs]
        remapped_qc.append(op, new_qargs, new_cargs)
    return remapped_qc

def reindex_circuit(qc: QuantumCircuit, index_map: dict) -> QuantumCircuit:
    """根据 index_map (旧物理索引 -> 新逻辑索引) 对电路进行重索引。"""
    new_qc = QuantumCircuit(len(index_map), qc.num_clbits)
    for instruction in qc.data:
        op, qargs, cargs = instruction.operation, instruction.qubits, instruction.clbits
        if isinstance(op, Barrier):
            qubits_for_new_barrier = [new_qc.qubits[index_map[q._index]] for q in qargs if q._index in index_map]
            if qubits_for_new_barrier:
                 new_qc.barrier(qubits_for_new_barrier)
            continue
        new_qargs = [new_qc.qubits[index_map[q._index]] for q in qargs]
        new_cargs = [new_qc.clbits[c._index] for c in cargs]
        new_qc.append(op, new_qargs, new_cargs)
    return new_qc

# --- 3. 主工作流 (Main Workflow) ---

def main():
    print("开始执行编译感知的动态电路生成工作流 (版本: v3_final)...")
    os.makedirs(OUTPUT_DIR_PHYSICAL, exist_ok=True)
    os.makedirs(OUTPUT_DIR_REINDEXED, exist_ok=True)
    G = load_coupling_graph(EDGE_LIST_PATH)
    num_chip_qubits = len(G.nodes())
    stats_records = []

    for fname in sorted(os.listdir(RANDOM_CIRCUIT_DIR)):
        if not fname.endswith('.qasm'): continue
        try:
            parts = fname[:-5].split('_')
            q_small = int(parts[0][1:])
            s_small = int(parts[2][1:])
            q_large = QUBIT_MAPPING.get(q_small)
            if q_large is None: continue
        except (ValueError, IndexError): continue

        print(f"\n--- 正在处理 {fname} (小电路: {q_small} qubits, 大电路: {q_large} qubits) ---")

        try:
            sub_nodes_initial = sample_connected_subgraph(G, q_large)
            print(f"  - 步骤 1: 已采样 {q_large} 个物理比特: {sub_nodes_initial}")
        except RuntimeError as e:
            print(f"  - 采样失败: {e}，跳过。")
            continue
        
        front_qc = generate_partial_circuit(num_chip_qubits, sub_nodes_initial, TOTAL_LAYERS // 2, G)
        qc_small = QuantumCircuit.from_qasm_file(os.path.join(RANDOM_CIRCUIT_DIR, fname))

        print("  - 步骤 2: 运行MQT Exact获取最优布局和置换信息...")
        sub_map_for_compile = [e for e in G.edges() if e[0] in sub_nodes_initial and e[1] in sub_nodes_initial]
        arch = GenericBackendV2(num_qubits=max(sub_nodes_initial) + 1, coupling_map=sub_map_for_compile)
        
        try:
            initial_layout, final_permutation, swap_count = get_layouts_from_compilation(qc_small, arch)
            
            print(f"    - 编译完成，理论最优SWAP数: {swap_count}")
            print(f"    - MQT报告的初始布局 (逻辑->物理): {initial_layout}")
            print(f"    - MQT报告的最终置换 (初物理->末物理): {final_permutation}")

            swapped_pairs = extract_swaps_from_permutation(final_permutation)
            if swapped_pairs:
                print(f"    - ✨ 分析出的SWAP操作作用于物理比特: {swapped_pairs}")
            else:
                print("    - 未检测到明确的SWAP对（可能是更复杂的置换或无置换）。")

            stats_records.append({
                'filename': fname,
                'exact_swaps': swap_count,
                'swapped_pairs': [list(p) for p in swapped_pairs] 
            })

        except Exception as e:
            print(f"    - MQT编译失败: {e}，跳过。")
            continue
        
        print("  - 步骤 3: 根据精确布局，重映射 *原始* 小电路 (不含SWAP)...")
        plugin_qc = remap_original_circuit(qc_small, initial_layout, num_chip_qubits)

        print("  - 步骤 4: 计算置换后的新物理比特列表用于生成 'back' 电路...")
        print(f"    - 初始物理比特列表: {sub_nodes_initial}")

        sub_nodes_permuted = [final_permutation.get(p, p) for p in sub_nodes_initial]
        
        assert set(sub_nodes_initial) == set(sub_nodes_permuted), \
            f"错误：置换前后物理比特集合不一致！\n初始: {set(sub_nodes_initial)}\n置换后: {set(sub_nodes_permuted)}"
        print(f"    - 置换后物理比特列表: {sub_nodes_permuted}")

        print("  - 步骤 5: 在新的物理比特列表上生成 'back' 电路...")
        back_qc = generate_partial_circuit(num_chip_qubits, sub_nodes_permuted, TOTAL_LAYERS - (TOTAL_LAYERS // 2), G)
        
        print("  - 步骤 6: 组合并保存最终电路...")
        final_qc = front_qc.compose(plugin_qc).compose(back_qc)
        
        out_name_base = f"q{q_large}_s{s_small-1}"
        out_name_phys = f"{out_name_base}_combined_phys.qasm"
        with open(os.path.join(OUTPUT_DIR_PHYSICAL, out_name_phys), 'w') as f:
            f.write(qasm2_dumps(final_qc))
        print(f"    - 成功生成物理版: {out_name_phys}")

        used_set = {q._index for instr in final_qc.data if not isinstance(instr.operation, Barrier) for q in instr.qubits}
        index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted(list(used_set)))}
        reindexed_qc = reindex_circuit(final_qc, index_map)
        out_name_reidx = f"{out_name_base}_combined_reidx.qasm"
        with open(os.path.join(OUTPUT_DIR_REINDEXED, out_name_reidx), 'w') as f:
            f.write(qasm2_dumps(reindexed_qc))
        print(f"    - 成功生成重索引版: {out_name_reidx}")

    if stats_records:
        with open(STATS_OUTPUT_JSON, 'w') as f:
            json.dump(stats_records, f, indent=2)
        print(f"\n已将 MQT Exact 统计信息保存到 {STATS_OUTPUT_JSON}")

if __name__ == '__main__':
    random.seed(42)
    main()
