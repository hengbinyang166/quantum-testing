import os
import random
import json
import networkx as nx
from qiskit import QuantumCircuit
from qiskit.qasm2 import dumps as qasm2_dumps
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.circuit.library import Barrier
from mqt import qmap
import traceback

# --- 1. 配置区 ---
RANDOM_CIRCUIT_DIR = 'quantum_dataset_random 10'
EDGE_LIST_PATH     = 'chip_topology.edgelist'
OUTPUT_DIR_PHYSICAL = 'combined_circuits_physical'
OUTPUT_DIR_REINDEXED = 'combined_circuits_reindexed'
STATS_OUTPUT_JSON = 'dynamic_compilation_stats.json'
OUTPUT_DIR_QMAP_EXACT = 'qmap_exact_output'

# --- 新增配置 ---
# ✨✨✨ 修改点 1：调整大电路的生成范围 ✨✨✨
# 您可以自由设置这个范围来补充数据，例如 (3, 7)
MIN_LARGE_QUBITS = 14
MAX_LARGE_QUBITS = 27 # 设置为-1或小于MIN_LARGE_QUBITS的值，代表一直运行到芯片最大比特数

NUM_INSTANCES_PER_SIZE = 10
TOTAL_LAYERS = 30


# --- 2. 辅助函数 (保持不变) ---
# (这里的所有辅助函数 load_coupling_graph, sample_connected_subgraph, 等... 均保持原样，未作改动)
def load_coupling_graph(path):
    return nx.read_edgelist(path, nodetype=int)

def sample_connected_subgraph(G: nx.Graph, k: int, max_attempts: int = 100) -> list:
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
    qc = QuantumCircuit(num_total_qubits)
    if not sub_nodes: return qc
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

def get_layout_and_compiler_swaps(qc_small: QuantumCircuit, arch: GenericBackendV2) -> tuple:
    mapped_qc, res = qmap.compile(qc_small, arch, method='exact')
    num_logical_qubits = qc_small.num_qubits
    initial_layout = { vq._index: ip for vq, ip in mapped_qc.layout.initial_layout._v2p.items() if vq._index < num_logical_qubits }
    compiler_swaps = []
    for instruction in mapped_qc.data:
        if instruction.operation.name == 'swap':
            q0_local_index = instruction.qubits[0]._index
            q1_local_index = instruction.qubits[1]._index
            compiler_swaps.append(tuple(sorted((q0_local_index, q1_local_index))))
    return initial_layout, compiler_swaps, res.output.swaps, mapped_qc

def remap_original_circuit(qc_orig: QuantumCircuit, initial_layout: dict, num_target_qubits: int) -> QuantumCircuit:
    remapped_qc = QuantumCircuit(num_target_qubits)
    original_qubits = list(qc_orig.qubits)
    for instruction in qc_orig.data:
        op, qargs, cargs = instruction.operation, instruction.qubits, instruction.clbits
        new_qargs = [remapped_qc.qubits[initial_layout[original_qubits.index(q)]] for q in qargs]
        new_cargs = [remapped_qc.clbits[c._index] for c in cargs]
        remapped_qc.append(op, new_qargs, new_cargs)
    return remapped_qc

def reindex_circuit(qc: QuantumCircuit, index_map: dict) -> QuantumCircuit:
    new_qc = QuantumCircuit(len(index_map), qc.num_clbits)
    for instruction in qc.data:
        op, qargs, cargs = instruction.operation, instruction.qubits, instruction.clbits
        if isinstance(op, Barrier):
            # 如果源电路中仍有Barrier，此代码块会处理它。
            # 但因为我们稍后会移除所有Barrier，所以这个代码块实际上不会被执行。
            qubits_for_new_barrier = [new_qc.qubits[index_map[q._index]] for q in qargs if q._index in index_map]
            if qubits_for_new_barrier: new_qc.barrier(qubits_for_new_barrier)
            continue
        new_qargs = [new_qc.qubits[index_map[q._index]] for q in qargs]
        new_cargs = [new_qc.clbits[c._index] for c in cargs]
        new_qc.append(op, new_qargs, new_cargs)
    return new_qc

# --- 请将这个新函数添加到您的辅助函数区域 ---

def generate_permuted_topology_circuit(num_total_qubits: int, original_nodes: list, permuted_nodes: list, num_layers: int, G: nx.Graph) -> QuantumCircuit:
    """
    在根据SWAP置换后的【虚拟拓扑】上生成部分电路。
    """
    qc = QuantumCircuit(num_total_qubits)
    if not original_nodes:
        return qc

    # 步骤 1: 获取原始活跃区的物理连接关系
    original_subgraph = G.subgraph(original_nodes)
    original_edges = list(original_subgraph.edges())

    # 步骤 2: 创建从原始节点到置换后节点的映射
    node_map = {original_nodes[i]: permuted_nodes[i] for i in range(len(original_nodes))}

    # 步骤 3: 根据映射创建新的“虚拟边”
    virtual_edges = []
    for u, v in original_edges:
        if u in node_map and v in node_map:
            virtual_edges.append((node_map[u], node_map[v]))

    # 步骤 4: 在新的虚拟边上随机生成CNOT门
    for _ in range(num_layers):
        if random.random() < 0.5 and virtual_edges:
            u, v = random.choice(virtual_edges)
            qc.cx(u, v)
        else:
            q = random.choice(permuted_nodes)
            gate = random.choice([qc.x, qc.y, qc.z, qc.h, qc.s])
            gate(q)
            
    # 注意：为了满足您的新需求，这里的barrier可以移除，
    # 或者保留（因为它最终也会被移除）。为了代码清晰，我们暂时保留，
    # 在最后统一处理。
    qc.barrier(permuted_nodes)
            
    return qc
    
# --- 3. 主工作流 ---
def main():
    print("开始执行编译感知的动态电路生成工作流...")
    os.makedirs(OUTPUT_DIR_PHYSICAL, exist_ok=True)
    os.makedirs(OUTPUT_DIR_REINDEXED, exist_ok=True)
    os.makedirs(OUTPUT_DIR_QMAP_EXACT, exist_ok=True)
    
    G = load_coupling_graph(EDGE_LIST_PATH)
    num_chip_qubits = len(G.nodes())
    print(f"芯片拓扑结构加载成功，共有 {num_chip_qubits} 个量子比特。")

    stats_records = []
    try:
        with open(STATS_OUTPUT_JSON, 'r') as f:
            stats_records = json.load(f)
        print(f"成功加载了 {len(stats_records)} 条已有的编译统计记录。")
        if not isinstance(stats_records, list):
            print("  - 警告: JSON文件内容不是一个列表，将重新开始记录。")
            stats_records = []
    except FileNotFoundError:
        print("未找到旧的统计文件，将创建新的记录。")
    except json.JSONDecodeError:
        print("  - 警告: JSON文件格式错误，将重新开始记录。")
        stats_records = []
    
    try:
        all_small_circuit_files = [f for f in os.listdir(RANDOM_CIRCUIT_DIR) if f.endswith('.qasm')]
        if not all_small_circuit_files:
            print(f"错误：在目录 '{RANDOM_CIRCUIT_DIR}' 中未找到任何 '.qasm' 文件。程序将退出。")
            return
        print(f"在 '{RANDOM_CIRCUIT_DIR}' 中找到 {len(all_small_circuit_files)} 个可用的小电路文件。")
    except FileNotFoundError:
        print(f"错误：目录 '{RANDOM_CIRCUIT_DIR}' 不存在。请检查路径。")
        return

    run_until_qubits = MAX_LARGE_QUBITS if MAX_LARGE_QUBITS >= MIN_LARGE_QUBITS else num_chip_qubits
    
    for q_large in range(MIN_LARGE_QUBITS, run_until_qubits + 1):
        print(f"\n{'='*60}")
        print(f"开始为 {q_large} 比特的大电路生成 {NUM_INSTANCES_PER_SIZE} 个实例")
        print(f"{'='*60}")

        eligible_files = []
        for f in all_small_circuit_files:
            try:
                q_s = int(f.split('_')[0][1:])
                if q_s <= q_large:
                    eligible_files.append(f)
            except (ValueError, IndexError):
                continue
        
        if not eligible_files:
            print(f"  - [警告] 对于 q_large={q_large}, 在 '{RANDOM_CIRCUIT_DIR}' 中没有找到合适的 (比特数 <= {q_large}) 小电路。")
            print("  - 跳过此尺寸的所有实例。")
            continue

        print(f"  - 对于 {q_large} 比特的大电路，有 {len(eligible_files)} 个候选小电路可供选择。")

        for instance_idx in range(1, NUM_INSTANCES_PER_SIZE + 1):
            fname = random.choice(eligible_files)
            try:
                parts = fname[:-5].split('_')
                q_small = int(parts[0][1:])
            except (ValueError, IndexError):
                print(f"  - [警告] 无法解析文件名 {fname}，跳过此实例。")
                continue

            print(f"\n--- 正在处理实例 {instance_idx}/{NUM_INSTANCES_PER_SIZE} (活跃区: {q_large} qubits) ---")
            print(f"  - 随机选择的小电路: {fname} (逻辑比特: {q_small} qubits)")

            try:
                active_nodes = sample_connected_subgraph(G, q_large)
                print(f"  - [步骤 1] 已采样 {q_large} 个物理比特作为活跃区: {active_nodes}")
            except RuntimeError as e:
                print(f"  - [步骤 1] 采样失败: {e}，跳过此实例。")
                continue
            
            front_qc = generate_partial_circuit(num_chip_qubits, active_nodes, TOTAL_LAYERS // 2, G)
            qc_small = QuantumCircuit.from_qasm_file(os.path.join(RANDOM_CIRCUIT_DIR, fname))
            
            print("  - [步骤 2] 准备在活跃区上编译小电路...")
            phys_to_local = {phys: i for i, phys in enumerate(active_nodes)}
            local_to_phys = {i: phys for phys, i in phys_to_local.items()}
            
            active_subgraph_edges = G.subgraph(active_nodes).edges()
            bidirectional_local_coupling_map = []
            for u, v in active_subgraph_edges:
                local_u, local_v = phys_to_local[u], phys_to_local[v]
                bidirectional_local_coupling_map.append((local_u, local_v))
                bidirectional_local_coupling_map.append((local_v, local_u))

            arch = GenericBackendV2(num_qubits=q_large, coupling_map=bidirectional_local_coupling_map)
            
            try:
                local_initial_layout, local_compiler_swaps, swap_count, mapped_qc = get_layout_and_compiler_swaps(qc_small, arch)
                
                qmap_out_name = f"q{q_large}_s{instance_idx}_qmap_exact.qasm"
                qmap_out_path = os.path.join(OUTPUT_DIR_QMAP_EXACT, qmap_out_name)
                with open(qmap_out_path, 'w') as f: f.write(qasm2_dumps(mapped_qc))

                initial_layout = {log_q: local_to_phys[local_q] for log_q, local_q in local_initial_layout.items()}
                physical_swaps = [tuple(sorted((local_to_phys[q0], local_to_phys[q1]))) for q0, q1 in local_compiler_swaps]

                print(f"    - [步骤 2] 编译完成，额外SWAP数: {swap_count}")
                print(f"    - [步骤 2] MQT中间结果已保存到: {qmap_out_path}")
                print(f"    - [步骤 2] 初始布局 (逻辑->物理): {initial_layout}")
                print(f"    - [步骤 2] 编译器添加的SWAP（物理比特）: {physical_swaps}")
                
                stats_records.append({ 'filename': fname, 'q_large': q_large, 'instance': instance_idx, 'swap_count': swap_count, 'compiler_swaps': physical_swaps })
                
                with open(STATS_OUTPUT_JSON, 'w') as f:
                    json.dump(stats_records, f, indent=4)
                print(f"    - [保存成功] 统计数据已更新。当前总记录数: {len(stats_records)}")

            except Exception as e:
                print(f"    - [步骤 2] MQT编译失败: {e}")
                print(traceback.format_exc()) # 打印详细错误信息
                continue
            
            plugin_qc = remap_original_circuit(qc_small, initial_layout, num_chip_qubits)
            
            permuted_nodes = active_nodes.copy()
            if physical_swaps:
                print("  - [信息] 检测到编译器添加了SWAP，正在调整back电路拓扑...")
                for q0, q1 in physical_swaps:
                    try:
                        idx0, idx1 = permuted_nodes.index(q0), permuted_nodes.index(q1)
                        permuted_nodes[idx0], permuted_nodes[idx1] = permuted_nodes[idx1], permuted_nodes[idx0]
                    except ValueError:
                        print(f"  - [错误] SWAP中的比特 {q0} 或 {q1} 不在活跃区中！")
                        continue
            else:
                 print("  - [信息] 编译器未添加SWAP，back电路拓扑保持不变。")

            assert set(active_nodes) == set(permuted_nodes), "置换后节点集合不匹配！"
            
            print(f"  - [步骤 3] 在【置换后的虚拟拓扑】上生成 back 电路。")
            back_qc = generate_permuted_topology_circuit(
                num_total_qubits=num_chip_qubits,
                original_nodes=active_nodes,
                permuted_nodes=permuted_nodes,
                num_layers=TOTAL_LAYERS - (TOTAL_LAYERS // 2),
                G=G
            )
            
            final_qc = front_qc.compose(plugin_qc).compose(back_qc)
            print(f"  - [步骤 4] 成功组合 front, plugin, back 电路。")
            
            # ✨✨✨ 最小化修改：替换此行以兼容旧版Qiskit ✨✨✨
            # 原来的代码: final_qc.remove_instruction(Barrier)
            # 新的代码通过列表推导式过滤掉Barrier，功能完全相同。
            final_qc.data = [instr for instr in final_qc.data if not isinstance(instr.operation, Barrier)]
            print(f"  - [步骤 5] 已从最终电路中移除所有 barrier 指令。")
            
            out_name_base = f"q{q_large}_s{instance_idx}"
            
            out_name_phys = f"{out_name_base}_combined_phys.qasm"
            with open(os.path.join(OUTPUT_DIR_PHYSICAL, out_name_phys), 'w') as f:
                f.write(qasm2_dumps(final_qc))
            print(f"    - 成功生成物理版 (无barrier): {out_name_phys}")

            # 因为 reindexed_qc 是基于已经移除了barrier的 final_qc 生成的，
            #所以它自然也不再包含barrier。
            used_set = {q._index for instr in final_qc.data if not isinstance(instr.operation, Barrier) for q in instr.qubits}
            index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted(list(used_set)))}
            reindexed_qc = reindex_circuit(final_qc, index_map)
            out_name_reidx = f"{out_name_base}_combined_reidx.qasm"
            with open(os.path.join(OUTPUT_DIR_REINDEXED, out_name_reidx), 'w') as f:
                f.write(qasm2_dumps(reindexed_qc))
            print(f"    - 成功生成重索引版 (无barrier): {out_name_reidx}")

    if stats_records:
        with open(STATS_OUTPUT_JSON, 'w') as f:
            json.dump(stats_records, f, indent=4)
        print(f"\n已将所有编译统计信息保存到 {STATS_OUTPUT_JSON}")
    
    print("\n所有任务已完成。")

if __name__ == '__main__':
    random.seed(42)
    main()