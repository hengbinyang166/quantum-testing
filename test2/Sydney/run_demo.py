import os
import random
import json
import sys
import networkx as nx
from qiskit import QuantumCircuit
from qiskit.qasm2 import dumps as qasm2_dumps
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.circuit.library import Barrier
from mqt import qmap
import traceback

# --- 1. 配置区 ---
DEMO_OUTPUT_DIR    = 'demo_output'
VIS_DIR            = os.path.join(DEMO_OUTPUT_DIR, 'visualizations')
EDGE_LIST_PATH     = 'chip_topology.edgelist'
OUTPUT_DIR_PHYSICAL = os.path.join(DEMO_OUTPUT_DIR, 'combined_circuits_physical')
OUTPUT_DIR_REINDEXED = os.path.join(DEMO_OUTPUT_DIR, 'combined_circuits_reindexed')
OUTPUT_DIR_QMAP_EXACT = os.path.join(DEMO_OUTPUT_DIR, 'qmap_exact_output')

# ✨ 新增配置：控制生成数量和随机电路深度
NUM_INSTANCES = 10
FRONT_LAYERS = 5
BACK_LAYERS = 5

# --- 2. 辅助函数 ---
def load_coupling_graph(path):
    if not os.path.exists(path):
        print(f"错误：芯片拓扑文件 '{path}' 未找到。")
        sys.exit(1)
    return nx.read_edgelist(path, nodetype=int)

def save_circuit_visualization(qc, file_path, title):
    try:
        qc.draw(output='mpl', filename=file_path, style={'name': 'bw'}, fold=-1)
        print(f"  - [可视化] '{title}' 电路图已保存到: {file_path}")
    except Exception as e:
        print(f"  - [可视化警告] 跳过'{title}'电路图生成。原因: {e}")

def generate_partial_circuit(num_total_qubits: int, sub_nodes: list, num_layers: int, G: nx.Graph) -> QuantumCircuit:
    """在【原始物理拓扑】上随机生成电路 (用于前置电路)"""
    qc = QuantumCircuit(num_total_qubits, name="Random_Front_QC")
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
    return qc

def generate_permuted_topology_circuit(num_total_qubits: int, original_nodes: list, permuted_nodes: list, num_layers: int, G: nx.Graph) -> QuantumCircuit:
    """在根据SWAP置换后的【虚拟拓扑】上随机生成电路 (用于后置电路)"""
    qc = QuantumCircuit(num_total_qubits, name="Dynamic_Back_QC")
    if not original_nodes: return qc
    original_edges = list(G.subgraph(original_nodes).edges())
    node_map = {original_nodes[i]: permuted_nodes[i] for i in range(len(original_nodes))}
    virtual_edges = []
    for u, v in original_edges:
        if u in node_map and v in node_map:
            virtual_edges.append((node_map[u], node_map[v]))
    for _ in range(num_layers):
        if random.random() < 0.5 and virtual_edges:
            u, v = random.choice(virtual_edges)
            qc.cx(u, v)
        else:
            q = random.choice(permuted_nodes)
            gate = random.choice([qc.x, qc.y, qc.z, qc.h, qc.s])
            gate(q)
    return qc

def remove_barriers(qc: QuantumCircuit) -> QuantumCircuit:
    new_qc = QuantumCircuit(qc.num_qubits, qc.num_clbits)
    for instr in qc.data:
        if instr.operation.name != 'barrier':
            new_qc.append(instr.operation, instr.qubits, instr.clbits)
    return new_qc

def get_layout_and_compiler_swaps(qc_small: QuantumCircuit, arch: GenericBackendV2) -> tuple:
    mapped_qc, res = qmap.compile(qc_small, arch, method='exact')
    num_logical_qubits = qc_small.num_qubits
    initial_layout = { vq._index: ip for vq, ip in mapped_qc.layout.initial_layout.get_virtual_bits().items() if vq._index < num_logical_qubits }
    compiler_swaps = []
    for instruction in mapped_qc.data:
        if instruction.operation.name == 'swap':
            q0_local_index = instruction.qubits[0]._index
            q1_local_index = instruction.qubits[1]._index
            compiler_swaps.append(tuple(sorted((q0_local_index, q1_local_index))))
    return initial_layout, compiler_swaps, res.output.swaps, mapped_qc

def remap_original_circuit(qc_orig: QuantumCircuit, initial_layout: dict, num_target_qubits: int) -> QuantumCircuit:
    remapped_qc = QuantumCircuit(num_target_qubits, qc_orig.num_clbits)
    original_qubits = list(qc_orig.qubits)
    for instruction in qc_orig.data:
        op, qargs, cargs = instruction.operation, instruction.qubits, instruction.clbits
        new_qargs = [remapped_qc.qubits[initial_layout[original_qubits.index(q)]] for q in qargs]
        new_cargs = [remapped_qc.clbits[c._index] for c in cargs] if cargs else []
        remapped_qc.append(op, new_qargs, new_cargs)
    return remapped_qc

def reindex_circuit(qc: QuantumCircuit, index_map: dict) -> QuantumCircuit:
    new_qc = QuantumCircuit(len(index_map), qc.num_clbits)
    for instruction in qc.data:
        op, qargs, cargs = instruction.operation, instruction.qubits, instruction.clbits
        if isinstance(op, Barrier):
            qubits_for_new_barrier = [new_qc.qubits[index_map[q._index]] for q in qargs if q._index in index_map]
            if qubits_for_new_barrier: new_qc.barrier(qubits_for_new_barrier)
            continue
        new_qargs = [new_qc.qubits[index_map[q._index]] for q in qargs]
        new_cargs = [new_qc.clbits[c._index] for c in cargs] if cargs else []
        new_qc.append(op, new_qargs, new_cargs)
    return new_qc

# --- 3. 核心执行函数 ---
def generate_one_instance(instance_idx: int, G: nx.Graph, num_chip_qubits: int):
    base_name = f"fixed_middle_circuit_inst_{instance_idx}"
    print(f"\n{'='*70}\n--- 正在生成实例 {instance_idx}: {base_name} ---\n{'='*70}")
    
    active_nodes = [9, 8, 11, 14, 13] 
    q_large = len(active_nodes)

    # --- 1. 动态生成随机的前置电路 ---
    print("  - [步骤 1] 正在动态生成随机的前置电路...")
    front_qc = generate_partial_circuit(num_chip_qubits, active_nodes, FRONT_LAYERS, G)
    save_circuit_visualization(front_qc, os.path.join(VIS_DIR, f"{base_name}_1_front_qc.png"), f"1. {base_name} - 前置电路")

    # --- 2. 使用固定的逻辑小电路 (中间部分) ---
    print("  - [步骤 2] 正在使用固定的逻辑小电路 (用于MQT编译)...")
    qc_small = QuantumCircuit(4, name="Logical_Middle_QC")
    qc_small.cx(0, 3)
    qc_small.cx(1, 0)
    qc_small.cx(3, 1)
    qc_small.x(2)
    save_circuit_visualization(qc_small, os.path.join(VIS_DIR, f"{base_name}_2_qc_small_logical.png"), f"2. {base_name} - 逻辑小电路")

    # --- 3. MQT Exact 编译核心流程 ---
    print("  - [步骤 3] 准备使用MQT Exact编译逻辑小电路...")
    active_subgraph = G.subgraph(active_nodes)
    phys_to_local = {phys: i for i, phys in enumerate(active_nodes)}
    bidirectional_local_coupling_map = []
    for u, v in active_subgraph.edges():
        local_u, local_v = phys_to_local.get(u, -1), phys_to_local.get(v, -1)
        if local_u != -1 and local_v != -1:
            bidirectional_local_coupling_map.append((local_u, local_v)); bidirectional_local_coupling_map.append((local_v, local_u))
    arch = GenericBackendV2(num_qubits=q_large, coupling_map=bidirectional_local_coupling_map)
    
    physical_swaps = []
    try:
        qc_small_no_barriers = remove_barriers(qc_small)
        local_initial_layout, local_compiler_swaps, swap_count, mapped_qc = get_layout_and_compiler_swaps(qc_small_no_barriers, arch)
        print(f"    - [编译成功] MQT 精确编译完成，报告额外SWAP数量: {swap_count}")
        
        # ✨✨✨ 核心修正：重新加入此行以保存MQT编译结果的可视化图片 ✨✨✨
        save_circuit_visualization(mapped_qc, os.path.join(VIS_DIR, f"{base_name}_3_mapped_qc.png"), f"3. {base_name} - MQT编译结果")
        
        local_to_phys = {i: phys for i, phys in enumerate(active_nodes)}
        initial_layout = {log_q: local_to_phys[local_q] for log_q, local_q in local_initial_layout.items()}
        physical_swaps = [tuple(sorted((local_to_phys[q0], local_to_phys[q1]))) for q0, q1 in local_compiler_swaps]
        print(f"    - [MQT 报告] 最终电路中插入的物理SWAP门: {physical_swaps}")
        print(f"    - [MQT 报告] 最终的最优初始布局 (逻辑 -> 物理比特): {initial_layout}")
    except Exception as e:
        print(f"  - [编译失败] MQT 编译时出错: {e}"); traceback.print_exc(); return

    # --- 4. 动态生成后置电路 ---
    print("  - [步骤 4] 正在根据SWAP信息，在虚拟拓扑上动态生成后置电路...")
    permuted_nodes = active_nodes.copy()
    if physical_swaps:
        for q0, q1 in physical_swaps:
            try:
                idx0, idx1 = permuted_nodes.index(q0), permuted_nodes.index(q1)
                permuted_nodes[idx0], permuted_nodes[idx1] = permuted_nodes[idx1], permuted_nodes[idx0]
            except ValueError:
                print(f"    - [警告] SWAP中的比特 {q0} 或 {q1} 不在活跃区中！")
    
    back_qc = generate_permuted_topology_circuit(
        num_total_qubits=num_chip_qubits,
        original_nodes=active_nodes,
        permuted_nodes=permuted_nodes,
        num_layers=BACK_LAYERS,
        G=G
    )
    save_circuit_visualization(back_qc, os.path.join(VIS_DIR, f"{base_name}_4_back_qc.png"), f"4. {base_name} - 动态后置电路")

    # --- 5. 组合所有部分 ---
    print("  - [步骤 5] 正在组合 前置 + (编译后的中间) + 后置 电路...")
    plugin_qc = remap_original_circuit(qc_small, initial_layout, num_chip_qubits)
    final_qc = front_qc.compose(plugin_qc).compose(back_qc)
    save_circuit_visualization(final_qc, os.path.join(VIS_DIR, f"{base_name}_5_final_qc.png"), f"5. {base_name} - 最终组合电路")
    
    # --- 6. 保存最终结果 ---
    final_qc_no_barriers = remove_barriers(final_qc)
    out_name_phys = f"{base_name}_phys.qasm"
    with open(os.path.join(OUTPUT_DIR_PHYSICAL, out_name_phys), 'w') as f: f.write(qasm2_dumps(final_qc_no_barriers))

    used_set = {q._index for instr in final_qc_no_barriers.data for q in instr.qubits}
    index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted(list(used_set)))}
    
    reindexed_qc = reindex_circuit(final_qc_no_barriers, index_map)
    out_name_reidx = f"{base_name}_reidx.qasm"
    with open(os.path.join(OUTPUT_DIR_REINDEXED, out_name_reidx), 'w') as f: f.write(qasm2_dumps(reindexed_qc))
    print(f"  - [保存成功] 已生成最终组合电路: {out_name_phys} 和 {out_name_reidx}")


# --- 4. 主函数 ---
def main():
    for d in [DEMO_OUTPUT_DIR, VIS_DIR, OUTPUT_DIR_PHYSICAL, OUTPUT_DIR_REINDEXED, OUTPUT_DIR_QMAP_EXACT]:
        os.makedirs(d, exist_ok=True)
        
    G = load_coupling_graph(EDGE_LIST_PATH)
    num_chip_qubits = len(G.nodes())
    print(f"芯片拓扑结构加载成功，共有 {num_chip_qubits} 个量子比特。")

    print(f"\n{'='*70}\n--- 开始生成 {NUM_INSTANCES} 个电路实例 ---\n{'='*70}")
    for i in range(1, NUM_INSTANCES + 1):
        try:
            generate_one_instance(i, G, num_chip_qubits)
        except Exception as e:
            print(f"\n生成实例 {i} 时遇到严重错误: {e}")
            traceback.print_exc()
            
    print(f"\n{'='*70}\n--- 所有 {NUM_INSTANCES} 个实例已生成完毕 ---\n{'='*70}")

if __name__ == '__main__':
    random.seed(42)
    main()