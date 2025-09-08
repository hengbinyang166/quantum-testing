import os
import csv
from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import CouplingMap
from qiskit.converters import circuit_to_dag
from qiskit.qasm2 import dumps
import networkx as nx
from networkx.algorithms import isomorphism

def load_coupling_map(edgelist_path: str) -> CouplingMap:
    """加载双向耦合图，避免单向边强制插入 SWAP。"""
    G = nx.read_edgelist(edgelist_path, nodetype=int)
    edges = list(G.edges())
    directed = edges + [(v, u) for u, v in edges]
    return CouplingMap(directed)

def strip_metadata(qc: QuantumCircuit) -> QuantumCircuit:
    """移除电路中的 barrier/measure/set_layout 元数据指令。"""
    # (此函数保持不变)
    clean = QuantumCircuit(qc.num_qubits)
    for instr in qc.data:
        op    = instr.operation if hasattr(instr, 'operation') else instr[0]
        qargs = instr.qubits    if hasattr(instr, 'qubits')    else instr[1]
        cargs = instr.clbits    if hasattr(instr, 'clbits')    else instr[2]
        if op.name not in ('barrier', 'measure', 'set_layout'):
            clean.append(op, qargs, cargs)
    return clean

def build_graph_from_dag(dag):
    """
    使用 topological_op_nodes 构建 MultiDiGraph。节点带 name 属性，边为依赖关系。
    """
    # (此函数保持不变)
    nodes = list(dag.topological_op_nodes())
    G = nx.MultiDiGraph()
    for node in nodes:
        G.add_node(node, name=node.name)
    for node in nodes:
        for succ in dag.successors(node):
            if succ in nodes:
                G.add_edge(node, succ)
    return G

def main():
    qasm_dir      = "quantum_dataset_remap"
    edgelist_path = "chip_topology.edgelist"
    output_dir    = "routed_dataset_remap"
    mapping_csv   = os.path.join(output_dir, "mapping_remap.csv")

    os.makedirs(output_dir, exist_ok=True)
    with open(mapping_csv, "w", newline="") as csvf:
        writer = csv.writer(csvf)
        # ✨✨✨ 修改点 1：更新CSV表头，增加 qubits 和 swaps 列 ✨✨✨
        writer.writerow(["filename", "qubits", "swaps", "is_isomorphic"])

    coupling_map = load_coupling_map(edgelist_path)

    total    = 0
    iso_count= 0
    q_stats  = {}

    for fname in sorted(os.listdir(qasm_dir)):
        if not fname.endswith(".qasm"):
            continue

        total += 1
        orig_path = os.path.join(qasm_dir, fname)
        qc_orig   = QuantumCircuit.from_qasm_file(orig_path)

        # 路由电路 (核心逻辑保持不变)
        qc_routed = transpile(
            qc_orig,
            coupling_map=coupling_map,
            layout_method='sabre',
            routing_method='sabre',
            optimization_level=0,
        )

        # ✨✨✨ 修改点 2：从编译后的电路中统计SWAP门的数量 ✨✨✨
        ops = qc_routed.count_ops()
        swap_count = ops.get('swap', 0)

        # 保存路由后的 QASM (逻辑保持不变)
        routed_path = os.path.join(output_dir, fname)
        with open(routed_path, 'w') as f:
            # (省略写入 layout 注释的代码)
            f.write(dumps(qc_routed))

        # 同构性检查 (逻辑保持不变)
        dag_orig   = circuit_to_dag(strip_metadata(qc_orig))
        dag_routed = circuit_to_dag(strip_metadata(qc_routed))
        G_orig   = build_graph_from_dag(dag_orig)
        G_routed = build_graph_from_dag(dag_routed)
        nm      = isomorphism.categorical_node_match('name', None)
        matcher = isomorphism.MultiDiGraphMatcher(G_orig, G_routed, node_match=nm)
        iso     = matcher.is_isomorphic()
        if iso:
            iso_count += 1

        # 更新按 q 统计 (逻辑保持不变)
        k = int(fname.split('_', 1)[0][1:])
        cnt, iso_cnt = q_stats.get(k, (0, 0))
        q_stats[k] = (cnt + 1, iso_cnt + (1 if iso else 0))

        # ✨✨✨ 修改点 3：将更详细的信息写入CSV ✨✨✨
        with open(mapping_csv, 'a', newline='') as csvf:
            writer = csv.writer(csvf)
            writer.writerow([fname, qc_orig.num_qubits, swap_count, iso])

        # ✨✨✨ 修改点 4：美化终端输出，增加列间距 ✨✨✨
        # 使用f-string的格式化功能，让输出像表格一样对齐
        print(f"{fname:<40} -> SWAPs: {swap_count:<4} | 同构: {'✅' if iso else '❌'}")

    # 总结输出 (逻辑保持不变)
    print(f"\n共处理 {total} 个电路，其中 {iso_count} 个保持同构。")
    print("按 qubit 数统计：")
    for k in sorted(q_stats):
        cnt, iso_cnt = q_stats[k]
        print(f"  q={k} 时，共处理 {cnt} 个电路，其中 {iso_cnt} 个保持同构")

if __name__ == '__main__':
    main()