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
    # qasm_dir = "quantum_dataset"
    # 改成用重映射后的电路（编号从 0 开始）
    qasm_dir      = "quantum_dataset_remap"
    edgelist_path = "chip_topology.edgelist"
    # output_dir = "routed_dataset"
    # 改成单独的路由输出目录
    output_dir    = "routed_dataset_remap"
    # mapping_csv = os.path.join(output_dir, "mapping.csv")
    # 改成对应的 mapping 文件
    mapping_csv   = os.path.join(output_dir, "mapping_remap.csv")

    os.makedirs(output_dir, exist_ok=True)
    with open(mapping_csv, "w", newline="") as csvf:
        writer = csv.writer(csvf)
        writer.writerow(["filename", "original_file", "is_isomorphic"])

    coupling_map = load_coupling_map(edgelist_path)

    # —— 新增：按 q 统计 —— #
    total    = 0
    iso_count= 0
    q_stats  = {}  # { q: [total_count, iso_count] }

    for fname in os.listdir(qasm_dir):
        if not fname.endswith(".qasm"):
            continue

        total += 1
        orig_path = os.path.join(qasm_dir, fname)
        qc_orig   = QuantumCircuit.from_qasm_file(orig_path)

        # 路由电路
        qc_routed = transpile(
            qc_orig,
            coupling_map=coupling_map,
            layout_method='sabre',  #basic这里就改为trivial sabre这里就改为sabre
            routing_method='sabre',  #basic这里改为basic sabre这里就改为sabre
            optimization_level=0,
            # basis_gates=['x', 'y', 'z', 'cx', 'swap']
        )

        # 保存路由后的 QASM
        routed_path = os.path.join(output_dir, fname)
        with open(routed_path, 'w') as f:

            # —— 在文件头插入逻辑 qubit → 物理 qubit 的映射注释 —— #
            transp_layout = qc_routed.layout
            final_layout  = getattr(transp_layout, 'final_layout', transp_layout)
            try:
                phys_bits = final_layout.get_physical_bits()
            except AttributeError:
                v2p = getattr(final_layout, '_v2p', None)
                if v2p is None:
                    raise RuntimeError("无法获取物理比特映射，请检查 Qiskit 版本")
                phys_bits = [v2p[q] for q in qc_routed.qubits]
            layout_map = {f"q[{i}]": phys_bits[i] for i in range(len(phys_bits))}
            f.write(f"// layout: {layout_map}\n")
            # —— 以上注释写出每个逻辑 qubit 对应的物理编号 —— #

            f.write(dumps(qc_routed))

        # 构建 DAG 并移除元数据
        dag_orig   = circuit_to_dag(strip_metadata(qc_orig))
        dag_routed = circuit_to_dag(strip_metadata(qc_routed))

        # 使用 topological_op_nodes 构建图
        G_orig   = build_graph_from_dag(dag_orig)
        G_routed = build_graph_from_dag(dag_routed)

        # 使用 VF2 比较图的同构性，仅按节点的 name 属性匹配
        nm      = isomorphism.categorical_node_match('name', None)
        matcher = isomorphism.MultiDiGraphMatcher(G_orig, G_routed, node_match=nm)
        iso     = matcher.is_isomorphic()
        if iso:
            iso_count += 1

        # —— 新增：更新 q_stats —— #
        # 从文件名中解析 q=k
        k = int(fname.split('_', 1)[0][1:])
        cnt, iso_cnt = q_stats.get(k, (0, 0))
        cnt += 1
        if iso:
            iso_cnt += 1
        q_stats[k] = (cnt, iso_cnt)
        # —— 新增结束 —— #

        # 写入结果
        with open(mapping_csv, 'a', newline='') as csvf:
            writer = csv.writer(csvf)
            writer.writerow([fname, fname, iso])

        print(f"{fname}: {'同构✅' if iso else '不等构❌'}")

    # 总结输出
    print(f"\n共处理 {total} 个电路，其中 {iso_count} 个保持同构。")

    # —— 新增：按 q 分组输出 —— #
    print("按 qubit 数统计：")
    for k in sorted(q_stats):
        cnt, iso_cnt = q_stats[k]
        print(f"  q={k} 时，共处理 {cnt} 个电路，其中 {iso_cnt} 个保持同构")
    # —— 新增结束 —— #

if __name__ == '__main__':
    main()