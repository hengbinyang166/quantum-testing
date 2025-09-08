import os
import csv
import time
import json
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.transpiler import CouplingMap
from qiskit.qasm2 import dumps
from qiskit.providers.fake_provider import GenericBackendV2
import networkx as nx
from networkx.algorithms import isomorphism
from mqt import qmap
from typing import Union

# ---------------- 配置区 ----------------
DATASET_DIR = "quantum_dataset_remap"
EDGE_LIST = "chip_topology.edgelist"
OUTPUT_DIR = "routed_dataset_qmap"
MAPPING_CSV = os.path.join(OUTPUT_DIR, "mapping_remap.csv")
ERROR_CSV = os.path.join(OUTPUT_DIR, "compile_errors.csv")
STATS_JSON = os.path.join(OUTPUT_DIR, "mapping_info.json")
FAILED_LIST = os.path.join(OUTPUT_DIR, "failed_files.txt")

# ---------------- 辅助函数 ----------------
def strip_metadata(qc: QuantumCircuit) -> QuantumCircuit:
    """移除电路中的元数据指令"""
    clean = QuantumCircuit(qc.num_qubits)
    for instr in qc.data:
        op = instr.operation
        if op.name not in ('barrier', 'measure', 'set_layout'):
            clean.append(op, instr.qubits, instr.clbits)
    return clean


def build_graph_from_dag(dag):
    nodes = list(dag.topological_op_nodes())
    G = nx.MultiDiGraph()
    for node in nodes:
        G.add_node(node, name=node.name)
    for node in nodes:
        for succ in dag.successors(node):
            if succ in nodes:
                G.add_edge(node, succ)
    return G


def get_layout_map(res, num_qubits):
    if hasattr(res, 'mapping'):
        layout_map = res.mapping
    elif hasattr(res, 'layout'):
        layout_map = res.layout
    else:
        return {i: i for i in range(num_qubits)}
    if isinstance(layout_map, list):
        return {logical: physical for logical, physical in enumerate(layout_map)}
    if isinstance(layout_map, dict) and all(isinstance(k, int) for k in layout_map.keys()):
        return layout_map
    try:
        return {logical: physical for physical, logical in layout_map.items()}
    except Exception:
        return {i: i for i in range(num_qubits)}


def safe_get_qasm(circuit):
    try:
        return circuit.qasm()
    except Exception:
        try:
            return dumps(circuit)
        except Exception:
            return str(circuit)


def compile_with_qmap(
    circuit: Union[str, QuantumCircuit],
    arch: GenericBackendV2,
    num_qubits: int
) -> (str, dict, int, bool):
    """
    使用 QMAP 编译电路，返回 (qasm_str, layout_map, swap_count, routed_ok)。
    如遇错误，回退到原始电路，swap_count 返回 0，routed_ok 返回 False。
    """
    try:
        # 调用 QMAP 编译
        if isinstance(circuit, QuantumCircuit):
            result = qmap.compile(
                circuit,
                arch=arch,
                method="heuristic",
                heuristic="gate_count_max_distance",
                initial_layout="dynamic",
                lookahead_heuristic="gate_count_max_distance",
                encoding="commander",
                commander_grouping="fixed3",
                swap_reduction="coupling_limit",
                pre_mapping_optimizations=False,
                post_mapping_optimizations=False,
                add_measurements_to_mapped_circuit=False
            )
        elif isinstance(circuit, str) and os.path.exists(circuit):
            with open(circuit, 'r') as f:
                qasm_content = f.read()
            result = qmap.compile(
                qasm_content,
                arch=arch,
                method="heuristic",
                heuristic="gate_count_max_distance",
                initial_layout="dynamic",
                lookahead_heuristic="gate_count_max_distance",
                encoding="commander",
                commander_grouping="fixed3",
                swap_reduction="coupling_limit",
                pre_mapping_optimizations=False,
                post_mapping_optimizations=False,
                add_measurements_to_mapped_circuit=False
            )
        else:
            result = qmap.compile(
                str(circuit),
                arch=arch,
                method="heuristic",
                heuristic="gate_count_max_distance",
                initial_layout="dynamic",
                lookahead_heuristic="gate_count_max_distance",
                encoding="commander",
                commander_grouping="fixed3",
                swap_reduction="coupling_limit",
                pre_mapping_optimizations=False,
                post_mapping_optimizations=False,
                add_measurements_to_mapped_circuit=False
            )
        if isinstance(result, tuple) and len(result) >= 2:
            qc_routed, res = result
        else:
            raise ValueError(f"未知返回类型: {type(result)}")
        qasm_str = safe_get_qasm(qc_routed)
        layout_map = get_layout_map(res, num_qubits)
        # 统计 swap 数量（仅参考，如需精确从文件读取）
        if hasattr(res, 'swaps'):
            initial_swaps = res.swaps
        elif isinstance(res, dict) and 'swaps' in res:
            initial_swaps = res['swaps']
        else:
            initial_swaps = 0
        return qasm_str, layout_map, initial_swaps, True
    except Exception as e:
        print(f"QMAP 编译失败，回退原电路: {e}")
        # 回退到原始电路
        if isinstance(circuit, QuantumCircuit):
            qasm_str = safe_get_qasm(circuit)
        elif isinstance(circuit, str) and os.path.exists(circuit):
            with open(circuit, 'r') as f:
                qasm_str = f.read()
        else:
            qasm_str = str(circuit)
        return qasm_str, {i: i for i in range(num_qubits)}, 0, False


def perform_isomorphism_check(qc_orig, qasm_str):
    try:
        qc_routed = QuantumCircuit.from_qasm_str(qasm_str)
        dag_orig = circuit_to_dag(strip_metadata(qc_orig))
        dag_routed = circuit_to_dag(strip_metadata(qc_routed))
        G_orig = build_graph_from_dag(dag_orig)
        G_routed = build_graph_from_dag(dag_routed)
        nm = isomorphism.categorical_node_match('name', None)
        matcher = isomorphism.MultiDiGraphMatcher(G_orig, G_routed, node_match=nm)
        return matcher.is_isomorphic()
    except Exception:
        return False

# ---------------- 主流程 ----------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    routed_qasm_dir = os.path.join(OUTPUT_DIR, "routed_qasm")
    os.makedirs(routed_qasm_dir, exist_ok=True)

    with open(MAPPING_CSV, 'w', newline='') as mc, open(ERROR_CSV, 'w', newline='') as ec:
        csv.writer(mc).writerow(["文件名", "成功", "同构", "交换门数"])
        csv.writer(ec).writerow(["文件名", "错误信息"])

    coupling_list = []
    # 读取无向耦合图，并添加双向边以支持bidirectional CNOT
    with open(EDGE_LIST, 'r') as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                u, v = map(int, line.split())
                coupling_list.append((u, v))
                coupling_list.append((v, u))
    coupling_map = CouplingMap(coupling_list)
    arch = GenericBackendV2(num_qubits=coupling_map.size(), coupling_map=coupling_map.get_edges())

    total, iso_count = 0, 0
    q_stats, failed_files, all_records = {}, [], []

    for fname in os.listdir(DATASET_DIR):
        if not fname.endswith('.qasm'):
            continue
        total += 1
        orig_path = os.path.join(DATASET_DIR, fname)
        routed_path = os.path.join(routed_qasm_dir, fname)
        start = time.time()
        success, iso = False, False
        swap_count = 0

        # 编译并标记是否成功路由
        qc_orig = None
        try:
            qc_orig = QuantumCircuit.from_qasm_file(orig_path)
            qasm_str, layout_map, _, routed_ok = compile_with_qmap(qc_orig, arch, qc_orig.num_qubits)
        except Exception as e:
            routed_ok = False
            print(f"加载或编译失败：{fname} -> {e}")

        # 仅在路由成功时保存并统计
        if routed_ok:
            with open(routed_path, 'w') as wf:
                wf.write(f"// initial_layout: {layout_map}\n")
                wf.write(qasm_str)
            # 从文件读取统计 swap
            with open(routed_path, 'r') as sf:
                swap_count = sf.read().count('swap ')
            iso = perform_isomorphism_check(qc_orig, qasm_str)
            success = True
        else:
            # 路由失败：记录错误且不写输出文件
            swap_count = 0
            success = False
            iso = False
            failed_files.append(fname)
            csv.writer(open(ERROR_CSV, 'a', newline='')).writerow([fname, '路由失败'])

        elapsed = time.time() - start
        # 终端输出
        print(f"{fname}：成功={success}， 同构={iso}， 交换门数={swap_count}， 用时={elapsed:.2f}s")
        # CSV 记录
        csv.writer(open(MAPPING_CSV, 'a', newline='')).writerow([fname, success, iso, swap_count])

        # 更新统计
        if success and fname.startswith('q'):
            k = int(fname.split('_', 1)[0][1:])
            cnt, ic = q_stats.get(k, (0, 0))
            q_stats[k] = (cnt + 1, ic + int(iso))
            iso_count += int(iso)

        all_records.append({"文件名": fname, "成功": success, "同构": iso, "交换门数": swap_count, "用时": round(elapsed, 2), "布局": layout_map if routed_ok else {}})

    # 保存统计 JSON
    with open(STATS_JSON, 'w', encoding='utf-8') as jf:
        json.dump({"总电路数": total, "同构数": iso_count, "失败数": len(failed_files), "各量子位统计": q_stats, "记录": all_records}, jf, ensure_ascii=False, indent=2)

    # 保存失败文件列表
    with open(FAILED_LIST, 'w') as ff:
        ff.writelines(f + "\n" for f in failed_files)

    print(f"共处理 {total} 个电路，其中 {iso_count} 个保持同构。")
    print("按 qubit 数统计：")
    for k in sorted(q_stats):
        cnt, ic = q_stats[k]
        print(f"  q={k} 时，共处理 {cnt} 个电路，其中 {ic} 个保持同构")

if __name__ == '__main__':
    main()