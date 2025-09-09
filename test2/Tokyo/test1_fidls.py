import os
import re
import sys
import csv
import json
import networkx as nx
from qiskit import QuantumCircuit
from qiskit.qasm2 import dumps

# 把 FiDLS 源码目录加入搜索路径（目录名为 filds）
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'fidls'))

# 选择 FiDLS 分支：G=贪心版，D=深度版
GVal = True
if GVal:
    from fidls_g import qct_old
    suffix = 'G'
else:
    from fidls_d import qct_old
    suffix = 'D'

# 导入 FiDLS 辅助函数
from utils import CreateCircuitFromQASM, ReducedCircuit, qubit_in_circuit

# 读取物理耦合图及预处理
edgelist_file = 'chip_topology.edgelist'
G_phys = nx.read_edgelist(edgelist_file, nodetype=int)
EG     = list(G_phys.edges()) + [(v, u) for u, v in G_phys.edges()]
V      = list(G_phys.nodes())
# 将 all_pairs_shortest_path_length 展开成扁平字典 SPL[(u,v)] = 距离
raw_SPL = dict(nx.all_pairs_shortest_path_length(G_phys))
SPL = { (u, v): d
        for u, dist_map in raw_SPL.items()
        for v, d in dist_map.items() }

QFilter = '01y'

# 加载 Top 初始映射（由 generate_top.py 生成）
arch_name = 'tokyo'
map_file = os.path.join(
    os.path.dirname(__file__),
    'fidls', 'inimap',
    f"{arch_name}_top.txt"
)
if not os.path.exists(map_file):
    raise FileNotFoundError(f"找不到 Top 初始映射文件：{map_file}")
with open(map_file, 'r', encoding='utf-8') as f:
    IM = json.load(f)
print(f"加载 Top 初始映射，共 {len(IM)} 条映射：{map_file}")

def strip_metadata(qc: QuantumCircuit) -> QuantumCircuit:
    """去除 barrier、measure、set_layout，只保留计算门。"""
    clean = QuantumCircuit(qc.num_qubits)
    for instr, qargs, cargs in qc.data:
        if instr.name not in ('barrier', 'measure', 'set_layout'):
            clean.append(instr, qargs, cargs)
    return clean

def file_key(name: str):
    """
    按文件名格式 q{qubit}_l{layout}_s{sample}.qasm，
    提取三段整数 (qubit, layout, sample) 作为排序键。
    """
    m = re.match(r'q(\d+)_l(\d+)_s(\d+)\.qasm$', name)
    if m:
        return (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    else:
        # 文件名不符合模式的，排到最后
        return (float('inf'), float('inf'), float('inf'))

def main():
    print(f"开始 FiDLS 路由 (初始映射=Top, 分支={suffix})…")
    qasm_dir   = 'combined_circuits_reindexed'
    output_dir = f'combined_circuits_{suffix}_reindexed_fidls'
    csv_path   = os.path.join(output_dir, f'mapping_remap_{suffix}.csv')

    os.makedirs(output_dir, exist_ok=True)
    # 写 CSV 表头
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            '文件名','状态','耗时(s)',
            '原始CNOT数','路由后CNOT数','Swap插入数'
        ])

    total = iso_count = 0
    stats = {}

    # 自然数字顺序加载 .qasm 文件
    files = sorted(
        [fn for fn in os.listdir(qasm_dir) if fn.endswith('.qasm')],
        key=file_key
    )
    if not files:
        print(f"*** 未在目录 '{qasm_dir}' 找到 .qasm 文件 ***")
        return

    for idx, fname in enumerate(files, start=1):
        total += 1

        # 1) 读取原始电路 & 提取纯 CNOT 列表
        cir = CreateCircuitFromQASM(fname, qasm_dir + os.sep)
        C   = ReducedCircuit(cir)
        Q   = qubit_in_circuit(list(range(len(C))), C)

        # 2) 构造 Top 初始映射 tau（物理→逻辑）
        tau = [-1] * len(V)
        for logic_q, phys_q in IM[idx-1][1]:
            tau[phys_q] = logic_q

        # 3) 调用 FiDLS 得到路由后 CNOT 序列 & 耗时
        C_out, cost = qct_old(tau, C, Q, G_phys, EG, V, SPL, QFilter)

        # 4) 构建路由后电路，展开 CNOT 并统计 Swap
        qc_routed   = QuantumCircuit(len(V))
        routed_flat = []
        swap_count  = 0
        i2 = 0
        # C_out 中元素是列表 [ctl, tgt]
        while i2 < len(C_out):
            ctl, tgt = C_out[i2]
            # 检测列表模式 Swap = [ctl,tgt], [tgt,ctl], [ctl,tgt]
            if (i2 + 2 < len(C_out)
                and C_out[i2]   == [ctl, tgt]
                and C_out[i2+1] == [tgt, ctl]
                and C_out[i2+2] == [ctl, tgt]):
                qc_routed.swap(ctl, tgt)
                # 展开成 3 条 CNOT
                routed_flat.extend([(ctl, tgt), (tgt, ctl), (ctl, tgt)])
                swap_count += 1
                i2 += 3
            else:
                qc_routed.cx(ctl, tgt)
                routed_flat.append((ctl, tgt))
                i2 += 1

        # 5) 原始 CNOT 列表
        original_cx = [tuple(p) for p in C]

        # 6) 判断同构与状态
        diff = len(routed_flat) - len(original_cx)
        if diff == 0:
            status = '同构'
            iso = True
        elif diff % 3 == 0:
            status = f'非同构（插入 swap {swap_count} 次）'
            iso = False
        else:
            status = f'异常（新增 CNOT {diff} 次，非3的倍数）'
            iso = False
        if iso:
            iso_count += 1

        # 统计按 qubit 数分组
        qn = cir.num_qubits
        cnt, ic = stats.get(qn, (0, 0))
        stats[qn] = (cnt+1, ic + int(iso))

        # 7) 写出路由后 QASM
        out_path = os.path.join(output_dir, fname)
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(dumps(qc_routed))

        # 8) 写 CSV & 打印中文日志
        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow([
                fname, status, round(cost, 3),
                len(original_cx), len(routed_flat), swap_count
            ])
        print(f"文件 {fname}：{status}，原CNOT={len(original_cx)}，"
              f"路由后CNOT={len(routed_flat)}，Swap={swap_count}，耗时={cost:.3f}s")

    # 最终汇总
    print(f"\n共处理 {total} 个电路，其中 {iso_count} 个同构。")
    print("按量子比特数统计同构：")
    for qn in sorted(stats):
        cnt, ic = stats[qn]
        print(f"  q={qn}: {ic}/{cnt} 同构")

if __name__ == '__main__':
    main()