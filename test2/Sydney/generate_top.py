import os
import sys
import json
import re
import networkx as nx

# 将 FiDLS 源码目录加入搜索路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'fidls'))

# 导入 FiDLS 初始映射方法
from inimap import _tau_bstg_
# 导入电路和映射辅助函数
from utils import CreateCircuitFromQASM, ReducedCircuit
# 导入物理架构定义
from ag import ArchitectureGraph, sydney

# --- 排序函数：按 q{qubit}_l{layout}_s{sample}.qasm 格式排序 ---
def file_key(name: str):
    m = re.match(r'q(\d+)_l(\d+)_s(\d+)\.qasm$', name)
    if m:
        return (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    return (float('inf'),) * 3


def main():
    # 1. 读取物理耦合图
    AG = ArchitectureGraph(sydney())
    G_phys = AG.graph

    # 2. 准备待映射电路文件列表
    qasm_dir = 'combined_circuits_reindexed'
    files = sorted(
        [fn for fn in os.listdir(qasm_dir) if fn.endswith('.qasm')],
        key=file_key
    )
    if not files:
        print(f"*** 未在目录 '{qasm_dir}' 找到 .qasm 文件 ***")
        return

    mapping_list = []
    # 时间限制参数，可根据需要调整
    anchor = True
    stop = 100  # 最长映射时间 (秒)

    # 3. 逐电路生成 Top 初始映射
    for idx, fname in enumerate(files, start=1):
        # 3.1 读取电路并提取纯 CNOT 列表
        cir = CreateCircuitFromQASM(fname, qasm_dir + os.sep)
        C   = ReducedCircuit(cir)

        # 3.2 调用 FiDLS 提供的 topgraph 初始映射方法
        # 返回值为 dict：逻辑比特 -> 物理比特
        if len(C) == 0:
            tau_dict = {}
        else:
            tau_dict = _tau_bstg_(C, G_phys, anchor, stop)
        # 转换为列表形式 [(logic, phys), ...]
        pairs = list(tau_dict.items())
        mapping_list.append([idx, pairs])

        print(f"生成映射 {idx}/{len(files)}: 电路 {fname} → 映射对 {len(pairs)}")

    # 4. 写出初始映射文件
    out_dir = os.path.join('fidls', 'inimap')
    os.makedirs(out_dir, exist_ok=True)
    map_file = os.path.join(out_dir, 'sydney_top.txt')
    with open(map_file, 'w', encoding='utf-8') as f:
        json.dump(mapping_list, f, indent=2, ensure_ascii=False)

    print(f"\n✔ 完成：已生成 Top 初始映射文件 {map_file}")

if __name__ == '__main__':
    main()