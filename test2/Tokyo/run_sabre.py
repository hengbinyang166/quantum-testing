import os
from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import CouplingMap
from qiskit.qasm2 import dumps as qasm2_dumps

def route_with_sabre(qc: QuantumCircuit, coupling_map: CouplingMap, output_path: str):
    """
    对输入电路应用 SABRE 布局和路由，并将结果以 QASM 格式写入 output_path。
    参数：
        qc: QuantumCircuit 对象，待路由电路
        coupling_map: 耦合图对象，用于约束路由
        output_path: 路由后 QASM 文件保存路径
    """
    # 调用 qiskit 的 transpile 接口进行 SABRE 路由
    routed = transpile(
        qc,
        coupling_map=coupling_map,
        layout_method='sabre',   # 布局算法使用 SABRE
        routing_method='sabre',  # 路由算法使用 SABRE
        optimization_level=0     # 不做额外优化，专注于布局和路由
    )
    # 将 QASM 写入文件
    with open(output_path, 'w') as f:
        f.write(qasm2_dumps(routed))


def main():
    # 获取当前工作目录
    base_dir = os.getcwd()
    
    datasets = [
        ('combined_circuits_physical', 'combined_circuits_physical_sabre'),
        ('combined_circuits_reindexed', 'combined_circuits_reindexed_sabre')
    ]

    print("正在加载全芯片耦合图...")
    cmap_path = os.path.join(base_dir, 'chip_topology.edgelist')
    if not os.path.isfile(cmap_path):
        print(f"错误: 耦合图文件 '{cmap_path}' 不存在。")
        return
        
    edges = []
    with open(cmap_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            try:
                u, v = line.split()
                edges.append((int(u), int(v)))
            except ValueError:
                print(f"警告: 无法解析行 '{line}'，跳过。")
                continue
    
    # --- ✨ 修改点 1：创建双向耦合图 ✨ ---
    # 这可以防止因 CNOT 方向问题而自动合成 H 门。
    bidirectional_edges = edges + [(v, u) for u, v in edges]
    cmap = CouplingMap(bidirectional_edges)
    print(f"已加载双向耦合图：{cmap_path}")

    # 遍历每组数据集目录并执行路由
    for input_name, output_name in datasets:
        input_dir = os.path.join(base_dir, input_name)
        output_dir = os.path.join(base_dir, output_name)
        
        os.makedirs(output_dir, exist_ok=True)

        if not os.path.isdir(input_dir):
            print(f"\n输入目录不存在，跳过：{input_dir}")
            continue
            
        print(f"\n--- 开始对目录 '{input_name}' 中的电路执行 SABRE 路由 ---")
        print(f"    结果将保存到 '{output_name}'")

        file_count = 0
        error_count = 0
        for fname in sorted(os.listdir(input_dir)):
            if not fname.endswith('.qasm'):
                continue
            
            input_path = os.path.join(input_dir, fname)
            out_path = os.path.join(output_dir, fname)
            
            try:
                # 从 QASM 文件加载电路
                qc = QuantumCircuit.from_qasm_file(input_path)

                # --- ✨ 修改点 2：净化电路，移除 barrier 和经典操作 ✨ ---
                # 这能让 SABRE 算法发挥全部性能。
                qc_pure = QuantumCircuit(qc.num_qubits)
                for instruction in qc.data:
                    # 只复制纯粹的量子指令（过滤掉 barrier 和经典操作）
                    if instruction.operation.name != 'barrier' and not instruction.clbits:
                         qc_pure.append(instruction)
                # --- ✨ 净化逻辑结束 ✨ ---

                # 使用净化后的 qc_pure 对象进行路由
                route_with_sabre(qc_pure, cmap, out_path)
                
                print(f"  - 已路由: {fname}")
                file_count += 1
            except Exception as e:
                print(f"  - 处理文件 {fname} 时出错: {type(e).__name__} - {e}")
                error_count += 1

        print(f"--- 目录 '{input_name}' 处理完成，共路由 {file_count} 个文件，失败 {error_count} 个 ---")

if __name__ == '__main__':
    main()