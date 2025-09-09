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
        optimization_level=0,    # 不做额外优化
        #basis_gates=['x', 'y', 'z', 'cx', 'swap']
    )
    # 将 QASM 写入文件
    with open(output_path, 'w') as f:
        f.write(qasm2_dumps(routed))


def main():
    # 获取当前工作目录
    base_dir = os.getcwd()
    
    # 定义主输出/输入目录
    DEMO_OUTPUT_DIR = 'demo_output'

    # 定义输入目录和对应的输出目录列表 (相对于 DEMO_OUTPUT_DIR)
    datasets = [
        # 处理物理版电路
        ('combined_circuits_physical','combined_circuits_physical_sabre'),
        # 处理重索引版电路
        ('combined_circuits_reindexed','combined_circuits_reindexed_sabre')
    ]

    # --- 您要求确认的代码块 ---
    # 仅加载一次全芯片的耦合图
    print("正在加载全芯片耦合图...")
    # 耦合图文件位于项目根目录
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
    
    # 注意：CouplingMap 会自动处理双向性
    cmap = CouplingMap(edges)
    print(f"已加载耦合图：{cmap_path}")
    # --- 代码块结束 ---

    # 遍历每组数据集目录并执行路由
    for input_name, output_name in datasets:
        # 所有路径都基于 DEMO_OUTPUT_DIR
        input_dir = os.path.join(base_dir, DEMO_OUTPUT_DIR, input_name)
        output_dir = os.path.join(base_dir, DEMO_OUTPUT_DIR, output_name)
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

        if not os.path.isdir(input_dir):
            print(f"\n输入目录不存在，跳过：{input_dir}")
            continue
            
        print(f"\n--- 开始对目录 '{input_name}' 中的电路执行 SABRE 路由 ---")
        print(f"    结果将保存到 '{output_name}'")

        # 处理该目录下所有 QASM 文件
        file_count = 0
        for fname in sorted(os.listdir(input_dir)):
            if not fname.endswith('.qasm'):
                continue
            
            input_path = os.path.join(input_dir, fname)
            out_path = os.path.join(output_dir, fname)
            
            try:
                # 从 QASM 文件加载电路
                qc = QuantumCircuit.from_qasm_file(input_path)
                # 执行路由并保存
                route_with_sabre(qc, cmap, out_path)
                print(f"  - 已路由: {fname}")
                file_count += 1
            except Exception as e:
                print(f"  - 处理文件 {fname} 时出错: {e}")

        print(f"--- 目录 '{input_name}' 处理完成，共路由 {file_count} 个文件 ---")

if __name__ == '__main__':
    main()