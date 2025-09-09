import os
from qiskit import QuantumCircuit
from qiskit.qasm2 import dumps as qasm2_dumps
from qiskit.providers.fake_provider import GenericBackendV2
from mqt import qmap

def map_with_qmap_heuristic(
    qc: QuantumCircuit, 
    backend: GenericBackendV2, 
    output_path: str,
    initial_layout: str = 'dynamic'
):
    """
    对输入电路应用 MQT QMAP 的 heuristic 方法进行编译。
    """
    # ✨✨✨ 核心修改点：采纳您的建议 ✨✨✨
    # 在这里添加 add_measurements_to_mapped_circuit=False 参数
    routed_qc, results = qmap.compile(
        qc,
        arch=backend,
        method='heuristic',
        initial_layout=initial_layout,
        add_measurements_to_mapped_circuit=False # 直接告诉QMAP不要添加测量操作
    )
    
    with open(output_path, 'w') as f:
        f.write(qasm2_dumps(routed_qc))


def main():
    base_dir = os.getcwd()
    
    datasets = {
        'combined_circuits_physical': {
            'output_name': 'combined_circuits_physical_qmap_heuristic',
            'initial_layout': 'identity'
        },
        'combined_circuits_reindexed': {
            'output_name': 'combined_circuits_reindexed_qmap_heuristic',
            'initial_layout': 'dynamic'
        }
    }

    print("正在加载芯片拓扑并创建 Backend 对象...")
    cmap_path = os.path.join(base_dir, 'chip_topology.edgelist')
    if not os.path.isfile(cmap_path):
        print(f"错误: 耦合图文件 '{cmap_path}' 不存在。")
        return
        
    edges = []
    with open(cmap_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'): continue
            try:
                u, v = line.split()
                edges.append((int(u), int(v)))
            except ValueError:
                print(f"警告: 无法解析行 '{line}'，跳过。")
                continue
    
    if not edges:
        print("错误：耦合图文件为空或无法解析。")
        return

    num_qubits = max(max(edge) for edge in edges) + 1

    try:
        # 保持双向连接以避免不必要的H门合成
        bidirectional_edges = edges + [(v, u) for u, v in edges]
        backend = GenericBackendV2(
            num_qubits=num_qubits, 
            coupling_map=bidirectional_edges,
            basis_gates=['id', 'rz', 'sx', 'x', 'cx']
        )
        print(f"已创建自定义 Backend 对象：{num_qubits} 比特, {len(edges)} 条物理连接。")

    except Exception as e:
        print(f"创建 Backend 对象时出错: {e}")
        return

    # 循环处理数据集
    for input_name, config in datasets.items():
        output_name = config['output_name']
        layout_strategy = config['initial_layout']
        
        input_dir = os.path.join(base_dir, input_name)
        output_dir = os.path.join(base_dir, output_name)
        
        os.makedirs(output_dir, exist_ok=True)

        if not os.path.isdir(input_dir):
            print(f"\n输入目录不存在，跳过：{input_dir}")
            continue
            
        print(f"\n--- 开始对目录 '{input_name}' 中的电路执行 QMAP Heuristic 路由 ---")
        print(f"    输出目录: '{output_name}'")
        print(f"    布局策略: '{layout_strategy}'")

        file_count = 0
        error_count = 0
        for fname in sorted(os.listdir(input_dir)):
            if not fname.endswith('.qasm'):
                continue
            
            input_path = os.path.join(input_dir, fname)
            out_path = os.path.join(output_dir, fname)
            
            try:
                qc = QuantumCircuit.from_qasm_file(input_path)
                
                # 保持电路净化逻辑，以移除 barrier 等指令，让算法发挥最佳性能
                qc_pure = QuantumCircuit(qc.num_qubits)
                for instruction in qc.data:
                    if instruction.operation.name != 'barrier' and not instruction.clbits:
                         qc_pure.append(instruction)

                # 将纯量子电路和对应的布局策略传入
                map_with_qmap_heuristic(qc_pure, backend, out_path, initial_layout=layout_strategy)
                
                print(f"  - 已路由: {fname}")
                file_count += 1
            except Exception as e:
                print(f"  - 处理文件 {fname} 时出错: {type(e).__name__} - {e}")
                error_count += 1

        print(f"--- 目录 '{input_name}' 处理完成，共路由 {file_count} 个文件，失败 {error_count} 个 ---")

if __name__ == '__main__':
    main()