import os
import re
import csv
from qiskit import QuantumCircuit

# --- 1. 配置区 ---

# ✨✨✨ 修改点 1：对比目标只保留 qmap_exact 和 heuristic_reindexed ✨✨✨
METHOD_DIRS = {
    'qmap_exact': 'qmap_exact_output',
    'heuristic_reindexed': 'combined_circuits_reindexed_qmap_heuristic'
}

# 统一的正则表达式，用于从文件名中解析出 (q_num, s_num)
unified_pattern = re.compile(r'^q(\d+)_s(\d+).*?\.qasm$')

# 存放所有找到的结果的字典
results = {}

# --- 2. 主逻辑 ---

def main():
    print("开始统计所有已定义方法输出电路中的 SWAP 门数量...")

    # --- 步骤 1：收集所有存在的结果 ---
    for method_name, folder_path in METHOD_DIRS.items():
        print(f"\n--- 正在扫描目录 '{folder_path}' (方法: '{method_name}') ---")
        
        if not os.path.isdir(folder_path):
            print(f"  - 警告: 目录 '{folder_path}' 不存在，已跳过。")
            continue

        for fname in sorted(os.listdir(folder_path)):
            match = unified_pattern.match(fname)
            if not match:
                continue

            q_num, s_num = map(int, match.groups())
            canonical_key = (q_num, s_num)

            path = os.path.join(folder_path, fname)
            try:
                qc = QuantumCircuit.from_qasm_file(path)
                ops = qc.count_ops()
                swap_count = ops.get('swap', 0)
            except Exception as e:
                print(f"  - 处理文件 {fname} 时出错: {e}")
                continue
            
            if canonical_key not in results:
                results[canonical_key] = {}
            results[canonical_key][method_name] = swap_count
            print(f"  - 文件 {fname} -> key: {canonical_key}, SWAPs: {swap_count}")

    # --- ✨✨✨ 修改点 2：添加过滤逻辑，只保留数据完整的实例 ✨✨✨
    print("\n--- 正在过滤数据，只保留所有方法都成功处理的实例 ---")
    
    all_methods = list(METHOD_DIRS.keys())
    total_methods_configured = len(all_methods)
    valid_keys = []

    for key, found_methods_dict in results.items():
        # 检查这个实例的结果数量是否和我们配置的方法数量一致
        if len(found_methods_dict) == total_methods_configured:
            valid_keys.append(key)
        else:
            missing_methods = set(all_methods) - set(found_methods_dict.keys())
            print(f"  - 实例 {key} 数据不完整，将被丢弃。缺失的方法: {', '.join(missing_methods)}")
            
    print(f"过滤完成。共有 {len(results)} 个实例被找到，其中 {len(valid_keys)} 个实例的数据是完整的。")
    # --- ✨ 过滤逻辑结束 ✨ ---


    # --- 步骤 3：保存过滤后的结果到 CSV 文件 ---
    output_file = 'swap_counts_exact_vs_heuristic.csv'
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # 动态生成表头
        header = ['q_large', 'instance_num'] + all_methods
        writer.writerow(header)
        
        # 只遍历包含完整数据的 valid_keys
        for key in sorted(valid_keys):
            q_num, s_num = key
            row = [q_num, s_num]
            for method_name in all_methods:
                row.append(results[key][method_name])
            writer.writerow(row)

    print(f"\n对比统计完成！结果已保存到 {output_file}")


if __name__ == '__main__':
    main()