import os
import re
import csv
from qiskit import QuantumCircuit

# --- 1. 配置区 (已修改) ---

# ✨✨✨ 修改点 1：在这里定义您所有想对比的方法和它们所在的目录 ✨✨✨
# “键”是您为方法取的名字（会成为CSV的列名），“值”是存放QASM文件的目录路径。
METHOD_DIRS = {
    'qmap_exact': 'qmap_exact_output',
    'final_reindexed': 'combined_circuits_reindexed_sabre'
    # 如果您有其他方法，比如用Sabre处理后的结果，可以在这里继续添加
    # 'sabre_reindexed': 'path/to/sabre/output'
}

# ✨✨✨ 修改点 2：使用一个统一的正则表达式来匹配所有文件名 ✨✨✨
# 这个模式会从 "q13_s2_combined_phys.qasm" 或 "q13_s2_qmap_exact.qasm" 中提取 (13, 2)
unified_pattern = re.compile(r'^q(\d+)_s(\d+).*?\.qasm$')

# 存放最终结果的字典
results = {}

# --- 2. 主逻辑 (已修改) ---

def main():
    print("开始统计所有已定义方法输出电路中的 SWAP 门数量...")

    # --- 统一处理所有目录 ---
    for method_name, folder_path in METHOD_DIRS.items():
        print(f"\n--- 正在统计目录 '{folder_path}' (方法: '{method_name}') ---")
        
        if not os.path.isdir(folder_path):
            print(f"  - 错误: 目录 '{folder_path}' 不存在，已跳过。")
            continue

        for fname in sorted(os.listdir(folder_path)):
            # ✨✨✨ 修改点 3：应用统一的解析逻辑 ✨✨✨
            match = unified_pattern.match(fname)
            if not match:
                # 如果需要，可以取消下面这行注释来查看不匹配的文件名
                # print(f"  - 文件名格式不匹配，跳过: {fname}")
                continue

            # 提取出的 (q_large, s_num) 作为标准键
            q_num, s_num = map(int, match.groups())
            canonical_key = (q_num, s_num)

            path = os.path.join(folder_path, fname)
            try:
                # 从QASM文件加载电路并统计SWAP门
                qc = QuantumCircuit.from_qasm_file(path)
                ops = qc.count_ops()
                swap_count = ops.get('swap', 0)
            except Exception as e:
                print(f"  - 处理文件 {fname} 时出错: {e}")
                continue
            
            # 将结果存入字典
            if canonical_key not in results:
                results[canonical_key] = {}
            results[canonical_key][method_name] = swap_count
            print(f"  - 文件 {fname} -> key: {canonical_key}, SWAPs: {swap_count}")

    # --- 3. 保存结果到 CSV 文件 ---
    output_file = 'swap_counts_comparison_all.csv'
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # ✨✨✨ 修改点 4：动态生成表头 ✨✨✨
        header = ['q_large', 'instance_num'] + list(METHOD_DIRS.keys())
        writer.writerow(header)
        
        # 按 (q_large, instance_num) 排序后写入数据
        for key in sorted(results.keys()):
            q_num, s_num = key
            row = [q_num, s_num]
            for method_name in METHOD_DIRS.keys():
                row.append(results[key].get(method_name, 'N/A')) # 如果某个方法没有对应文件，则填'N/A'
            writer.writerow(row)

    print(f"\n对比统计完成！结果已保存到 {output_file}")


if __name__ == '__main__':
    main()