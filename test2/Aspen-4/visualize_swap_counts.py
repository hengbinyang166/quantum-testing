import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# --- 1. 配置区 ---

# 定义要分析和绘图的芯片及其对应的CSV数据文件
CHIP_DATA = {
    'Tokyo': 'swap_counts_comparison_all_tokyo.csv',
    'Aspen-4': 'swap_counts_comparison_all_aspen.csv',
    'Sydney': 'swap_counts_comparison_all_sydney.csv'
}

# 定义内部列名到显示名称的映射
# “键”必须和CSV列标题完全一致，“值”是您想在图上显示的漂亮名字。
METHOD_DISPLAY_NAMES = {
    'qmap_exact': 'Exact',
    'final_reindexed': 'SABRE'
}
METHODS_TO_COMPARE = list(METHOD_DISPLAY_NAMES.keys())

# 定义与上述方法对应的显示颜色
METHOD_COLORS = ['skyblue', 'lightcoral']


def plot_per_chip_avg_by_qubits():
    """
    为每个芯片单独生成一张图。
    X轴为大电路比特数，Y轴为该比特数下所有实例的平均SWAP数。
    """
    print("开始为每个芯片生成按比特数统计的平均SWAP数对比图...")

    # --- ✨✨✨ 主逻辑：遍历每个芯片并为其生成一张图 ✨✨✨ ---
    for chip_name, csv_path in CHIP_DATA.items():
        print(f"\n{'='*50}")
        print(f"--- 正在处理芯片: {chip_name} ---")
        
        # --- 1. 加载数据 ---
        if not os.path.exists(csv_path):
            print(f"  - 错误: 找不到数据文件 '{csv_path}'，已跳过此芯片。")
            continue

        try:
            df = pd.read_csv(csv_path)
            df.replace('N/A', np.nan, inplace=True)
            for col in METHODS_TO_COMPARE:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

        except Exception as e:
            print(f"  - 读取或处理文件 {csv_path} 时发生错误: {e}")
            continue

        # --- 2. ✨✨✨核心改动：按'q_large'分组并计算平均值✨✨✨ ---
        # .groupby('q_large') 会把所有q_large相同的行分为一组
        # .mean() 会计算每组中我们感兴趣的列的平均值
        try:
            avg_df = df.groupby('q_large')[METHODS_TO_COMPARE].mean().reset_index()
            print("  - 数据按比特数分组计算平均值完成。")
            print(avg_df)
        except KeyError:
            print(f"  - 错误：CSV文件 '{csv_path}' 中缺少 'q_large' 列或指定的方法列，无法分组。")
            continue

        if avg_df.empty:
            print("  - 数据为空或无法计算平均值，跳过绘图。")
            continue

        # --- 3. 准备绘图参数 ---
        labels = avg_df['q_large'].astype(int).tolist() # X轴标签：3, 4, 5...
        x = np.arange(len(labels))
        num_methods = len(METHODS_TO_COMPARE)
        total_width = 0.8
        bar_width = total_width / num_methods
        
        fig, ax = plt.subplots(figsize=(16, 9))

        # --- 4. 动态绘制所有方法的柱状图 ---
        for i, method_name in enumerate(METHODS_TO_COMPARE):
            if method_name not in avg_df.columns: continue
            
            offset = (i - (num_methods - 1) / 2) * bar_width
            averages = avg_df[method_name]
            
            display_name = METHOD_DISPLAY_NAMES.get(method_name, method_name)
            color = METHOD_COLORS[i % len(METHOD_COLORS)]
            
            rects = ax.bar(x + offset, averages, bar_width, label=display_name, color=color)
            ax.bar_label(rects, padding=3, fmt='%.2f', fontsize=8, rotation=45)

        # --- 5. 美化和定制图形 ---
        try:
            plt.rcParams['font.sans-serif'] = ['SimHei']
            plt.rcParams['axes.unicode_minus'] = False
        except Exception:
            print("  - 警告：找不到中文字体'SimHei'。")

        # 动态设置标题
        ax.set_title(f'芯片架构 {chip_name}: 平均SWAP门数对比 (按比特数)', fontsize=18)
        ax.set_ylabel('平均SWAP门数量', fontsize=14)
        ax.set_xlabel('大电路活跃区比特数 (q_large)', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend(title="编译方法", fontsize=11)
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)
        ax.set_ylim(bottom=0)
        fig.tight_layout()

        # --- 6. 保存图形 ---
        output_filename = f'avg_swap_by_qubits_{chip_name}.png'
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"  - 绘图成功！对比图已保存为 '{output_filename}'")
        plt.close(fig) # 关闭当前图形，防止在循环中互相影响

    print(f"\n{'='*50}")
    print("所有芯片处理完毕。")


if __name__ == '__main__':
    plot_per_chip_avg_by_qubits()