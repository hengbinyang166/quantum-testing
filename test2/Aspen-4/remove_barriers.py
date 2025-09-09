import os

def remove_barriers_in_file(file_path: str):
    """
    读取一个 QASM 文件，移除所有包含 'barrier' 的行，
    然后用修改后的内容覆盖原文件。

    参数:
        file_path: QASM 文件的完整路径。
    """
    try:
        # 1. 以读取模式打开文件，将所有行读入内存
        with open(file_path, 'r') as f:
            all_lines = f.readlines()

        # 2. 创建一个新列表，只包含不含 'barrier' 的行
        processed_lines = [line for line in all_lines if 'barrier' not in line]

        # 3. 以写入模式重新打开同一个文件（这将清空原文件内容）
        with open(file_path, 'w') as f:
            # 4. 将处理过的、不含 barrier 的行写回文件
            f.writelines(processed_lines)
            
        return True # 表示成功
    except IOError as e:
        print(f"    - 读写文件时出错 {os.path.basename(file_path)}: {e}")
        return False # 表示失败

def main():
    """
    主函数，遍历指定目录并处理所有 QASM 文件。
    """
    # 获取当前脚本所在的目录
    base_dir = os.getcwd()
    
    # 定义需要处理的目录列表
    target_dirs = [
        'combined_circuits_physical', #
        'combined_circuits_reindexed' #
    ]

    print("开始从 QASM 文件中移除 barrier 指令...")

    # 遍历每个目标目录
    for dir_name in target_dirs:
        # 构建完整的目录路径
        full_dir_path = os.path.join(base_dir, dir_name)

        # 检查路径是否存在并且确实是一个目录
        if not os.path.isdir(full_dir_path):
            print(f"\n目录 '{dir_name}' 不存在，跳过。")
            continue
        
        print(f"\n--- 正在处理目录: {dir_name} ---")
        
        processed_count = 0
        # 遍历目录中的所有文件
        for fname in sorted(os.listdir(full_dir_path)):
            # 只处理 .qasm 文件
            if fname.endswith('.qasm'):
                file_path = os.path.join(full_dir_path, fname)
                if remove_barriers_in_file(file_path):
                    print(f"  - 已处理: {fname}")
                    processed_count += 1
        
        print(f"--- 目录 '{dir_name}' 处理完成，共更新 {processed_count} 个文件 ---")
        
    print("\n所有指定目录处理完毕。")

if __name__ == '__main__':
    main()