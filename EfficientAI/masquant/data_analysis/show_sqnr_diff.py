# coding=utf-8 
# Copyright (c) 2025, Alibaba Cloud and its affiliates;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pandas as pd

def count_file1_greater_than_file2_detailed(file1_path, file2_path):
    """
    读取两个行数和列数相同的 CSV 文件的最后一列，
    统计并打印出第一个文件严格大于第二个文件的具体行信息。

    Args:
        file1_path (str): 第一个 CSV 文件的路径。
        file2_path (str): 第二个 CSV 文件的路径。
    
    Returns:
        int: 第一个文件最后一列大于第二个文件对应行最后一列的行数。
    """
    try:
        # 1. 读取 CSV 文件
        # 假设文件没有索引列，并使用第一行作为列头
        df1 = pd.read_csv(file1_path)
        df2 = pd.read_csv(file2_path)
        
        # 2. 确定最后一列的名称和数据
        last_col_name1 = df1.columns[-1]
        last_col_name2 = df2.columns[-1]
        
        # 提取最后一列，并确保数据类型是数值
        # errors='coerce' 会将无法转换为数字的值变成 NaN (这些行将被忽略比较)
        col1_values = pd.to_numeric(df1[last_col_name1], errors='coerce')
        col2_values = pd.to_numeric(df2[last_col_name2], errors='coerce')

        # 检查和对齐行数 (如果用户要求相同，我们只做警告和切片)
        min_len = min(len(df1), len(df2))
        col1_values = col1_values.head(min_len)
        col2_values = col2_values.head(min_len)

        # 3. 执行逐行比较: col1 > col2
        comparison_result = (col1_values > col2_values)
        
        # 4. 统计 True 的数量
        count_greater = comparison_result.sum()
        
        # 5. 找出所有满足条件的行 (Series 索引就是行号 - 1)
        # 这里的索引是基于 Pandas DataFrame 的默认 0-based 索引
        greater_indices = comparison_result[comparison_result == True].index.tolist()
        
        # 6. 打印详细信息
        print("\n" + "="*50)
        print("          ✨ CSV 最后一列比较结果详情 ✨")
        print("="*50)
        print(f"文件 1: **{file1_path}**")
        print(f"文件 2: **{file2_path}**")
        print(f"比较的列名: **{last_col_name1}**")
        print(f"实际比较的行数: {min_len}")
        
        print("-" * 50)
        print(f"第一个文件大于第二个文件的总行数: **{int(count_greater)}**")
        print("-" * 50)

        if count_greater > 0:
            print("\n**▶️ 满足条件 (File 1 > File 2) 的具体行信息：**")
            
            # 格式化输出表格
            header = "{:<10} | {:<20} | {:<20}".format("行号", 
                                                      f"文件 1 值 ({last_col_name1})", 
                                                      f"文件 2 值 ({last_col_name2})")
            print(header)
            print("-" * (len(header) + 4))

            # 遍历大于的行索引并打印值
            for index in greater_indices:
                # 打印行号时需要 +1 (如果第一行是标题，这里是数据的第一行)
                row_number = index + 1 
                val1 = col1_values.iloc[index]
                val2 = col2_values.iloc[index]
                
                # 使用 f-string 格式化浮点数，保留两位小数，并居中对齐
                print("{:<10} | {:<20.4f} | {:<20.4f}".format(row_number, val1, val2))
        else:
            print("没有找到第一个文件严格大于第二个文件的行。")
            
        return int(count_greater)

    except FileNotFoundError:
        print("❌ 错误: 文件未找到。请检查路径是否正确。")
        return -1
    except KeyError:
        print("❌ 错误: 尝试访问的列不存在。请检查文件是否为空或格式是否正确。")
        return -1
    except Exception as e:
        print(f"❌ 发生未知错误: {e}")
        return -1

# --- 示例用法 (请替换为您的实际文件路径) ---

file_path_1 = './poc/sqnr_text_audio_vision_8_sample//sqnr_Qwen2.5-Omni-3B_split_scales_omnibench_20percent_w4a8.csv'  # <-- **请替换**
file_path_2 = './poc/sqnr_cali_8_sample//sqnr_Qwen2.5-Omni-3B_split_scales_omnibench_20percent_w4a8.csv'  # <-- **请替换**

# 注意：运行此代码需要确保您有这两个文件，并且它们位于指定的路径
# 运行统计函数
count = count_file1_greater_than_file2_detailed(file_path_1, file_path_2)

