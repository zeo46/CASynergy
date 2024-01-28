import os
import pandas as pd

def read_gene_symbols(file_path):
    gene_symbols = {}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(',')
            if len(parts) >= 2:
                key = parts[0].strip('"')  # 去掉键的引号
                value = parts[1].strip('"')  # 去掉值的引号
                gene_symbols[key] = value
                gene_symbols[parts[0]] = parts[1]
    return gene_symbols

def replace_ids_with_gene_symbols(input_file, output_file, gene_symbols):
    with open(input_file, 'r') as input_f, open(output_file, 'w') as output_f:
        for line in input_f:
            parts = line.strip().split('\t')
            if len(parts) >= 2 and parts[0] in gene_symbols:
                output_f.write(f"{gene_symbols[parts[0]]}\t{parts[1]}\n")
            else:
                output_f.write(line)
# 读取两个文件：
# 第一个是 cell_line_name.txt 文件，有两列，第一列是 ID 第二列是 VALUE, 如果有其他列则忽略
# 第一个是 cell_line_name_annot.txt 文件，有两列，第一列是 ID 第二列是 Gene symbol
# 要求按照cell_line_name_annot.txt文件的ID和Gene symbol的对应关系，将cell_line_name.txt文件的ID换为Gene symbol
def get_gene_exp(file_path, file_path_processed, cell_line_name):
    # 构建文件路径
    cell_line_annot_file = os.path.join(file_path, cell_line_name + '_annot.txt')
    cell_line_data_file = os.path.join(file_path, cell_line_name +'.txt')
    cell_line_processed_file = os.path.join(file_path_processed, cell_line_name + '_processed.txt')

    # 读取 Gene symbols
    gene_symbols = read_gene_symbols(cell_line_annot_file)

    # 替换 IDs 为 Gene symbols
    replace_ids_with_gene_symbols(cell_line_data_file, cell_line_processed_file, gene_symbols)

# 读取两个文件：
# 第一个是 cell_line_name_processed.txt 文件，有两列，第一列是 Gene symbol 第二列是 VALUE
# 第二个是 genes_name.xlsx 文件，只有一列是 Gene symbol
# 要求按照genes_name.xlsx 文件的Gene symbol，从cell_line_name.txt文件挑选出对应的行，如果没有那么VALUE就填充 0
def pick_exp(file_path, cell_line_name):
    # 构建文件路径
    cell_line_processed_file = os.path.join(file_path, cell_line_name + '_processed.txt')
    genes_name_file = os.path.join(file_path, 'genes_name.xlsx')

    # 读取 cell_line_name_processed.txt 文件
    cell_line_df = pd.read_csv(cell_line_processed_file, sep='\t', names=['Gene symbol', 'VALUE'])

    # 读取 genes_name.xlsx 文件
    genes_name_df = pd.read_excel(genes_name_file, names=['Gene symbol'])

    # 使用 merge 连接两个数据框，以 genes_name_df 中的 Gene symbol 为基准
    result_df = genes_name_df.merge(cell_line_df, on='Gene symbol', how='left')

    # 将没有匹配的行中的 VALUE 填充为 0
    result_df['VALUE'].fillna(0, inplace=True)
    # 去重结果中的 Gene symbol
    result_df.drop_duplicates(subset='Gene symbol', keep='first', inplace=True)
    # 输出结果到文件，如果需要
    result_df.to_csv(os.path.join(file_path, f'{cell_line_name}_pick_exp.csv'), sep=',', index=False, header=False)

# 读取两个文件：
# 第一个是 DrugCombDB_gene_exp.csv 文件，每行代表一个细胞系，每列代表一个基因，列名称是gene symbol
# 第二个是 genes_name.xlsx 文件，只有一列是 Gene symbol
# 要求按照genes_name.xlsx 文件的Gene symbol，从cell_line_name.txt文件挑选出对应的列，如果没有那么VALUE就填充 0
def pick_exp_all(file_path):
    # 构建文件路径
    gene_exp_file = os.path.join(file_path, 'DrugCombDB_gene_exp.csv')
    genes_name_file = os.path.join(file_path, 'genes_name.xlsx')

    # 读取 gene_exp_file 文件并转置
    gene_exp_df = pd.read_csv(gene_exp_file, index_col=0).T
    # 读取 genes_name.xlsx 文件
    genes_name_df = pd.read_excel(genes_name_file, names=['Gene symbol'])

    # 使用 merge 连接两个数据框，以 genes_name_df 中的 Gene symbol 为基准
    result_df = gene_exp_df.join(genes_name_df.set_index('Gene symbol'), how='right')

    # 填充没有匹配的列为 0
    result_df.fillna(0, inplace=True, axis=1)
    # 转置结果数据框
    result_df = result_df.T
    # 输出结果到文件，如果需要
    result_df.to_csv(os.path.join(file_path, 'result_gene_exp.csv'), index=True)

if __name__ == '__main__':
    file_path = '/home/zl/CAESynergy/data/raw_data_process/exp_5/processed_data/'
    file_path_processed = '/home/zl/CAESynergy/data/raw_data_process/exp_5/processed_data/'
    cell_line_name = "TC32"  # COLO205 DU145 MOLT4 SNB19 TC32
    # get_gene_exp(file_path, file_path_processed, cell_line_name)
    pick_exp(file_path, cell_line_name)
    # pick_exp_all(file_path)
    print("挑选并填充完成")


