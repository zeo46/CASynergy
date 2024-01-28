import pandas as pd
# 输入为两个csv文件，第一个是cell_mask.csv
# 第二个是cell line_gene_expression.csv
# 现在要求按照文件一：cell_mask.csv的列名称，选取文件二：cell line_gene_expression.csv的列，意思就是选文件二中列名称和文件一的列名称一样的子集。
def cell_line_expression(file_path):
    # 读取信息
    cell_mask = pd.read_csv(file_path + 'cell_mask.csv')
    gene_expression = pd.read_csv(file_path + 'cell line_gene_expression.csv', index_col=0)

    # 选取与 cell_mask 列名相匹配的列
    matching_columns = [col for col in gene_expression.columns if col in cell_mask.columns]
    gene_expression_sub = gene_expression[matching_columns]

    # 将结果保存为一个新的 CSV 文件
    gene_expression_sub.to_csv(file_path + 'gene_expression_sub.csv', index=True, index_label=gene_expression.index.name)

if __name__ == '__main__':
    file_path = '/home/zl/CAESynergy/data/raw_data_process/DrugcombDB/pick_gene_exp/'
    cell_line_expression(file_path)
    print("y1")


