import pandas as pd

# 按照基因和蛋白序号的对应关系转换 gene symbol 为 protein index
# 输入为两个csv文件，第一个是gene_expression.csv，每一行代表一个细胞系，每列是一个基因名称，值是基因表达值。
# 第二个是cell_protein_map.csv，第一列是基因名称GeneSym，第二列是蛋白序号protein。
# 现在要求按照cell_protein_map.csv的对应关系替换gene_expression.csv的列名称
def cell_line_expresion(file_path):
    # 读取信息
    express_data = pd.read_csv(file_path + 'gene_expression.csv')
    gene_protein_map = pd.read_excel(file_path + 'cell_genes.xlsx')
    gene_protein_map['protein'] = gene_protein_map['protein'].astype(int)
    # 创建一个字典，将GeneSym映射到protein
    gene_symbol_to_protein = dict(zip(gene_protein_map['GeneSym'], gene_protein_map['protein']))
    # 获取 gene_expression.csv 的基因名称列表
    gene_names = express_data.columns.tolist()

    # 用字典映射来替换列名称
    express_data.rename(columns=gene_symbol_to_protein, inplace=True)

    # 将结果保存为一个新的CSV文件
    express_data.to_csv(file_path + 'gene_expression_renamed.csv', index=False)

# 按照基因和蛋白序号的对应关系转换 gene symbol 为 protein index
# 输入为两个csv文件，第一个是gene_expression.csv，每一行代表一个细胞系，每列是一个基因名称，值是基因表达值。
# 第二个是cell_protein_map.csv，第一列是基因名称GeneSym，第二列是蛋白序号protein。
# 现在要求按照cell_protein_map.csv的对应关系找出gene_expression.csv的列名称对应的蛋白序号
def get_protein_index(file_path):
    # 读取信息
    express_data = pd.read_csv(file_path + 'gene_expression.csv')
    gene_protein_map = pd.read_excel(file_path + 'cell_genes.xlsx')
    gene_protein_map['protein'] = gene_protein_map['protein'].astype(int)
    # 创建一个字典，将GeneSym映射到protein
    gene_symbol_to_protein = dict(zip(gene_protein_map['GeneSym'], gene_protein_map['protein']))

    # 获取gene_expression.csv的列名称
    gene_columns = express_data.columns

    # 创建一个新的DataFrame来存储基因名称到蛋白序号的映射
    protein_mapping = pd.DataFrame({'GeneSym': gene_columns})

    # 使用字典映射基因名称到蛋白序号
    protein_mapping['protein'] = protein_mapping['GeneSym'].map(gene_protein_map)

    # 保存映射结果到一个新文件
    protein_mapping.to_csv(file_path + 'gene_to_protein_mapping.csv', index=False)

if __name__ == '__main__':
    file_path = '/home/zl/CAESynergy/data/raw_data_process/cline_exp/DrugcombDB/'
    cell_line_expresion(file_path)
    # get_protein_index(file_path)
    print("y1")



