import pandas as pd

# 输入为两个文件，第一个是cell_line_ppi.xlsx，第一列第二列都是蛋白序号，第三列是细胞系名称。
# 第二个是cell line_gene_expression.csv，第一列是细胞系名称，第一行是蛋白序号protein，值是基因表达数据。
# 创建一个cell_mask.csv文件, 行列名称都和cell_line_ppi.xlsx文件一样，要求按照cell_line_ppi.xlsx的对应关系，如果这个细胞的第一列第二列有这个蛋白序号，那么就值为1，没有则为0
def get_cline_mask(file_path):
    # Load data from files
    ppi_data = pd.read_excel(file_path + 'cell_line_ppi.xlsx')
    expression_data = pd.read_csv(file_path + 'cell line_gene_expression.csv')

    # Extract protein numbers and cell line names
    proteins = list(expression_data.columns)[1:]  # Extract protein columns
    cell_lines = ppi_data.iloc[:, 2].unique()

    # Create an empty dataframe to store the mask
    mask = pd.DataFrame(0.0001, index=cell_lines, columns=proteins)

    # Iterate through ppi_data and mark occurrences in the mask
    for _, row in ppi_data.iterrows():
        protein1, protein2, cell_line = row
        if str(protein1) in mask.columns:
            mask.loc[cell_line, str(protein1)] = 1
        if str(protein2) in mask.columns:
            mask.loc[cell_line, str(protein2)] = 1

    # Filter columns with all values as 0.0001
    mask = mask.loc[:, (mask != 0.0001).any()]

    # Save the mask to a CSV file
    mask.to_csv(file_path + 'cell_mask.csv')

    # 获取列名相同的列
    common_columns = expression_data.columns.intersection(mask.columns)
    expression_data = expression_data[common_columns]
    # Save the mask to a CSV file
    expression_data.to_csv(file_path + 'cell_gene_exp.csv')

if __name__ == '__main__':
    file_path = '/home/zl/CAESynergy/data/raw_data_process/Oncology/cline_mask/'
    get_cline_mask(file_path)
    print("y1")



