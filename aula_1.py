
# %%
import pandas as pd

# %%
df_excel = pd.read_excel('bases\Chess.xlsx', sheet_name='Chess')

# %%
df_csv = pd.read_csv('bases\Tomato.csv', sep=',')

# %%
df_excel.head()

# %%
df_csv.head()

# %%
# verifica tamanho / formato do banco de dados
df_csv.shape

# %%
df_excel.shape

# %%
df_excel.info()

# %%
df_csv.info()

# %%
# verificar estatísticas básica da base de dados
df_csv.describe()

# %%
df_excel.describe()

# %%
df_csv.describe().T

# %%
# verifica uma única coluna
df_excel.head()

# %%
#verifica as categorias de uma variável categórica
set(df_excel['victory_status'])

# %%
# cria coluna e gera valor pra essas colunas
df_csv.head()

# %%
df_csv.describe()

# %%
# categorizando os tomates de acordo com a média
def categorizar_tomate_media(media):
    if media >= 40 and media <= 70:
        return 'tomate medio'
    elif media < 40:
        return 'tomate pequeno'
    else:
        return 'tomate grande'

# %%
# cria coluna e insere dados
df_csv['categoria_tomate'] = df_csv['Average'].apply(categorizar_tomate_media)

# %%
df_csv.head()

# %%
# faz a estatística descritiva de uma coluna de variável categórica, por categoria
df_csv.groupby(['categoria_tomate']).describe()

# %%
# filtrando dados no Pandas - todos os tomates com média < 40
filtro = df_csv['Average'] < 40
df_csv.loc[filtro]


# %%
import numpy as np

# %%
# cria um array
arr_list = np.array([1, 2, 3, 4, 5, 6, 7, 8])

# %%
print(arr_list)

# %%
# cria uma matriz / table
arr_zeros = np.zeros((4, 6))
print(arr_zeros)

# %%
arr_ones = np.ones((3, 4))
print(arr_ones)

# %%
# cria uma matriz aleatória de 0 e 1
arr_random = np.random.rand(3, 4)
print(arr_random)

# %%
# manipulando arrays
# veriricar a dimensão de uma matriz
print(arr_random.shape)

# %%
# transforma o formato do array
arr_random_reshape = arr_random.reshape((4, 3))
print(arr_random_reshape)
# transformou a matriz em uma tabela de 4 linhas e 3 colunas

# %%
# concatenando arrays
arr1 = np.array([[1, 2], [3, 4]])
arr2 = np.array([[5, 6], [7, 8]])
arr3 = np.array([[9, 10], [11, 12]])

# %%
arr1

# %%
arr2

# %%
arr3

# %%
# concatenando todos os arrays criados no arr4
arr4 = np.concatenate((arr1, arr2, arr3), axis = 1)

# %%
arr4

# %%
# dividir o array em 2
arr4_split = np.split(arr4, 2)
print(arr4_split)

# %%
# transposição de matriz
arr4_transpose = np.transpose(arr4)
print(arr4_transpose)


# %%
# reverte transposição de matriz
arr4_transpose_revertido = arr4_transpose.T
arr4_transpose_revertido

# %%
arr_a = np.array([1, 7, 27])
arr_b = np.array([1, 5, 1])

# %%
# cria um array que soma o a e b
arr_a_b = np.add(arr_a, arr_b)
arr_a_b

# %%
# cria um array que subtrai arr_a e arr_b
arr_sub_a_b = np.subtract(arr_a, arr_b)
arr_sub_a_b