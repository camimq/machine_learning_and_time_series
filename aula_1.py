
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

# %%
# transformou a matriz em uma tabela de 4 linhas e 3 colunas


