# %%
# importando bibliotecas necessárias
import pandas as pd

# %%
#Subindo a base dos dados:
df = pd.read_csv('https://raw.githubusercontent.com/camimq/machine_learning_and_time_series/main/02_machine_learning_avancado/bases/Frutas.csv', sep=';')

# %%
df.head() #visualizando os dados

# %%
from sklearn.preprocessing import LabelEncoder #importando LabelEncoder

# %%
# Instanciando o labelencoder
labelencoder = LabelEncoder()

# %%
# Atribuindo valores numéricos e armazenando em outra coluna
df['CategoriasFrutas'] = labelencoder.fit_transform(df['Fruta'])

# %%
df.head() #visualizando o resultado com labelencoder

# %% [markdown]
# ## One Hot Enconding / Dummies

# %%
# criando a classificação binária por tipo de categoria
dum_df = pd.get_dummies(df, columns=["Fruta"])

# %%
dum_df

# %%



