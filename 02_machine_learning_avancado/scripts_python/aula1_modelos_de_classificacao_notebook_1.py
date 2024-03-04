# %% [markdown]
# # Modelos de Classificação

# %% [markdown]
# ## Introdução a modelos de classificação de dados em Machine Learning

# %% [markdown]
# Você sabe como funciona um modelo supervisionado de classificação em machine learning? Na aula de hoje vamos aprender a como criar um classificador automático e inteligente utilizando ferramentas de machine learning. Vamos lá? 😀
# 
# Case: Classificação de insetos gafanhotos e esperanças
# 
# Um certo cientista coletou dados de amostra sobre uma população de insetos da espécie gafanhoto e esperança para realizar um estudo e identificar uma forma de encontrar diferenças entre os tipos de insetos analisando algumas das características presentes no corpo dos insetos.
# 
# Com base em suas pesquisas e análises, o cientista identificou que as características do tamanho do abdomên e comprimento das antenas desses insetos podem ser um fator muito relevante para a identificação da espécie.
# 
# O cientista precisa encontrar uma maneira de identificar de forma automática e precisa os padrões dessas características que podem classificar quando um inseto é do tipo gafanhoto ou do tipo esperança.
# 
# Vamos aplicar machine learning para resolver esse problema?

# %%
import pandas as pd

# %%
dados = pd.read_csv('https://raw.githubusercontent.com/camimq/machine_learning_and_time_series/main/02_machine_learning_avancado/bases/gaf_esp.csv', sep = ';')

# %%
dados.head()

# %%
dados.describe()

# %%
dados.groupby('Espécie').describe()

# %%
dados.plot.scatter(x='Comprimento do Abdômen', y='Comprimento das Antenas')

# %%
from sklearn.model_selection import train_test_split

# %%
x = dados[['Comprimento do Abdômen', 'Comprimento das Antenas']]
y = dados['Espécie']

# %%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)

# %%
list(y_train).count('Gafanhoto')

# %%
print("Total base de treino: ", len(x_train))
print("Total base de teste: ", len(y_test))

# %%
from sklearn.neighbors import KNeighborsClassifier

# %%
# Hiperparametro do nosos modelo é o número de vizinhos considerado (n_neighbors)
modelo_classificador = KNeighborsClassifier(n_neighbors=3)

# Está fazendo o treinamento do meu modelo de ML
modelo_classificador.fit(x_train, y_train)

# %%
# Comprimento AB: 8
# Comprimento AT: 6
modelo_classificador.predict([[8,6]])

# %%
from sklearn.metrics import accuracy_score

# %%
y_predito = modelo_classificador.predict(x_test)

# %%
accuracy_score(y_true = y_test, y_pred=y_predito)


