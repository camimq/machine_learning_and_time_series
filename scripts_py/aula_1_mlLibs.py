# %%
import pandas as pd

# %%
df_diabetes = pd.read_csv('https://raw.githubusercontent.com/camimq/machine_learning_and_time_series/main/bases/diabetes.csv', sep=',')
df_diabetes.head()

# %% [markdown]
# ## Separação de Dados para Iniciar o processo de ML
# 
# - Separação dos dados, normalmente, é feita considerando `variável x` e `variável y`,
# - Na separação dos dados, é importante considerar sempre base de treino e teste para as duas variáveis,

# %%
# importa a parte que será utilizada da biblioteca
from sklearn.model_selection import train_test_split

# %%
# traz todas as variáveis característica, excluindo a coluna / variavel Class Variable
x = df_diabetes.drop(['Class variable'], axis=1)
x

# %%
# cria variável y com a coluna Class variable
# 0 = Não tem diabetes
# 1 = Tem diabetes
y = df_diabetes['Class variable']
y

# %%
# criando base de treino e teste
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3) #fazendo o test size, automaticamente, a lib entende que o restante é para treino

# %%
len(x_train)

# %%
df_diabetes.shape

# %%
len(x_test)

# %%
#chamando o algoritmo de treinamento
from sklearn.neighbors import KNeighborsClassifier

# %%
# passa os hiperparâmetros da amostra, para treinar o modelo
knn = KNeighborsClassifier(n_neighbors=3)

# %%
# treina o modelo
knn.fit(x_train, y_train)

# %%
# testa o modelo por acurácia
accuracy = knn.score(x_test, y_test)

# %%
accuracy


