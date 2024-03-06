# %% [markdown]
# ### Classificando problemas ortopédicos

# %% [markdown]
# Esse dataset contém dados sobre problemas ortopédicos na coluna vertebral, diagnosticados no Centre Médico-Chirurgical de Réadaptation des Massues, em Lyon, France. Contém 6 atributos biomecânicos para 310 entradas **anonimizadas**, sendo **100** de pacientes considerados sem problemas **(Normal - NO)**, **60** de pacientes com **Hérnia de Disco (Disk Hernia - DH)** e **150** de pacientes com Espondilolistese **(Spondylolisthesis - SL)**. 
# 
# O dataset está disponível em https://www.openml.org/d/1523
# 

# %% [markdown]
# # Passo 1: Carregar a base de dados

# %%
from sklearn.datasets import fetch_openml # importando o open ML
import pandas as pd                       
import matplotlib.pyplot as plt           
import seaborn as sns                    
import numpy as np                        

from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier   
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, cohen_kappa_score                      
from sklearn.preprocessing import StandardScaler, MinMaxScaler                             

# %%
dados = fetch_openml(data_id=1523)

# %%
tabela_dados = pd.DataFrame(data=dados['data'])

# %%
tabela_dados.head()

# %% [markdown]
# ### Transformando a base de dados:

# %%
classes = {'1':'Disk Hernia',
           '2':'Normal',
           '3':'Spondylolisthesis'}

# %%
classes['1'], classes['2'], classes['3']

# %%
tabela_dados['diagnostic'] = [classes[target] for target in dados.target]

# %%
tabela_dados.head()

# %% [markdown]
# ## Passo 2: Análise exploratória dos dados

# %%
tabela_dados.info()

# %%
len(tabela_dados)

# %%
tabela_dados.describe()

# %% [markdown]
# Como está a média dos dados?

# %%
tabela_dados.groupby('diagnostic').mean()

# %%
tabela_dados.groupby('diagnostic').describe()

# %% [markdown]
# Analisando o outlier:

# %%
tabela_dados.loc[tabela_dados['V6'] > 400]

# %% [markdown]
# # Removendo o outlier:

# %%
tabela_dados.drop(tabela_dados.loc[tabela_dados['V6'] > 400].index, inplace=True)

# %%
tabela_dados.loc[tabela_dados['V6'] > 400]

# %% [markdown]
# ## Passo 3: Separação dos dados

# %%
x = tabela_dados.drop(columns=['diagnostic'])
y = tabela_dados['diagnostic'] # O que eu quero prever. (Target)

# %%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42) 

# %% [markdown]
# ### Passo 4: Normalizando os dados

# %%
# scaler = StandardScaler() # chamando o metodo de padronização dos dados (média e std)
scaler = MinMaxScaler() # chamando o metodo de normalização dos dados (0-1)

scaler.fit(x_train) # qual média e std será utilizado para o escalonamento

x_train_scaled = scaler.transform(x_train)
x_test_scaled= scaler.transform(x_test)

# %% [markdown]
# ### Passo 5: Criação do modelo

# %% [markdown]
# ### Treinando o modelo com dados sem escalonamento:

# %%
modelo_classificador = KNeighborsClassifier(n_neighbors=3)

modelo_classificador.fit(x_train_scaled, y_train) 

# %% [markdown]
# ### Validação do modelo (testando o modelo)

# %%
# Predição sem escalolnamento
y_predito = modelo_classificador.predict(x_test_scaled)

# %%
tabela_dados.groupby('diagnostic').count()

# %%
from sklearn.metrics import ConfusionMatrixDisplay

# %%
matriz_confusao = confusion_matrix(y_true = y_test,
                                   y_pred = y_predito,
                                   labels=['Disk Hernia', 'Normal', 'Spondylolisthesis'])

# plotando uma figura com a matriz de confusao
figure = plt.figure(figsize=(15, 5))
disp = ConfusionMatrixDisplay(confusion_matrix = matriz_confusao,
                              display_labels=['Disk Hernia', 'Normal', 'Spondylolisthesis'])
disp.plot(values_format='d') 

# %%
print(classification_report(y_test, y_predito))

# %% [markdown]
# ### Testando o KNN com validação cruzada:

# %% [markdown]
# ### Incluindo a etapa de validação do modelo de knn:
# 
# Estamos fazendo a VALIDAÇÃO dos melhores hiperparâmetros do modelo de ML:
# 
# Idealmente, devemos ter 5 conjuntos de dados (treino, teste, validacao) ou usar a validação cruzada em cima dos dados de treino.

# %%
from sklearn.model_selection import cross_val_score 
from sklearn.model_selection import KFold 

# %%
kfold  = KFold(n_splits=5, shuffle=True) # shuffle=True, Shuffle (embaralhar) os dados.
result = cross_val_score(modelo_classificador, x, y, cv = kfold)

print("K-Fold (R^2) Scores: {0}".format(result))
print("Mean R^2 for Cross-Validation K-Fold: {0}".format(result.mean()))

# %% [markdown]
# A saída exibirá a precisão média do modelo (ou seja, a acurácia) com uma medida de variação associada (calculada pela multiplicação do desvio padrão por 2). 

# %% [markdown]
# Escolhendo os melhores hiperparâmetros com validação cruzada:

# %%
from sklearn.model_selection import GridSearchCV # método para seleção dos melhores Ks (#Basicamente a busca por força bruta)
from sklearn.metrics import make_scorer, accuracy_score, f1_score # métricas de validação
import numpy as np
import matplotlib.pyplot as plt

# %%
error = [] # armazenar os erros

# Calculating error for K values between 1 and 15
for i in range(1, 15):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train_scaled, y_train)
    pred_i = knn.predict(x_test_scaled)
    error.append(np.mean(pred_i != y_test))

# %%
plt.figure(figsize=(12, 6))
plt.plot(range(1, 15), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')

# %%
# Buscar os melhores parâmetros
# Vamos usar uma técnica chamada Gridsearch que é basicamente a busca por força bruta
# Iremos utilizar a técnica de validação cruzada de 5 folds (divisões) em cima do conjunto de treinamento
# Como métrica de melhor desemepnho usaremos a acurácia, isto é, estamos buscando os hiperparâmetros que maximizam a acurácia

# Parâmetros testados
param_grid = {'n_neighbors':[8,14], # total de vizinhos
              'weights': ['uniform', 'distance'], # Função de peso usada na previsão
              'metric':['cosine', 'euclidean', 'manhattan'] # métrica para cálculo da distância
             }

# Métrica de desempenho = melhor acurácia
gs_metric = make_scorer(accuracy_score, greater_is_better=True)

grid = GridSearchCV(KNeighborsClassifier(), 
                    param_grid=param_grid,
                    scoring=gs_metric,
                    cv=5, n_jobs=4, verbose=3) # cv = gerador de validação cruzada / n_jobs = determina a quantidade de jobs processados

grid.fit(x_train_scaled, y_train) # Selecionando os melhores hiperparametros com base nos dados de treino
knn_params = grid.best_params_
print('KNN', knn_params)

# %%
grid.cv_results_ # analisando todos os possíveis testes

# %% [markdown]
# ### Testando vários tipos de algoritmos de classificação:

# %%
def AplicaValidacaoCruzada(x_axis, y_axis):
  # Linear Models.
  from sklearn.neighbors import KNeighborsClassifier  # k-vizinhos mais próximos (KNN)
  from sklearn.ensemble import RandomForestClassifier # RandomForest
  from sklearn.svm import SVC                         # Maquina de Vetor Suporte SVM

  # Cross-Validation models.
  from sklearn.model_selection import cross_val_score
  from sklearn.model_selection import KFold

  # Configuração de KFold.
  kfold  = KFold(n_splits=10, shuffle=True) 

  # Axis
  x = x_axis
  y = y_axis

  # Criando os modelos

  # KNN
  knn = KNeighborsClassifier(n_neighbors=8, metric= 'euclidean', weights='distance') 
  knn.fit(x_train_scaled, y_train) 

  # SVM
  svm = SVC()
  svm.fit(x_train_scaled, y_train)

  # RandomForest
  rf = RandomForestClassifier(random_state=7) 
  rf.fit(x_train_scaled, y_train) 

  # Applyes KFold to models.
  knn_result = cross_val_score(knn, x, y, cv = kfold)
  svm_result = cross_val_score(svm, x, y, cv = kfold)
  rf_result = cross_val_score(rf, x, y, cv = kfold)

  # Creates a dictionary to store Linear Models.
  dic_models = {
    "KNN": knn_result.mean(),
    "SVM": svm_result.mean(),
    "RF": rf_result.mean()
  }
  # Select the best model.
  melhorModelo = max(dic_models, key=dic_models.get)

  print("KNN (R^2): {0}\nSVM (R^2): {1}\nRandom Forest (R^2): {2}".format(knn_result.mean(), svm_result.mean(), rf_result.mean()))
  print("O melhor modelo é : {0} com o valor: {1}".format(melhorModelo, dic_models[melhorModelo]))



# %%
  AplicaValidacaoCruzada(x, y)


