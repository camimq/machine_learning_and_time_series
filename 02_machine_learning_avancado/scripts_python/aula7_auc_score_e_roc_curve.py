# %% [markdown]
# # Recrutamento preditivo
# 
# A empresa de tecnologia HighTech busca contratar os melhores profissionais do mercado para fazer parte do time e gerar valor para a empresa. A HighTech vem investindo muito nos últimos anos no uso de ciência de dados no setor do RH para trazer ganhos no processo de seleção e recrutamento. O time de ciência de dados junto com o time de RH vem realizando juntos um projeto de recrutamento preditivo.
# 
# O recrutamento preditivo é uma técnica de People Analytics para encontrar os melhores candidatos para contratação da empresa, na prática, o recrutamento preditivo aumenta as chances do recrutador potencializar o processo de seleção. Por meio da coleta e análise de dados, é possível avaliar o perfil e o fit cultural dos profissionais para entender se existe uma boa aderência à vaga. 
# 
# **Problema de negócio:**
# 
# O objetivo da HighTech é identificar quais são os melhores indicadores para realizar o recrutamento de profissionais. 
# 
# **Base de dados**
# 
# Este conjunto de dados consiste em algumas características como: percentual de ensino médio e superior e especialização, experiência de trabalho e ofertas salariais para os profissionais colocados.
# 
# **Desafio**
# 
# Você como cientista de dados do time de dados da HighTech tem o desafio de criar um modelo preditivo de recrutamento para prever como e quais são as melhores variáveis que podem colocar um profissional bem qualificado na HighTech.
# 

# %% [markdown]
# **Importando a base de dados**

# %%
import pandas as pd

# %%
dados = pd.read_csv('https://raw.githubusercontent.com/camimq/machine_learning_and_time_series/main/02_machine_learning_avancado/bases/Recrutamento.csv', sep = ';')

# %%
dados.shape

# %%
dados.head()

# %% [markdown]
# **Inferência sobre a base de dados:**
# 
# Podemos observar que temos algumas variáveis como: gênero, desempenho educacional, score de desempenho educacional, status de contratação, salário.
# 
# **Variável Target:** No nosso case a target é a coluna status.

# %%
set(dados.status)

# %% [markdown]
# **Analisando os dados**

# %%
dados.describe()

# %% [markdown]
# Inferência sobre os dados:
# 
# Métricas de pontuação sobre ensino:
# ssc_p
# hsc_p
# degree_p
# estet_p
# mba_p
# 
# sl_no é um código, então não faz sentido na análise.
# 
# salary vem após a contratação.
# 
# 

# %% [markdown]
# Observando valores nulos:

# %%
import missingno as msno 
msno.matrix(dados)

# %%
dados.isnull().sum() 

# %%
import seaborn as sb

# %%
sb.boxplot(x='status', y='salary', data=dados, palette='hls')

# %% [markdown]
# Inferência sobre os dados:
# 
# Observe que para a variável salário, os valores nulos estão atribuídos a variável do tipo status quando o status é "não", ou seja, para os não contratados temos algumas pessoas da base sem salário atribuído.
# 
# Como podemos realizar a tratativa dos valores nulos?

# %%
dados['salary'].fillna(value=0, inplace=True)

# %%
dados.isnull().sum()

# %% [markdown]
# Analisando as variáveis numéricas: 
# 
# Vamos analisar e compreender a distribuição dos dados para cada métrica de pontuação de ensino. Será que temos outliers na base?
# 

# %%
sb.boxplot(x=dados["hsc_p"])

# %%
sb.histplot(data=dados, x="hsc_p")

# %%
sb.boxplot(x=dados["degree_p"])

# %%
sb.histplot(data=dados, x="degree_p")

# %%
sb.boxplot(x=dados["etest_p"])

# %%
sb.histplot(data=dados, x="etest_p")

# %%
sb.boxplot(x=dados["mba_p"])

# %%
sb.histplot(data=dados, x="mba_p")

# %%
sb.boxplot(x=dados["salary"])

# %%
sb.histplot(data=dados, x="salary")

# %% [markdown]
# Será que os scores acadêmicos influenciam na contratação? E a experiência de trabalho?

# %%
sb.set_theme(style="whitegrid", palette="muted")

ax = sb.swarmplot(data=dados, x="mba_p", y="status", hue="workex")
ax.set(ylabel="")

# %% [markdown]
# Inferência dos dados:
# 
# Podemos observar que a pontuação de MBA pode influência sim na decisão de contratação, temos um grande concetração de dados sobre profissionais que possuem score de mba_p e com esperiência de trabalho. 

# %% [markdown]
# Existe algum viés de gênero ao oferecer remuneração?

# %%
! pip install plotly_express

# %%
import plotly_express as px

# %%
px.violin(dados,y="salary",x="specialisation",color="gender",box=True,points="all")

# %% [markdown]
# Inferência sobre os dados:
# 
# Os maiores salários foram dados aos homens.
# O salário médio oferecido também foi maior para homens.
# 

# %% [markdown]
# Vamos analisar a correlação entre as pontuações de desempenho acadêmico com a contratação:

# %%
sb.pairplot(dados,vars=['ssc_p','hsc_p','degree_p','mba_p','etest_p'],hue="status")

# %% [markdown]
# Inferência sobre os dados:
# 
# Candidatos com pontuação alta no ensino médio e na graduação foram contratados.
# Quem obteve notas altas em suas escolas foi contratado.

# %%
import matplotlib.pyplot as plt
%matplotlib inline

# %% [markdown]
# Entendendo as correlações:

# %%
import matplotlib.pyplot as plt 

# %%
correlation_matrix = dados.corr(numeric_only=True).round(2)

fig, ax = plt.subplots(figsize=(8,8))    
sb.heatmap(data=correlation_matrix, annot=True, linewidths=.5, ax=ax)

# %% [markdown]
# Aqui somente conseguimos analisar a correlação entre as variáveis numéricas. Será que não seria importante também entender a correlação com as variéveis numéricas?
# 
# Vamos aplicar técnicas de transformação nos dados:
# 
# 
# Vamos utilizar label enconder para tratar variáveis categoricas que possuem apenas dois tipos de categorias, como genero, especialização e status.
# 
# Para as demais categorias, vamos aplicar a tecnica de one hot enconing.

# %%
from sklearn.preprocessing import LabelEncoder

# %%
dados.head()

# %%

colunas=['gender','workex','specialisation','status']

label_encoder = LabelEncoder()
for col in colunas:
    dados[col] = label_encoder.fit_transform(dados[col])
dados.head()

# %% [markdown]
# Aplicando a técnica de one hot enconding:
# 

# %%
dummy_hsc_s=pd.get_dummies(dados['hsc_s'], prefix='dummy')
dummy_degree_t=pd.get_dummies(dados['degree_t'], prefix='dummy')

dados_coeded = pd.concat([dados,dummy_hsc_s,dummy_degree_t],axis=1)
dados_coeded.drop(['hsc_s','degree_t','salary'],axis=1, inplace=True)
dados_coeded.head()

# %%
correlation_matrix = dados_coeded.corr(numeric_only=True).round(2)

fig, ax = plt.subplots(figsize=(12,12))    
sb.heatmap(data=correlation_matrix, annot=True, linewidths=.5, ax=ax)

# %% [markdown]
# Agora, conseguimos analisar as correlações!
# 
# Analisando a correlação e a análise de dados, podemos considerar algumas variáveis como possíveis fortes features para nosso modelo de classificação!
# 
# Mas lembre-se, correlação não é causalidade!
# 
# Analisando algumas variáveis e sua correlação com a variável status, podemos identificar que as variáveis workex, degree_p, hsc_p e ssc_p possuem uma correlação interessante na contratação.
# 
# A maior correlação de status de contratação está com o score de ssc_p, ou seja, pessoas com alto score de ssc_p são mais contratadas.
# 
# Vamos analisar?
# 

# %%
sb.relplot(x="status", y="ssc_p", hue="status", size="ssc_p",
            sizes=(40, 400), alpha=.5, palette="muted",
            height=6, data=dados_coeded)

# %%
x = dados_coeded[['ssc_p', 'hsc_p', 'degree_p', 'workex', 'mba_p']] #variaveis independentes
y = dados_coeded['status'] #target

# %%
from sklearn.model_selection import train_test_split #separação em treino e teste
from sklearn.neighbors import KNeighborsClassifier   #knn

# %%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=7) 

# %%
x_train.shape

# %%
x_test.shape

# %%
from sklearn.preprocessing import StandardScaler, MinMaxScaler  

# %%
scaler = StandardScaler() 
#scaler = MinMaxScaler() 

scaler.fit(x_train)

x_train_escalonado = scaler.transform(x_train)
x_test_escalonado = scaler.transform(x_test) 

# %%
import numpy as np

# %%
error = []

# Calculating error for K values between 1 and 10
for i in range(1, 10): #range de tentativas para k
    knn = KNeighborsClassifier(n_neighbors=i)# aqui definimos  o k
    knn.fit(x_train_escalonado, y_train) #treinando o algoritmo para encontrar o erro
    pred_i = knn.predict(x_test_escalonado) #armazenando as previsões
    error.append(np.mean(pred_i != y_test)) #armazenando o valor do erro médio na lista de erros

# %%
plt.figure(figsize=(12, 6))
plt.plot(range(1, 10), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')

# %%
#!pip install --upgrade scikit-learn

# %%
modelo_classificador = KNeighborsClassifier(n_neighbors=5)

modelo_classificador.fit(x_train_escalonado, y_train) 

# %%
y_predito = modelo_classificador.predict(x_test_escalonado) 

# %%
y_predito

# %%
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay # matriz de confusão
from sklearn.metrics import classification_report

# %%
from sklearn.metrics import confusion_matrix

# %%
matriz_confusao = confusion_matrix(y_true = y_test,    #dados reais
                                   y_pred = y_predito #dados predito
                                   )

# plotando uma figura com a matriz de confusao
figure = plt.figure(figsize=(15, 5))

disp = ConfusionMatrixDisplay(confusion_matrix = matriz_confusao)

disp.plot(values_format='d') 

# %%
# Metricas de precisão, revocação, f1-score e acurácia.
print(classification_report(y_test, y_predito)) #relatório de validação das métrica de desempenho.

# %% [markdown]
# Testando com o modelo SVM:

# %%
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

# %%
svm = Pipeline([
    ("linear_svc", LinearSVC(C=1))
])

svm.fit(x_train_escalonado, y_train) 

# %%
y_predito_svm = svm.predict(x_test_escalonado) 

# %%
matriz_confusao_svm = confusion_matrix(y_true = y_test,       #dados reais
                                   y_pred = y_predito_svm #dados predito
                                   )

# plotando uma figura com a matriz de confusao
figure = plt.figure(figsize=(15, 5))

disp = ConfusionMatrixDisplay(confusion_matrix = matriz_confusao_svm)

disp.plot(values_format='d') 

# %%
# Metricas de precisão, revocação, f1-score e acurácia.
print(classification_report(y_test, y_predito_svm)) #relatório de validação das métrica de desempenho.

# %%
from sklearn import metrics
from sklearn.metrics import roc_curve, auc

# %%
y_prob = modelo_classificador.predict_proba(x_test_escalonado)[:,1] 

# %%
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(false_positive_rate, true_positive_rate)
roc_auc

# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(10,10))
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],linestyle='--')
plt.axis('tight')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')


