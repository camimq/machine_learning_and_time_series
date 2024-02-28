# %% [markdown]
# # Aula 2 - Machine Learning Com Python

# %% [markdown]
# ## Análise Exploratória de Dados
# ### Case Spotify - Base de dados do Rolling Stones

# %% [markdown]
# As colunas (características) neste conjunto de dados são:
# 
# - **`nome`:** o nome da música
# - **`album`:** o nome do álbum
# - **`release_date`:** o dia, mês e ano em que o álbum foi lançado
# - **`número da faixa`:** a ordem em que a música aparece no álgum
# - **`id`:** o id do Spotify para a música
# - **`uri`:** o uri do Spotify para a música
# - **`acústica`:** uma medida de confiança de 0,0 a 1,0 se a faixa é acústica. 1.0 representa alta confiança de que a faixa é acústica.
# - **`danceability`:** descreve o quanto uma faixa é adequada para dancçar com base em uma combinação de elementos musicais, incluindo andamento, estabilidade do ritmo, força da batida e regularidade geral. Um valor de 0,0 é o menos dançavel e 1,0 é o mais dançavel.
# - **`energia`:** a energia é uma medida de 0,0 a 1,0 e representa uma medida perceptiva de intensidade e atividade. Normalmente, as faixas energéticas parecem rápidas, altas e barulhentas. Por exemplo, _death metal_ tem alta energia, enquanto um prelúdio de Bach pontua baixo na escala. Os recursos perceptivos que contribuem para esse atributo incluem faixa dinâmica, sonoridade percebida, timbre, taxa de início e entropia geral.
# - **`instrumentless`:** prevê se uma faixa não contém vocais. Os sons "Ooh" e "aah" são tratados como instrumentais neste contexto. Faixas de rap ou palavras faladas são claramente "vocais". Quanto mais próximo o valor da instrumentalidade estiver de 1.0. maior a probabilidade de a faixa **não** conter nenhum conteúdo vocal. Valores acima de 0,5 destinam-se a representar faixas instrumentais, mas a confiança é maior conforme o valor se aproxima de 1,0.
# - **`vivacidade`:** detecta presença de uma audiência na gravação. Valores mais altos de vivacidade representam uma probabilidade maior de que a faixa tenha sido tocada ao vivo. Um valor acima de 0,8 fornece uma forte probabilidade de que a faixa esteja ativa.
# - **`loudness`:** o volume geral de uma faixa em decibéis (dB). os valores de voluma são calculados em média em toda a faixa e são úteis para comparar o colume relativo das faixas. Loudness é a qualidade de um som que é o principal correlato psicológico da força física (amplitude). Os valores, geralmente, varia entre -60 e 0 dB.
# - **`locução`:** detecta a presença de palavras faladas em uma faixa. Quanto mais exclusivamente semelhante à fala for a gravação (por exemplo, tal show, livro de áudio, poesia etc), mais próximo de 1,0 será o valor do atributo. Valores acima de 0,66 descrevem faixas que provavelmente são feitas interiamente de palavras faladas. Valores entre 0,33 e 0,66 descrevem faixas que podem conter música e fala, seja em seções ou em camadas, incluindo casos como rap. Valores abaixo de 0,33 provavelmente representam música e outras faixas não semelhantes à fala.
# - **`tempos`:** o tempo geral estimado de uma faixa em batidas por minuto (BPM). Na terminologia musical, é tempo e a velocidade ou ritmo de uma determinada peça e deriva diretamente da duração média da batida.
# - **`valência`:** uma medida de 0,0 a 1,0 que descreve a positividade musical transmitida por uma faixa. Faixas com alta valência soam mais positivas (por exemplo, alegres, eufóricas), enquanto faixas com baixa valência soam mais negativas (por exemplo, tristes, deprimidas, zangadas)
# - **`popularidade`:** a popularidade da música de 0 a 100
# - **`duration_ms`:** a duração da faixa em milissegundos

# %%
import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sns

# %%
for file in glob.glob('./*.xlsx'):
    if '~$' in file:
        continue
    else:
        df = pd.read_excel('https://view.officeapps.live.com/op/view.aspx?src=https%3A%2F%2Fraw.githubusercontent.com%2FFIAP%2FPos_Tech_DTAT%2F61a634996879d5c1384af6f567e29659ad68b727%2FAula%252003%2FBase%2520de%2520Dados%2Fdataset_rolling_stones.xlsx&wdOrigin=BROWSELINK', engine='openpyxl')
df.head()

# %%
df.info()

# %%
df.shape

# %%
# printa data da primeira música lançada
print('Data Inicial: ', df['release_date'].min())

# %%
# printa última data de lançamento de música
print('Data final: ', df['release_date'].max())

# %%
# verifica se tem dados nulos dentro da base
df.isnull().sum()

# %%
# verifica dados duplicados
df.duplicated().sum()

# %%
# verifica quais são os dados duplicados
df[df.duplicated()]

# %% [markdown]
# O que está duplicado, é o nome do álbum, por isso, não pode ser deletado.

# %%
# puxa a estatística descritiva do dataframe
df.describe()

# %%
# transforma milissegundos em minutos
df['duracao_minutos'] = df['duration_ms'] / 60000 # cria uma coluna nova que transforma a coluna de milissegundos para minutos
df.head()


# %%
df.describe()

# %% [markdown]
# A minutagem média passa de milissegundos para 4,29 minutos na última coluna (em média).

# %%
# agrupa as músicas por album e mostra a média de minutos por album
df.groupby('album')['duracao_minutos'].mean()

# %%
# criando dataframe que agrupa os alguns e ordena as musicas por tamanho de musica (do maior para o menor)
df_maior_duracao_musica = df.groupby('album')['duracao_minutos'].mean().sort_values(ascending=False)
df_maior_duracao_musica

# %%
# cria gráfico de barras com top 5 músicas mais longas
df_maior_duracao_musica.head(5).plot(kind='bar')
plt.xlabel=('Álbuns')
plt.ylabel=('Média de Duração das Músicas')
plt.title('Top 5 Álbuns com Maior Duração Média de Música')
plt.show()

# %%
# cria variável com top 10 albuns com maior quantidade de músicas
top_albuns = df['album'].value_counts().head(10)
top_albuns

# %%
# cria o gráfico de barras em formato diferente do padrão matplotlib
plt.barh(top_albuns.index, top_albuns.values)
plt.title('Top 10 Albuns com Mais Músicas')
plt.show()

# %% [markdown]
# #### Analisando a popularidade da banda na última década

# %%
# cria variável que traz as músicas lançadas na última década
df_ultima_decada = df[df['release_date'].between(pd.to_datetime('2011'), pd.to_datetime('2020'))]
df_ultima_decada.head()

# %%
# cria variável com agrupamento de album por popularidade e ordenado do maior para o menor - TOP 10
df_por_album = df_ultima_decada.groupby('album')['popularity'].sum().sort_values(ascending=False).head(10)
df_por_album

# %%
# cria variável que calcula a porcentagem de popularidade dos top 10
# cria variável de labels
# cria variável de size
total_popularidade = df_por_album.sum()

df_porcentagem = df_por_album / total_popularidade * 100

labels = df_porcentagem.index.tolist()
size = df_porcentagem.values.tolist()

# estrutura do gráfico
figura, grafico = plt.subplots(figsize = (18,6))
grafico.pie(size, autopct='%1.1f%%')
grafico.axis('equal')
plt.title = ('Porcentagem de Popularidade de Álguns na Última Década (Top 10 Álguns)')
plt.legend(labels, loc = 'best')
plt.show()

# %% [markdown]
# #### Outliers

# %% [markdown]
# - São valores extremos na base seja para cima ou para baixo

# %%
# cria o boxplot para ajudar na identificação dos outliers
sns.set(style='whitegrid')
fig, axes = plt.subplots(figsize = (12,6))
sns.boxplot(x = 'duracao_minutos', data = df)
axes.set_title('Boxplot')
plt.show()

# %% [markdown]
# - O boxplot ajuda a visualizar também a distribuição de dados
#     - Onde está o retângulo azul, é a maior concentração de dados: neste caso a média de range de duração das músicas vai em torno de 3 a 6 minutos
#     - Dentro da caixa azul, há uma linha que divide a caixa, ela representa a média -> média de 4 minutos por música
#     - Os pontinhos pretos são os outliers. Neste gráfico temos outliers para cima e para baixo
#         - Quanto mais pontinhos pretos no gráfico, maior é a quantidade de dados fora do padrão
# - Tirar ou não tirar os outliers da base, dependem da necessidade de análise. Não é uma regra que diz que outlier precisa ser tirado.

# %%
# criação de um gráfico violino
fig, axes = plt.subplots(figsize=(12,6))
sns.violinplot(x = 'duracao_minutos', data = df, color = 'gray')
axes.set_title('Gráfico de Violino')
plt.show()

# %% [markdown]
# - Este gráfico traz uma representação diferente do boxplot
#     - Aqui também, a maior concentração de dados, é onde tem o corpo maior em cinza
#     - Os outliers são representados na "linha" mais fina do gráfico

# %%
# juntando os dois gráficos
fig, ax = plt.subplots(figsize = (8,6))

sns.violinplot(x = 'duracao_minutos', data = df, ax = ax, color = 'lightgray')
sns.boxplot(x = 'duracao_minutos', data = df, ax = ax, whis=1.5, color = 'blue')

axes.set_title('Visualização Violino & Boxplot')
plt.show()

# %%
df.head()

# %%
# verificar (através de função) se a músca é ao vivo ou não, baseado no liveness (qto mais próximo de 1, é ao vivo)
def classifica_musica_ao_vivo(df):
    if df['liveness'] >= 0.8:
        return True
    else:
        return False

# %%
#cria uma coluna ao vivo e aplica a função para classificar o que é ao vivo ou não)
df['ao_vivo'] = df.apply(classifica_musica_ao_vivo, axis = 1)
df

# %%
# agrupa música ao vivo e musica nao ao vivo
df.groupby('ao_vivo')['ao_vivo'].count()

# %%
# cria um dataframe para cada grupo (ao vivo e não ao vivo)
df_gravado_em_estudio = df[df['ao_vivo'] == False]
df_show_ao_vivo = df[df['ao_vivo'] == True]

# %%
df_gravado_em_estudio.head()

# %%
df_show_ao_vivo.head()

# %%
# calcula e printa média de músicas ao vivo
print('Média das músicas ao vivo: ', df_show_ao_vivo['duracao_minutos'].mean())

# %%
# calcula e printa média de músicas ao vivo
print('Média das músicas em estúdio: ', df_gravado_em_estudio['duracao_minutos'].mean())

# %% [markdown]
# Olhando para esses dados, é possível confirmar que os _outliers_ que apareceram nos gráficos acima, de fato existem. São as músicas de show ao vivo. Se fossem retirados, perderíamos todos dados de música ao vivo.
