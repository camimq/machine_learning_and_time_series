
# Machine Learning com Python

## Aula 1 | Bibliotecas fundamentais e Primeiros Passos em ML

- Quando estamos trabalhando com a importação de uma planilha excel para o Python, existe um parâmentro `sheet_name`, que permite acessar uma _sheet_ específica dentro do arquivo, caso haja mais que um.
- Estatística descritiva serve para entender como estão os dados da base. O comando `df.describe()`, vai trazer todas as informações iniciais, em cima das colunas numéricas da base.
- E possível transpor a tabela da estatística descritiva, utilizando o `.T`; para algumas pessoas fica mais fácil (ex: `df.describe().T`).
- A "tabela" de estatística descritiva, é uma matriz de dados. Matriz é composta por linhas e colunas.
- Variáveis categóricas, são aquelas que não conseguimos analisar de forma numérica / estatística; forma "categorias" dentro do banco de dados.
- Array é uma lista de números.
    - uma lista de array, é sempre criada com `[]` e tuplas com `()`.

### `df_csv` Tomato base

- **Shape (`df_csv.shape`):** 2741 linhas e 6 colunas
- **Info (`df_csv.info()`):** 
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2741 entries, 0 to 2740
    Data columns (total 6 columns):
     #   Column   Non-Null Count  Dtype  
    ---  ------   --------------  -----  
     0   Date     2741 non-null   object 
     1   Unit     2741 non-null   object 
     2   Minimum  2741 non-null   int64  
     3   Maximum  2741 non-null   int64  
     4   Average  2741 non-null   float64
     5   Market   2741 non-null   object 
    dtypes: float64(1), int64(2), object(3)
    memory usage: 128.6+ KB
- **Estatística descritiva dos dados (`df_csv.describe()`):**
    	    Minimum	    Maximum	    Average
    count	2741.000000	2741.000000	2741.000000
    mean	35.089748	41.281284	38.185516
    std	    16.648425	17.364135	16.970949
    min	    8.000000	12.000000	10.000000
    25%	    22.000000	30.000000	25.000000
    50%	    30.000000	38.000000	35.000000
    75%	    45.000000	50.000000	47.500000
    max	    115.000000	120.000000	117.500000

### `df_excel` Chess base

- **Shape (`df_excel.shape`):** 20058 linhas e 14 colunas
- **Info (`df_excel.info()`):**
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 20058 entries, 0 to 20057
    Data columns (total 14 columns):
    #   Column          Non-Null Count  Dtype  
    ---  ------          --------------  -----  
    0   rated           20058 non-null  bool   
    1   created_at      20058 non-null  float64
    2   last_move_at    20058 non-null  float64
    3   turns           20058 non-null  int64  
    4   victory_status  20058 non-null  object 
    5   winner          20058 non-null  object 
    6   increment_code  20058 non-null  object 
    7   white_id        20058 non-null  object 
    8   white_rating    20058 non-null  int64  
    9   black_id        20058 non-null  object 
    10  black_rating    20058 non-null  int64  
    11  opening_eco     20058 non-null  object 
    12  opening_name    20058 non-null  object 
    13  opening_ply     20058 non-null  int64  
    dtypes: bool(1), float64(2), int64(4), object(7)
    memory usage: 2.0+ MB

## Aula 2 | Análise Exploratória de Dados (EDA)

### Problemas no código

- [ ] Verificar a legenda do gráfico de Densidade da aula IX.
- [x] Verificar a criação de matriz de sentimento da aula XII.
- [x] Verificar o heatmap da aula XII.
- [ ] Verificar scatterplot da aula XII.
