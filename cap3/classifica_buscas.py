import pandas as pd
from sklearn.naive_bayes import MultinomialNB

df = pd.read_csv('buscas.csv')

x_df = df[['home', 'busca', 'logado']]
y_df = df['comprou']

x_dummies_df = pd.get_dummies(x_df)
y_dummies_df = y_df

x = x_dummies_df.values
y = y_dummies_df.values

porcentagem_treino = 0.9
tamanho_treino = int(porcentagem_treino * len(y))
tamanho_teste = len(y) - tamanho_treino

treino_dados = x[:tamanho_treino]
treino_marcacoes = y[:tamanho_treino]

teste_dados = x[-tamanho_teste:]
teste_marcacoes = y[-tamanho_teste:]

modelo = MultinomialNB()
modelo.fit(treino_dados, treino_marcacoes)

resultado = modelo.predict(teste_dados)

print(resultado)

diferencas = resultado - teste_marcacoes

acertos = [d for d in diferencas if d == 0]
total_acertos = len(acertos)
total_elementos = len(teste_dados)

taxa_acerto = 100.0 * (total_acertos / total_elementos)

print(taxa_acerto)
print(total_elementos)