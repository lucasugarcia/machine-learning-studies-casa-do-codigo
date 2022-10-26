import pandas as pd
from collections import Counter
import numpy as np
from sklearn.model_selection import cross_val_score
import nltk

def vetorizar_texto(texto, tradutor, stemmer):
    vetor = [0] * len(tradutor)
    
    for palavra in texto:

        if len(palavra) <= 0:
            continue

        raiz = stemmer.stem(palavra)

        if raiz not in tradutor:
            continue
        
        posicao = tradutor[raiz]
        vetor[posicao] += 1

    return vetor

classificacoes = pd.read_csv('emails.csv')

textos_puros = classificacoes['email']
frases = textos_puros.str.lower()

textos_quebrados = [nltk.tokenize.word_tokenize(frase) for frase in frases]

textos_quebrados = textos_puros.str.lower().str.split(' ')

stopwords = nltk.corpus.stopwords.words('portuguese')
stemmer = nltk.stem.RSLPStemmer()

dicionario = set()

for lista in textos_quebrados:
    validas = [stemmer.stem(palavra) for palavra in lista if palavra not in stopwords and len(palavra) > 2]
    dicionario.update(validas)

total_palavras = len(dicionario)

tuplas = zip(dicionario, range(total_palavras))
tradutor = {palavra:indice for palavra, indice in tuplas}

vetores_texto = [vetorizar_texto(texto, tradutor, stemmer) for texto in textos_quebrados]

marcas = classificacoes['classificacao']

x = vetores_texto
y = marcas

porcentagem_treino = 0.8
tamanho_treino = int(porcentagem_treino * len(y))
tamanho_validacao = len(y) - tamanho_treino

treino_dados = x[0:tamanho_treino]
treino_marcacoes = y[0:tamanho_treino]

validacao_dados = x[tamanho_treino:]
validacao_marcacoes = y[tamanho_treino:]

def fit_and_predict(nome, modelo, treino_dados, treino_marcacoes):
    k = 10
    scores = cross_val_score(modelo, treino_dados, treino_marcacoes, cv = k)
    taxa_de_acerto = np.mean(scores)

    mensagem = "Taxa de acerto do {0}: {1}".format(nome, taxa_de_acerto)
    print(mensagem)

    return taxa_de_acerto

resultados = {}

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
modelo_one_vs_rest = OneVsRestClassifier(LinearSVC(random_state = 0))
resultado_one_vs_rest = fit_and_predict("OneVsRest", modelo_one_vs_rest, treino_dados, treino_marcacoes)
resultados[resultado_one_vs_rest] = modelo_one_vs_rest

from sklearn.multiclass import OneVsOneClassifier
modelo_one_vs_one = OneVsOneClassifier(LinearSVC(random_state = 0))
resultado_one_vs_one = fit_and_predict("OneVsOne", modelo_one_vs_one, treino_dados, treino_marcacoes)
resultados[resultado_one_vs_one] = modelo_one_vs_one

from sklearn.naive_bayes import MultinomialNB
modelo_multinomial = MultinomialNB()
resultado_multinomial = fit_and_predict("MultinomialNB", modelo_multinomial, treino_dados, treino_marcacoes)
resultados[resultado_multinomial] = modelo_multinomial

from sklearn.ensemble import AdaBoostClassifier
modelo_adaboost = AdaBoostClassifier(random_state = 0)
resultado_adaboost = fit_and_predict("AdaBoostClassifier", modelo_adaboost, treino_dados, treino_marcacoes)
resultados[resultado_adaboost] = modelo_adaboost

maximo = max(resultados)
vencedor = resultados[maximo]

print("Vencedor: ")
print(vencedor)

vencedor.fit(treino_dados, treino_marcacoes)

def teste_real(modelo, validacao_dados, validacao_marcacoes):
    resultado = modelo.predict(validacao_dados)
    acertos = resultado == validacao_marcacoes

    total_de_acertos = sum(acertos)
    total_de_elementos = len(validacao_marcacoes)

    taxa_de_acerto = 100.0 * total_de_acertos / total_de_elementos

    msg = "Taxa de acerto do vencedor entre os dois algoritmos no mundo real: {0}".format(taxa_de_acerto)
    
    print(msg)

teste_real(vencedor, validacao_dados, validacao_marcacoes)

acerto_base = max(Counter(validacao_marcacoes).values())
taxa_de_acerto_base = 100.0 * acerto_base / len(validacao_marcacoes)
print("Taxa de acerto base: %f" % taxa_de_acerto_base)

total_de_elementos = len(validacao_dados)
print("Total de teste: %d" % total_de_elementos)