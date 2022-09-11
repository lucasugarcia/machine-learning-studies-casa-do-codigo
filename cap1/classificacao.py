from sklearn.naive_bayes import MultinomialNB

# [é gordinho?, tem perna curta?, late?]
porco1 = [1, 1, 0]
porco2 = [1, 1, 0]
porco3 = [1, 1, 0]

cachorro1 = [1, 1 , 1]
cachorro2 = [0, 1 , 1]
cachorro3 = [0, 1 , 1]

dados = [porco1, porco2, porco3, cachorro1, cachorro2, cachorro3]

# 1 representa porco
# -1 representa cachorro
marcacoes = [1, 1, 1, -1, -1, -1]

animal_misterioso = [1, 1, 1]

modelo = MultinomialNB()
modelo.fit(dados, marcacoes)

print(modelo.predict([animal_misterioso]))