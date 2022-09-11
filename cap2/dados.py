import csv

def carregar_acessos():
    # Dados
    x = []

    # Marcações
    y = []

    arquivo = open('acesso.csv', 'r')

    leitor = csv.reader(arquivo)

    next(leitor)

    for home, como_funciona, contato, comprou in leitor:
        dado = [int(home), int(como_funciona), int(contato)]

        x.append(dado)
        y.append(int(comprou))

    return x, y