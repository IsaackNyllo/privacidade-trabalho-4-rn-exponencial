"""Trabalho 4 de Privacidade de Dados: r-N normal contra r-N com o mecanismo Exponencial.

A ideia: o classificador r-N olha todo mundo da base de treino que está a até r de distância
da pessoa de teste e conta quantos são de cada classe. Sem privacidade, ele escolhe a classe
com a maior contagem. Com privacidade, ele SORTEIA a classe, dando mais chance a quem tem
mais contagem (mecanismo Exponencial). No fim comparamos quanto de acerto se perde.

Como usar:
    python rn_privado.py            lê dados/adult.csv e grava tudo em resultados/
    python rn_privado.py --teste    só confere, rapidinho, se o sorteio e o r-N estão certos

Referência: Gursoy, Inan, Nergiz e Saygin, "Differentially private nearest neighbor
classification", Data Mining and Knowledge Discovery, 2017 (Definição 9 e Algoritmo 2).
"""
import csv
import sys
from pathlib import Path

import numpy as np

# Rodando como .exe, a pasta de referência é a do executável; rodando como .py, a do script.
EXE = getattr(sys, "frozen", False)
AQUI = Path(sys.executable).parent if EXE else Path(__file__).parent
BASE = AQUI / "dados" / "adult.csv"
SAIDA = AQUI / "resultados"

# Colunas que o enunciado manda jogar fora, e a coluna que queremos prever.
REMOVER = ["fnlwgt", "education", "capital-gain", "capital-loss", "hours-per-week"]
ALVO = "income"
NUMERICAS = ["age", "educational-num"]   # as demais viram One-Hot

RAIOS = [0.5, 1, 1.5, 2, 6]
EPSILONS = [0.1, 0.5, 1, 5, 10, 100]   # cada valor é o orçamento de privacidade do teste INTEIRO
TAM_TESTE = 1000             # 500 de cada classe
SENSIBILIDADE = 1            # colocar ou tirar uma pessoa mexe numa contagem em no máximo 1
SEMENTE = 2026               # deixa o sorteio repetível: rodando de novo, sai o mesmo resultado
REPETICOES = 20              # o sorteio da classe é aleatório; repetimos 20 vezes para o gráfico não depender da sorte
LOTE = 200                   # quantas pessoas de teste calculamos de uma vez, para não estourar a memória
TOLERANCIA = 1e-9            # os dados agora são decimais; sem folga, "distância = r" poderia falhar por arredondamento


def carregar():
    """Lê a base e faz o pré-processamento pedido no enunciado.

    Joga fora as linhas com "?" e as cinco colunas que não vamos usar. Cada coluna de
    texto vira One-Hot (uma coluna 0/1 por categoria, categorias em ordem alfabética).
    Idade e escolaridade ficam entre 0 e 1 pelo MinMax: (valor - mínimo) / (máximo - mínimo),
    com mínimo e máximo da base inteira (decisão registrada: no treino dá o mesmo).
    """
    with open(BASE, encoding="utf-8", newline="") as f:
        # Guardamos também o número da linha no arquivo (o cabeçalho é a linha 1),
        # para dar para achar cada pessoa no adult.csv depois.
        validas = [(n, l) for n, l in enumerate(csv.DictReader(f), start=2)
                   if "?" not in (v.strip() for v in l.values())]
    linha_csv = np.array([n for n, _ in validas])
    linhas = [l for _, l in validas]

    colunas = [c for c in linhas[0] if c not in REMOVER and c != ALVO]
    blocos, nomes = [], []
    for c in colunas:
        if c in NUMERICAS:
            v = np.array([float(l[c]) for l in linhas])
            blocos.append(((v - v.min()) / (v.max() - v.min()))[:, None])
            nomes.append(c)
        else:
            valores = [l[c].strip() for l in linhas]
            cats = sorted(set(valores))
            blocos.append(np.array([[v == cat for cat in cats] for v in valores], dtype=np.float64))
            nomes += [f"{c}={cat}" for cat in cats]
    X = np.hstack(blocos)

    classes = sorted({l[ALVO].strip() for l in linhas})
    y = np.array([classes.index(l[ALVO].strip()) for l in linhas], dtype=np.int64)
    return X, y, linha_csv, classes, colunas, nomes


def dividir(X, y, rng):
    """Sorteia o teste: TAM_TESTE/2 pessoas de cada classe. O resto é treino.
    Devolve também as posições de quem foi para o teste, para os arquivos de saída."""
    por_classe = TAM_TESTE // len(np.unique(y))
    teste = np.sort(np.concatenate([rng.choice(np.flatnonzero(y == c), por_classe, replace=False)
                                    for c in np.unique(y)]))
    e_treino = np.ones(len(X), dtype=bool)
    e_treino[teste] = False
    return X[e_treino], y[e_treino], X[teste], y[teste], teste


def vizinhanca(Xtr, ytr, Xte, n_classes, raios=RAIOS):
    """Conta, para cada pessoa de teste e cada raio, quantos treinos de cada classe estão
    a uma distância de até r. Devolve um array (teste, raios, classes).

    A distância é calculada uma vez só, e as contagens de todos os raios saem da mesma passada.
    """
    contagens = np.empty((len(Xte), len(raios), n_classes), dtype=np.int64)
    tr2 = (Xtr ** 2).sum(1)
    for i in range(0, len(Xte), LOTE):
        te = Xte[i:i + LOTE]
        # Distância euclidiana ao quadrado, usando |a|² + |b|² - 2ab. Sem raiz: comparamos com r².
        # O clip em 0 tira os pequenos negativos que o arredondamento cria quando a distância é 0.
        d2 = np.clip((te ** 2).sum(1)[:, None] + tr2[None, :] - 2 * te @ Xtr.T, 0, None)
        for j, r in enumerate(raios):
            dentro = d2 <= r ** 2 + TOLERANCIA
            for c in range(n_classes):
                contagens[i:i + LOTE, j, c] = (dentro & (ytr == c)).sum(1)
    return contagens


def rn_normal(contagens, rng):
    """r-N tradicional (Definição 9 do artigo): a classe com a maior contagem.
    Em caso de empate (inclusive 0 a 0, quando ninguém está no raio), sorteia entre as empatadas.
    Assim nenhuma classe ganha vantagem de graça no teste balanceado."""
    empatadas = contagens == contagens.max(1, keepdims=True)
    return np.where(empatadas, rng.random(contagens.shape), -1).argmax(1)


def eps_consulta(epsilon, n_teste):
    """ε′ de cada pessoa de teste. Cada uma é uma consulta à mesma base de treino, então
    os gastos se somam (composição sequencial): n_teste * (ε / n_teste) = ε.
    Não dividimos pelo número de classes: o Exponencial faz UMA escolha por pessoa."""
    return epsilon / n_teste


def exponencial(contagens, eps_linha, rng):
    """Mecanismo Exponencial (Algoritmo 2 do artigo), uma escolha de classe por pessoa de teste.

    Utilidade: q = contagem da classe no raio. Sensibilidade: 1.
    A chance da classe I é  exp(ε′·c_I / (2·sensibilidade))  dividido pela soma disso para todas as classes.
    Para a conta não estourar com contagens grandes, subtraímos o maior expoente antes do exp
    (as probabilidades não mudam). A classe é sorteada comparando um número uniforme
    com a probabilidade acumulada. Devolve as classes sorteadas e as probabilidades.
    """
    expoente = eps_linha * contagens / (2 * SENSIBILIDADE)
    peso = np.exp(expoente - expoente.max(1, keepdims=True))
    prob = peso / peso.sum(1, keepdims=True)
    u = rng.random(len(contagens))
    escolha = (u[:, None] >= prob.cumsum(1)).sum(1)
    return np.minimum(escolha, contagens.shape[1] - 1), prob   # o minimum protege de arredondamento no fim da fila


def gravar(nome, linhas_teste, yte, pred, classes, contagens, prob=None):
    """Salva um CSV com o rótulo real, a previsão e as contagens de cada pessoa de teste
    (e, no caso privado, as probabilidades que o mecanismo usou para sortear)."""
    with open(SAIDA / nome, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        extra = [f"prob_{c}" for c in classes] if prob is not None else []
        w.writerow(["linha_no_adult_csv", "rotulo_real", "predicao"] + [f"contagem_{c}" for c in classes] + extra)
        for i, (r, p) in enumerate(zip(yte, pred)):
            linha = [linhas_teste[i], classes[r], classes[p]] + [int(v) for v in contagens[i]]
            if prob is not None:
                linha += [f"{v:.6f}" for v in prob[i]]
            w.writerow(linha)


def grafico(acc_normal, medias, desvios):
    """Um gráfico só: acurácia por ε, uma cor por raio. Linha cheia com barra de desvio = r-N privado;
    linha tracejada da mesma cor = r-N sem privacidade daquele raio."""
    import matplotlib
    matplotlib.use("Agg")   # desenha direto no arquivo, sem abrir janela
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    x = range(len(EPSILONS))
    for j, r in enumerate(RAIOS):
        cor = f"C{j}"
        ax.errorbar(x, medias[:, j], yerr=desvios[:, j], fmt="o-", color=cor, lw=2, capsize=3,
                    label=f"r = {r}  (sem privacidade: {acc_normal[j]:.1%})")
        ax.axhline(acc_normal[j], ls="--", color=cor, lw=1.5)
    ax.plot([], [], "o-", color="gray", label=f"linha cheia: privado, média de {REPETICOES} sorteios\n(barra = desvio padrão)")
    ax.plot([], [], "--", color="gray", label="tracejada: sem privacidade, mesma cor")
    ax.set_xticks(x, [f"{e}" for e in EPSILONS])
    ax.set_xlabel("ε (orçamento de privacidade)")
    ax.set_ylabel("acurácia no teste (1000 tuplas, 500 por classe)")
    ax.set_ylim(0.45, 0.8)
    ax.set_title("Acurácia do r-N privado (Exponencial) por ε e por raio")
    ax.grid(alpha=.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=7)
    fig.tight_layout()
    fig.savefig(SAIDA / "acuracia.png")


def teste():
    """Confere o básico antes de confiar nos números grandes."""
    rng = np.random.default_rng(0)

    # As probabilidades somam 1.
    _, p = exponencial(np.array([[3, 7], [0, 0], [100, 2]]), 0.5, rng)
    assert np.allclose(p.sum(1), 1), p

    # ε enorme: quase sempre sai a classe de maior contagem.
    pred, _ = exponencial(np.tile([10, 3], (2000, 1)), 50.0, rng)
    assert (pred == 0).mean() > 0.999, (pred == 0).mean()

    # Contagens iguais, e também 0 a 0 (raio vazio): perto de 50/50.
    for c in ([5, 5], [0, 0]):
        pred, _ = exponencial(np.tile(c, (20000, 1)), 1.0, rng)
        assert abs(pred.mean() - 0.5) < 0.02, (c, pred.mean())

    # O r-N sem privacidade também sorteia no empate 0 a 0, e no resto escolhe a maior.
    pred = rn_normal(np.tile([0, 0], (20000, 1)), rng)
    assert abs(pred.mean() - 0.5) < 0.02, pred.mean()
    assert rn_normal(np.array([[4, 9], [8, 2]]), rng).tolist() == [1, 0]

    # exp estável: contagens de 44222 com ε = 100 (ε′ = 0,1) não dão NaN nem infinito.
    pred, p = exponencial(np.array([[44222, 30000], [44222, 44221], [0, 44222]]), eps_consulta(100, 1000), rng)
    assert np.isfinite(p).all() and np.allclose(p.sum(1), 1), p
    assert p[0, 0] > 0.999 and p[2, 1] > 0.999 and 0.5 < p[1, 0] < 0.6, p

    # ε′ = ε / 1000 para um teste de 1000 pessoas.
    assert eps_consulta(1, 1000) == 0.001 and eps_consulta(100, 1000) == 0.1

    # Uma base de brinquedo: três pontos da classe 0 num canto e dois da classe 1 no outro.
    Xtr = np.array([[0, 0], [0, 1], [1, 0], [9, 9], [9, 8]], dtype=float)
    ytr = np.array([0, 0, 0, 1, 1])
    cont = vizinhanca(Xtr, ytr, np.array([[0, 0], [9, 9], [5, 5]], dtype=float), 2, raios=[1, 1.5, 100])
    assert cont[0].tolist() == [[3, 0], [3, 0], [3, 2]], cont[0]   # r=1 pega o ponto na borda (distância = r exata)
    assert cont[1].tolist() == [[0, 2], [0, 2], [3, 2]], cont[1]
    assert cont[2].tolist() == [[0, 0], [0, 0], [3, 2]], cont[2]   # no meio ninguém está perto: 0 a 0
    print("teste ok")


def main():
    rng = np.random.default_rng(SEMENTE)
    X, y, linha_csv, classes, colunas, nomes = carregar()
    Xtr, ytr, Xte, yte, pos_teste = dividir(X, y, rng)
    linhas_teste = linha_csv[pos_teste]
    print(f"base: {len(X)} tuplas, {len(colunas)} atributos, {X.shape[1]} dimensões depois do One-Hot")
    print(f"treino: {len(Xtr)}  teste: {len(Xte)} ({np.bincount(yte).tolist()} por classe)  classes: {classes}")
    SAIDA.mkdir(exist_ok=True)

    contagens = vizinhanca(Xtr, ytr, Xte, len(classes))   # (teste, raios, classes)

    # Análise (não privada): quantos vizinhos cada raio enxerga.
    resumo = []
    for j, r in enumerate(RAIOS):
        total = contagens[:, j].sum(1)
        resumo.append(f"r = {r}: vizinhos por pessoa de teste: mínimo {total.min()}, mediana {int(np.median(total))}, "
                      f"máximo {total.max()}; sem nenhum vizinho: {(total == 0).sum()} de {len(total)}")
    print("\n".join(resumo))
    (SAIDA / "vizinhos.txt").write_text("\n".join(resumo) + "\n", encoding="utf-8")

    linhas = [["modelo", "r", "epsilon", "epsilon_por_consulta", "acuracia_media", "desvio_padrao",
               "acuracia_1o_sorteio", "acuracia_esperada"]]

    # r-N sem privacidade (referência; NÃO é privado).
    acc_normal = []
    for j, r in enumerate(RAIOS):
        pred = rn_normal(contagens[:, j], rng)
        acc_normal.append((pred == yte).mean())
        gravar(f"predicoes_rn_r_{r}.csv", linhas_teste, yte, pred, classes, contagens[:, j])
        linhas.append(["rn", r, "", "", f"{acc_normal[j]:.4f}", "", "", ""])
        print(f"r-N sem privacidade, r = {r}: acurácia {acc_normal[j]:.4f}")

    # r-N privado. Para cada (ε, r) sorteamos as classes várias vezes: o arquivo de
    # predições guarda o primeiro sorteio, e o gráfico usa a média de todos.
    medias = np.empty((len(EPSILONS), len(RAIOS)))
    desvios = np.empty_like(medias)
    for i, eps in enumerate(EPSILONS):
        eps_linha = eps_consulta(eps, len(Xte))
        for j, r in enumerate(RAIOS):
            accs = []
            for rep in range(REPETICOES):
                pred, prob = exponencial(contagens[:, j], eps_linha, rng)
                accs.append((pred == yte).mean())
                if rep == 0:
                    gravar(f"predicoes_rn_privado_eps_{eps}_r_{r}.csv", linhas_teste, yte, pred, classes,
                           contagens[:, j], prob)
            esperada = prob[np.arange(len(yte)), yte].mean()   # as probabilidades não dependem do sorteio
            medias[i, j], desvios[i, j] = np.mean(accs), np.std(accs)
            linhas.append(["rn_privado", r, eps, f"{eps_linha:g}", f"{medias[i, j]:.4f}", f"{desvios[i, j]:.4f}",
                           f"{accs[0]:.4f}", f"{esperada:.4f}"])
            print(f"ε = {eps:>5}, r = {r:>3}: acurácia média {medias[i, j]:.4f} ± {desvios[i, j]:.4f}  "
                  f"(1º sorteio {accs[0]:.4f}, esperada {esperada:.4f})")

    with open(SAIDA / "acuracias.csv", "w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows(linhas)
    grafico(acc_normal, medias, desvios)
    print(f"arquivos em {SAIDA}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")   # o terminal do Windows engasga com o "ε" sem isto
    if "--teste" not in sys.argv and not BASE.exists():
        print(f"Não achei a base de dados: {BASE}\nColoque o adult.csv na pasta 'dados', ao lado do programa.")
        if EXE:
            input("\nAperte Enter para fechar.")
        sys.exit(1)
    teste() if "--teste" in sys.argv else main()
    if EXE:
        input("\nPronto. Aperte Enter para fechar.")   # com dois cliques no .exe, a janela não some antes de dar para ler
