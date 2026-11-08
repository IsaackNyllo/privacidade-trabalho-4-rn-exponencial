"""Trabalho 4 (extra): quatro formas de rodar o r-N com o mecanismo Exponencial, para ver o que melhora a acurácia.

    A  enunciado     ε′ = ε/1000 por pessoa de teste (o que o rn_privado.py já faz).
    B  algoritmo3    ε′ = ε/MCS por teste, MCS = clique máxima da componente conexa do teste no grafo de testes
                     (aresta se distância <= 2r); teste isolado usa ε/1.
    C  pesos         pesos por classe (contra o desbalanceamento), medidos no treino com Laplace.
    D  algoritmo3+pesos   os pesos de C com o ε′ de B.

Por que B vale (Algoritmo 3 do artigo): um ponto de treino t só muda as contagens dos testes a distância <= r dele.
Esses testes estão a <= 2r entre si, logo formam uma clique. Dois testes em componentes diferentes estão a > 2r,
então não têm o mesmo ponto de treino no raio: t só afeta testes de UMA componente, no máximo MCS_comp deles, cada um
com ε/MCS_comp: total <= ε (composição sequencial dentro da componente, paralela entre componentes).

Por que C vale: gasta 0,01·ε medindo as contagens das classes no treino (histograma com Laplace, sensibilidade 1,
usado uma vez só) e 0,99·ε/1000 em cada resposta. Peso w_I = (menor contagem ruidosa)/(contagem ruidosa de I) <= 1,
nota q_I = w_I·c_I, então a sensibilidade continua 1. Total = 0,01ε + 0,99ε = ε. D: 0,01ε + 0,99ε/MCS_comp.
Os pesos são pós-processamento do Laplace, então não custam mais privacidade.

Como usar:
    python rn_melhorias.py            grava tudo em resultados/melhorias/
    python rn_melhorias.py --teste    confere o básico
"""
import csv
import sys
import time

import numpy as np

import rn_privado as rp
from rn_privado import (carregar, dividir, vizinhanca, rn_normal, exponencial, eps_consulta, gravar,
                        RAIOS, EPSILONS, REPETICOES, SEMENTE, TOLERANCIA)

sys.setrecursionlimit(10000)   # a clique de 1000 vértices recursa 1000 níveis
SAIDA = rp.SAIDA / "melhorias"
FRACAO_PESOS = 0.01            # parte de ε gasta medindo as contagens das classes (variantes C e D)
VARIANTES = {"A": "enunciado", "B": "algoritmo3", "C": "pesos", "D": "algoritmo3_pesos"}


def max_clique(nodes, adj):
    """Clique máxima exata: Bron-Kerbosch com bitsets e poda por coloração gulosa (Tomita).
    adj[v] é o conjunto de vizinhos de v em bits."""
    best = [0]

    def color_bound(P):
        order, bounds, k, U = [], [], 0, P
        while U:
            k += 1
            Q = U
            while Q:
                v = (Q & -Q).bit_length() - 1
                Q &= ~(1 << v) & ~adj[v]
                U &= ~(1 << v)
                order.append(v)
                bounds.append(k)
        return order, bounds

    def expand(R, P):
        order, bounds = color_bound(P)
        for i in range(len(order) - 1, -1, -1):
            if R + bounds[i] <= best[0]:
                return
            v = order[i]
            newP = P & adj[v]
            if newP:
                expand(R + 1, newP)
            elif R + 1 > best[0]:
                best[0] = R + 1
            P &= ~(1 << v)

    P = 0
    for v in nodes:
        P |= 1 << v
    expand(0, P)
    return best[0]


def mcs_do_teste(Xte, r):
    """Para cada teste, a clique máxima da componente conexa dele (grafo: aresta se distância <= 2r).
    Devolve (vetor de MCS por teste, número de componentes)."""
    n = len(Xte)
    d2 = np.clip((Xte ** 2).sum(1)[:, None] + (Xte ** 2).sum(1)[None] - 2 * Xte @ Xte.T, 0, None)
    A = d2 <= 4 * (r ** 2 + TOLERANCIA)   # 2r com a mesma folga do raio: (2·sqrt(r²+tol))²
    np.fill_diagonal(A, False)
    adj = [sum(1 << int(j) for j in np.flatnonzero(A[i])) for i in range(n)]
    visto = np.zeros(n, bool)
    mcs, comps = np.ones(n, dtype=np.int64), 0
    for i in range(n):
        if visto[i]:
            continue
        comps += 1
        pilha, comp = [i], []
        visto[i] = True
        while pilha:
            u = pilha.pop()
            comp.append(u)
            viz = np.flatnonzero(A[u] & ~visto)
            visto[viz] = True
            pilha.extend(viz.tolist())
        mcs[comp] = max_clique(comp, adj) if len(comp) > 1 else 1
    return mcs, comps


def pesos_exatos(ytr, n_classes):
    """Pesos sem ruído: menor contagem / contagem da classe. O maior peso é 1."""
    n = np.bincount(ytr, minlength=n_classes).astype(float)
    return n.min() / n


def pesos_laplace(ytr, n_classes, epsilon, rng):
    """Pesos medidos com privacidade: contagens do treino + Laplace(1/ε0), ε0 = 0,01·ε, corte em >= 1."""
    n = np.bincount(ytr, minlength=n_classes) + rng.laplace(0, 1 / (FRACAO_PESOS * epsilon), n_classes)
    n = np.maximum(n, 1)
    return n.min() / n


def sorteios(cont, yte, eps_roleta, rng, pesos=lambda: 1.0):
    """REPETICOES sorteios do Exponencial com nota q = peso·contagem. `pesos()` devolve os pesos de cada sorteio
    (os Laplace são sorteados de novo a cada um). Devolve acurácias, previsão e prob. do 1º sorteio, peso do 1º."""
    accs = []
    for rep in range(REPETICOES):
        w = pesos()
        pred, prob = exponencial(cont * w, np.reshape(eps_roleta, (-1, 1)) if np.ndim(eps_roleta) else eps_roleta, rng)
        accs.append((pred == yte).mean())
        if rep == 0:
            primeiro = (pred, prob, np.ones(cont.shape[1]) * w)
    return accs, *primeiro


def gravar_com_peso(nome, linhas_teste, yte, pred, classes, cont, prob, peso_le50k, mcs_comp=None):
    """O gravar do rn_privado, mais peso_le50k (peso usado para <=50K) e, se houver, mcs_componente de cada teste."""
    gravar(f"melhorias/{nome}", linhas_teste, yte, pred, classes, cont, prob)
    with open(SAIDA / nome, encoding="utf-8", newline="") as f:
        linhas = list(csv.reader(f))
    linhas[0] += ["peso_le50k"] + (["mcs_componente"] if mcs_comp is not None else [])
    for k, l in enumerate(linhas[1:]):
        l += [f"{peso_le50k:.6f}"] + ([int(mcs_comp[k])] if mcs_comp is not None else [])
    with open(SAIDA / nome, "w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows(linhas)


def grafico(medias, ref_rn, ref_pesos):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ROTULO = {"A": "enunciado (ε/1000)", "B": "Algoritmo 3", "C": "pesos", "D": "Algoritmo 3 + pesos"}
    plt.rcParams.update({"font.size": 11})
    fig, axs = plt.subplots(2, 3, figsize=(12, 6.5), dpi=150, sharey=True)
    x = range(len(EPSILONS))
    cores = {"A": "C0", "B": "C1", "C": "C2", "D": "C3"}
    for j, r in enumerate(RAIOS):
        ax = axs.flat[j]
        for v in "ABCD":
            ax.plot(x, [medias[v][i, j] for i in x], "o-", color=cores[v], lw=1.6, ms=4, label=f"{v}: {ROTULO[v]}")
        ax.axhline(ref_rn[j], ls="--", color="gray", lw=1.3, label="E: sem privacidade")
        ax.axhline(ref_pesos[j], ls=":", color="black", lw=1.3, label="F: sem privacidade, com pesos")
        ax.set_xticks(x, [f"{e}" for e in EPSILONS])
        ax.set_title(f"r = {r}", fontsize=13)
        ax.grid(alpha=.3)
        if j >= 2:
            ax.set_xlabel("ε (orçamento de privacidade)")
        if j % 3 == 0:
            ax.set_ylabel("acurácia média no teste")
    h, l = axs.flat[0].get_legend_handles_labels()
    axs.flat[5].axis("off")
    axs.flat[5].legend(h, l, loc="center", fontsize=12, frameon=False)
    fig.suptitle("r-N privado: enunciado, Algoritmo 3 e pesos (média de 20 sorteios)")
    fig.tight_layout()
    fig.savefig(SAIDA / "melhorias.png")


def teste():
    rng = np.random.default_rng(0)
    # Clique máxima num grafo conhecido: um triângulo (0,1,2) mais o caminho 2-3-4.
    arestas = [(0, 1), (0, 2), (1, 2), (2, 3), (3, 4)]
    adj = [0] * 5
    for a, b in arestas:
        adj[a] |= 1 << b
        adj[b] |= 1 << a
    assert max_clique(range(5), adj) == 3
    # Maior peso = 1, para contagens exatas e ruidosas.
    ytr = np.array([0] * 10 + [1] * 30)
    w = pesos_exatos(ytr, 2)
    assert w.max() == 1 and abs(w[1] - 1 / 3) < 1e-12, w
    assert pesos_laplace(ytr, 2, 1.0, rng).max() == 1
    # Probabilidades somam 1 com pesos e ε/MCS.
    _, p = exponencial(np.array([[3, 7], [0, 0], [100, 2]]) * w, 0.99 * 1.0 / 72, rng)
    assert np.allclose(p.sum(1), 1)
    # Variante A reproduz a 1ª linha (ε = 0,1, r = 0,5) do acuracias.csv atual.
    X, y, _, classes, _, _ = carregar()
    rng = np.random.default_rng(SEMENTE)
    Xtr, ytr, Xte, yte, _ = dividir(X, y, rng)
    cont = vizinhanca(Xtr, ytr, Xte, len(classes), raios=RAIOS)
    for j in range(len(RAIOS)):   # mesma ordem de sorteios do rn_privado
        rn_normal(cont[:, j], rng)
    accs, *_ = sorteios(cont[:, 0], yte, eps_consulta(EPSILONS[0], len(Xte)), rng)
    with open(rp.SAIDA / "acuracias.csv", encoding="utf-8") as f:
        ref = next(l for l in csv.DictReader(f) if l["modelo"] == "rn_privado" and float(l["r"]) == RAIOS[0]
                   and float(l["epsilon"]) == EPSILONS[0])
    assert f"{np.mean(accs):.4f}" == ref["acuracia_media"], (np.mean(accs), ref)
    print("teste ok")


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEMENTE)
    X, y, linha_csv, classes, _, _ = carregar()
    Xtr, ytr, Xte, yte, pos_teste = dividir(X, y, rng)
    linhas_teste = linha_csv[pos_teste]
    nc = len(classes)
    SAIDA.mkdir(parents=True, exist_ok=True)
    cont = vizinhanca(Xtr, ytr, Xte, nc)
    n_tr = np.bincount(ytr, minlength=nc)
    w_ex = pesos_exatos(ytr, nc)
    print(f"treino por classe {classes}: {n_tr.tolist()}  peso exato de {classes[0]}: {w_ex[0]:.6f}")

    mcs = {r: mcs_do_teste(Xte, r) for r in RAIOS}
    print("MCS máximo por r:", ", ".join(f"{r}: {m.max()}" for r, (m, _) in mcs.items()))
    print("componentes por r:", ", ".join(f"{r}: {c}" for r, (_, c) in mcs.items()))

    # A usa `rng` na mesma ordem do rn_privado (r-N sem privacidade primeiro), para sair idêntica ao acuracias.csv.
    # B, C, D e a referência com pesos usam um gerador à parte, para não deslocar a sequência de A.
    ref_rn = [(rn_normal(cont[:, j], rng) == yte).mean() for j in range(len(RAIOS))]
    rng2 = np.random.default_rng(SEMENTE)
    ref_pesos = [(rn_normal(cont[:, j] * w_ex, rng2) == yte).mean() for j in range(len(RAIOS))]
    linhas = [["variante", "r", "epsilon", "epsilon_roleta", "mcs_max", "componentes", "acuracia_media", "desvio_padrao",
               "acuracia_1o_sorteio", "acuracia_esperada"]]
    for j, r in enumerate(RAIOS):
        linhas.append(["rn", r, "", "", "", "", f"{ref_rn[j]:.4f}", "", "", ""])
        linhas.append(["rn_pesos", r, "", "", "", "", f"{ref_pesos[j]:.4f}", "", "", ""])
        print(f"sem privacidade, r = {r}: r-N {ref_rn[j]:.4f}  com pesos exatos {ref_pesos[j]:.4f}")

    medias = {v: np.empty((len(EPSILONS), len(RAIOS))) for v in "ABCD"}
    for i, eps in enumerate(EPSILONS):
        for j, r in enumerate(RAIOS):
            m, comps = mcs[r]
            # variante: (ε da roleta, usa pesos?)
            config = {"A": (eps_consulta(eps, len(Xte)), False), "B": (eps / m, False),
                      "C": (0.99 * eps / len(Xte), True), "D": (0.99 * eps / m, True)}
            texto = []
            for v, (e_roleta, com_pesos) in config.items():
                g = rng if v == "A" else rng2
                pesos = (lambda: pesos_laplace(ytr, nc, eps, g)) if com_pesos else (lambda: 1.0)
                accs, pred, prob, w1 = sorteios(cont[:, j], yte, e_roleta, g, pesos)
                medias[v][i, j] = np.mean(accs)
                esperada = f"{prob[np.arange(len(yte)), yte].mean():.4f}" if not com_pesos else ""
                linhas.append([v, r, eps, (f"{e_roleta:g}" if np.ndim(e_roleta) == 0 else ("eps/MCS" if v == "B" else "0.99*eps/MCS")),
                               m.max() if v in "BD" else "", comps if v in "BD" else "", f"{np.mean(accs):.4f}",
                               f"{np.std(accs):.4f}", f"{accs[0]:.4f}", esperada])
                if v != "A":
                    gravar_com_peso(f"predicoes_{VARIANTES[v]}_eps_{eps}_r_{r}.csv", linhas_teste, yte, pred,
                                    classes, cont[:, j], prob, w1[0], m if v in "BD" else None)
                texto.append(f"{v} {np.mean(accs):.4f}" + (f" (w≤50K {w1[0]:.4f})" if com_pesos else ""))
            print(f"ε = {eps:>5}, r = {r:>3}: " + " | ".join(texto))

    with open(SAIDA / "acuracias.csv", "w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows(linhas)
    grafico(medias, ref_rn, ref_pesos)
    print(f"arquivos em {SAIDA}  ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    if "--teste" not in sys.argv and not rp.BASE.exists():
        print(f"Não achei a base de dados: {rp.BASE}\nColoque o adult.csv na pasta 'dados', ao lado do programa.")
        if rp.EXE:
            input("\nAperte Enter para fechar.")
        sys.exit(1)
    teste() if "--teste" in sys.argv else main()
    if rp.EXE:
        input("\nPronto. Aperte Enter para fechar.")
