# Trabalho 4 - Privacidade Diferencial: r-N com mecanismo Exponencial

Privacidade de Dados - 2026. Dupla: Isaack Nyllo e Antonio dos Santos Leite Filho.

Compara o r-N tradicional com um r-N diferencialmente privado no Adult Income Dataset.
O r-N privado segue o Algoritmo 2 de Gursoy et al. (2017): em vez de escolher a classe com a maior contagem, ele sorteia a classe pelo mecanismo Exponencial.

## Como rodar

Requisitos: Python 3.10+, `numpy` e `matplotlib`.

```
pip install numpy matplotlib
python rn_privado.py            # r-N do enunciado: lê dados/adult.csv e grava em resultados/
python rn_melhorias.py          # melhorias: grava em resultados/melhorias/
python rn_privado.py --teste    # checagem rápida (idem para rn_melhorias.py --teste)
```

**Sem Python (só na entrega em zip, não estão neste repositório):** dê dois cliques em `rn_privado.exe` (o r-N do enunciado) ou em `rn_melhorias.exe` (as melhorias). Cada um grava em `resultados/` (as melhorias em `resultados/melhorias/`), na mesma pasta. Para só a checagem rápida, rode `rn_privado.exe --teste` ou `rn_melhorias.exe --teste` no terminal. Os dois precisam da pasta `dados/` ao lado. Foram gerados com PyInstaller a partir de `rn_privado.py` e `rn_melhorias.py`.

Na apresentação: itens obrigatórios nos slides 6, 7, 9, 10, 11, 12; demonstração no slide 14; melhorias nos slides 16 a 20.

## O que o programa faz

1. **Pré-processamento.** Remove `fnlwgt`, `education`, `capital-gain`, `capital-loss` e `hours-per-week`. Remove as linhas com `?`. Sobram 45222 tuplas.
2. **Codificação.** Cada coluna de texto vira One-Hot (uma coluna 0/1 por categoria, em ordem alfabética). `age` e `educational-num` ficam entre 0 e 1 pelo MinMax: `(valor - mínimo) / (máximo - mínimo)`.
   - O mínimo e o máximo vêm da base inteira, antes da divisão. O enunciado descreve o pré-processamento antes da divisão. Com a semente 2026, o treino tem os mesmos limites da base, então as distâncias saem iguais.
3. **Alvo.** `income`, com 2 classes: `<=50K` e `>50K`.
4. **Divisão.** Teste com 1000 tuplas, 500 de cada classe, sorteadas com semente 2026. As outras 44222 são o treino.
5. **r-N tradicional** (Definição 9 do artigo). Para cada tupla de teste, conta quantos pontos de treino de cada classe estão a distância euclidiana ≤ r. Prevê a classe com a maior contagem. Implementado sem bibliotecas de aprendizado; o numpy só faz as contas de matriz.
   - Empate de contagem (inclusive 0 a 0, quando ninguém está no raio): sorteia entre as classes empatadas, como o artigo permite. Escolher sempre `<=50K` daria vantagem de graça a uma classe no teste 50/50.
6. **r-N privado** (Algoritmo 2). Para cada tupla de teste, usa as mesmas contagens e sorteia a classe pelo mecanismo Exponencial. Roda para cada ε ∈ {0,1; 0,5; 1; 5; 10; 100} e cada r ∈ {0,5; 1; 1,5; 2; 6}.

## Mecanismo Exponencial

- **Utilidade.** `q(x, I)` = quantos pontos de treino da classe I estão a distância ≤ r da tupla de teste x.
- **Sensibilidade global Δq = 1.** Pôr ou tirar uma tupla do treino muda a contagem de uma classe em no máximo 1.
- **Probabilidade.** `Pr[classe I] ∝ exp(ε′ · q(I) / (2 · Δq))`, dividida pela soma disso sobre as classes. O 2 cobre a mudança do numerador e a do denominador.
- **Saída.** Sempre uma classe válida, nunca um número com ruído. Mesmo com raio vazio (0 a 0), sorteia 50/50.
- **Estabilidade numérica.** Antes do `exp`, o programa subtrai o maior expoente. As probabilidades não mudam, e `exp` não estoura com contagens grandes (ex.: 44222).

## Orçamento

Cada ε é o orçamento **total** do conjunto de teste. Cada tupla de teste é uma consulta à mesma base de treino. Pela composição sequencial, os gastos se somam:

`ε′ = ε / |Xtest| = ε / 1000`, e 1000 · ε′ = ε.

**Por que NÃO dividir por L.** O Exponencial faz **uma** seleção por tupla de teste: olha as contagens de todas as classes de uma vez e sorteia uma classe. É um uso só do mecanismo, com ε′ inteiro. O enunciado pede para não dividir por L. No Trabalho 3, o Laplace fazia L contagens por tupla, e cada uma usava `ε / (L · |Xtest|)`.

| | Trabalho 3 (Laplace) | Trabalho 4 (Exponencial) |
|---|---|---|
| Usos por tupla de teste | L contagens com ruído | 1 seleção de classe |
| ε de cada uso | `ε / (L · \|Xtest\|)` | `ε′ = ε / \|Xtest\|` |
| Tamanho do teste | 13567 | 1000 |
| O que sai | contagens com ruído, maior vence | classe sorteada |

## Resultados (semente 2026)

Acurácia no teste de 1000 tuplas (500 de cada classe). O sorteio da classe é aleatório, então cada (ε, r) foi rodado 20 vezes. A tabela e o gráfico mostram média ± desvio padrão (em pontos percentuais). Os arquivos de predição guardam só o primeiro sorteio: os 20 sorteios servem para a média e o desvio do gráfico, e publicar 20 respostas independentes gastaria 20ε.

| r | sem privacidade | ε = 0,1 | ε = 0,5 | ε = 1 | ε = 5 | ε = 10 | ε = 100 |
|---|---|---|---|---|---|---|---|
| 0,5 | 71,6% | 49,9 ± 1,0 | 50,9 ± 1,4 | 51,4 ± 1,3 | 55,0 ± 1,3 | 57,6 ± 1,8 | 66,8 ± 0,9 |
| 1 | 71,0% | 50,2 ± 1,2 | 50,8 ± 1,6 | 51,2 ± 1,3 | 54,6 ± 1,1 | 57,6 ± 1,5 | 66,7 ± 0,9 |
| 1,5 | 63,7% | 50,4 ± 1,2 | 52,0 ± 1,3 | 54,7 ± 1,3 | 60,1 ± 0,9 | 61,0 ± 1,1 | 62,3 ± 0,4 |
| 2 | 59,7% | 50,6 ± 1,4 | 52,0 ± 1,4 | 53,5 ± 1,3 | 56,5 ± 1,1 | 56,8 ± 0,6 | 58,8 ± 0,4 |
| 6 | 50,0% | 50,2 ± 1,3 | 50,0 ± 0,1 | 50,0 ± 0,0 | 50,0 ± 0,0 | 50,0 ± 0,0 | 50,0 ± 0,0 |

Vizinhos no raio (`resultados/vizinhos.txt`):

| r | mediana de vizinhos | tuplas de teste sem vizinho |
|---|---|---|
| 0,5 | 245 | 56 de 1000 |
| 1 | 261 | 50 de 1000 |
| 1,5 | 2677 | 2 de 1000 |
| 2 | 2857 | 0 de 1000 |
| 6 | 44222 (todos) | 0 de 1000 |

**Leitura.**
- **r = 6 dá 50%.** A distância máxima entre duas pessoas, depois do One-Hot e do MinMax, é 3,94. Com r = 6, toda tupla de teste vê os 44222 pontos de treino, com as mesmas contagens. A previsão é sempre a mesma classe (`<=50K`), e acerta 50% no teste 50/50. Não é erro do código.
- **r = 0,5 e r = 1 saem quase iguais.** Mudar uma categoria no One-Hot custa √2 ≈ 1,41 de distância. Até r = 1, só entra quem tem todas as categorias iguais; o raio só muda a tolerância em idade e escolaridade. Com r = 1,5, já entra quem difere em uma categoria.
- **O r-N sem privacidade fica entre 50% e 72%.** O treino é desbalanceado (cerca de 75% `<=50K`) e o teste é 50/50. As contagens no raio puxam para `<=50K`. Quanto maior o raio, maior o puxão.
- **ε maior aproxima o privado do sem privacidade.** Com ε′ = ε/1000, ε pequeno deixa a escolha quase cara ou coroa (perto de 50%). Em ε = 100 (ε′ = 0,1), r = 0,5 e r = 1 chegam a 66,8% e 66,7%, contra 71,6% e 71,0% sem privacidade.
- **Raio vazio.** Nas tuplas sem vizinho as contagens são 0 e 0, e o sorteio é 50/50.

## Limites (para a arguição)

- **Pontos sem vizinho.** Com r = 0,5 são 56 de 1000 tuplas de teste; a predição sai só do sorteio.
- **Teste 50/50 contra treino desbalanceado.** Por isso o r-N sem privacidade acerta menos que o esperado.
- **MinMax na base inteira.** Os limites de idade e escolaridade são o domínio público dos atributos e não revelam ninguém. Com a semente usada, calcular só no treino daria o mesmo resultado.
- **Números de análise.** Vizinhos por raio, acurácia do r-N sem privacidade e a acurácia esperada do CSV são calculados direto da base. Servem só para entender o resultado; não fazem parte da saída privada.
- **Os 20 sorteios.** São análise do experimento, não 20 respostas publicadas.
- **A privacidade protege o treino.** Ela limita o que as respostas do classificador revelam sobre cada pessoa do treino. Não anonimiza a base.

## Melhorias (extra, além do enunciado)

Código em `rn_melhorias.py` (importa o `rn_privado.py`, que não foi alterado). A variante A é a do enunciado. As outras mexem só no orçamento ou nas contagens. Resultados em `resultados/melhorias/`, mesma semente 2026 e 20 sorteios por (ε, r).

- **A. Enunciado.** ε′ = ε/1000 para cada tupla de teste.
- **B. Algoritmo 3.** Faz um grafo com os testes: aresta se a distância entre dois testes é ≤ 2r. Cada teste usa ε′ = ε / (clique máxima da sua componente). Por que é seguro: uma pessoa do treino só entra no raio de testes que estão a ≤ 2r entre si, ou seja, numa clique da mesma componente. Esses testes gastam no máximo (clique máxima) · ε/(clique máxima) = ε, e componentes diferentes não compartilham pessoas (composição paralela).
- **C. Pesos.** Contagens das classes do treino com ruído de Laplace, usando 1% do ε (peso ≤50K = 10.708/33.514 ≈ 0,32). O maior peso é 1, então Δq continua 1. A roleta usa os outros 99% do ε (0,99 · ε/1000).
- **D. B + C.** Algoritmo 3 e pesos juntos (roleta com 99% do ε).
- **E.** r-N sem privacidade (referência, não é privado).
- **F.** r-N sem privacidade, com pesos exatos (referência, não é privado).

Acurácia média (%), lida de `resultados/melhorias/acuracias.csv`:

| ε | r | A | B | C | D | E | F |
|---|---|---|---|---|---|---|---|
| 1 | 0,5 | 51,4 | 69,9 | 51,3 | 74,0 | 71,6 | 78,4 |
| 1 | 1,5 | 54,7 | 55,4 | 57,6 | 60,3 | 63,7 | 78,3 |
| 1 | 2 | 53,5 | 53,6 | 58,1 | 58,4 | 59,7 | 78,5 |
| 10 | 0,5 | 57,6 | 71,4 | 59,4 | 78,1 | 71,6 | 78,4 |
| 10 | 1,5 | 61,0 | 61,1 | 71,1 | 72,1 | 63,7 | 78,3 |
| 10 | 2 | 56,8 | 56,8 | 70,9 | 71,1 | 59,7 | 78,5 |
| 100 | 0,5 | 66,8 | 71,4 | 70,4 | 78,4 | 71,6 | 78,4 |
| 100 | 1,5 | 62,3 | 62,7 | 76,1 | 76,5 | 63,7 | 78,3 |
| 100 | 2 | 58,8 | 58,8 | 76,1 | 76,2 | 59,7 | 78,5 |

Mais detalhes em `resultados/melhorias/acuracias.csv` e `resultados/melhorias/melhorias.png`.

## Arquivos

| Arquivo | Conteúdo |
|---|---|
| `rn_privado.py` | código-fonte do r-N do enunciado |
| `rn_melhorias.py` | código-fonte das melhorias (A a D); importa o `rn_privado.py` |
| `rn_privado.exe` | executável do r-N do enunciado, para Windows (não precisa de Python instalado) |
| `rn_melhorias.exe` | executável das melhorias, para Windows |
| `apresentacao.pptx` | slides da apresentação |
| `dados/adult.csv` | a base (Adult Income Dataset, do Kaggle indicado no enunciado) |
| `resultados/predicoes_rn_r_<r>.csv` | predições do r-N sem privacidade, um arquivo por r |
| `resultados/predicoes_rn_privado_eps_<ε>_r_<r>.csv` | predições, contagens e probabilidades do r-N privado (1º sorteio), um arquivo por ε e r |
| `resultados/acuracias.csv` | acurácia média, desvio, 1º sorteio e acurácia esperada, por ε e r |
| `resultados/vizinhos.txt` | vizinhos por raio (mínimo, mediana, máximo e tuplas sem vizinho) |
| `resultados/melhorias/` | predições das variantes B, C e D, `acuracias.csv` e `melhorias.png` |
| `resultados/acuracia.png` | gráfico da acurácia por ε, uma cor por raio, com o r-N sem privacidade tracejado |

Nos CSVs de predição, `linha_no_adult_csv` é a linha da pessoa no arquivo original (o cabeçalho é a linha 1).

## Referência

Gursoy, M. E., Inan, A., Nergiz, M. E., Saygin, Y. Differentially private nearest neighbor classification. *Data Mining and Knowledge Discovery* 31 (2017): 1544-1575.
