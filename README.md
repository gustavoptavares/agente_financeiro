# 📈 Agente de Investimentos Avançado com Streamlit, LangGraph e Mistral 7B

## 🔍 Visão Geral

Este projeto é uma aplicação interativa construída com **Streamlit**, que utiliza **LangGraph**, **OpenAI API (Mistral 7B via OpenRouter)** e a biblioteca **yfinance** para realizar uma **análise técnica e fundamentalista comparativa de ações da bolsa de valores**.

A ferramenta permite ao usuário inserir dois símbolos de ações, coletar dados históricos e fundamentais de cada uma, gerar um relatório automatizado com análise detalhada e exibir gráficos interativos de indicadores como médias móveis e RSI.

---

## ❗ Problema e Solução

### O Problema
Investidores e analistas frequentemente precisam comparar ações com base em diversos indicadores técnicos e fundamentais. Esse processo pode ser:
- Manual e demorado;
- Sujeito a vieses interpretativos;
- Dependente de múltiplas ferramentas e fontes de dados.

### A Solução
Este projeto oferece uma **solução automatizada, interativa e assistida por IA**, que:
- Coleta dados técnicos e fundamentais automaticamente com `yfinance`;
- Apresenta visualizações interativas com `plotly`;
- Utiliza um modelo de linguagem (Mistral 7B) para gerar relatórios detalhados e personalizados;
- Permite a qualquer investidor tomar decisões mais informadas em menos tempo.

---

## ⚙️ Processo

### Tecnologias Utilizadas
- **Python**
- **Streamlit** – Interface de usuário interativa
- **LangGraph** – Definição do fluxo de execução (grafo de agentes)
- **yfinance** – Coleta de dados de ações
- **Plotly Express** – Visualizações interativas (médias móveis, RSI)
- **OpenAI API (via OpenRouter)** – Geração de relatório com IA (modelo Mistral 7B)

### Etapas do Processo

1. **Entrada do Usuário**:
   - Chave da API OpenRouter;
   - Símbolos das ações a serem analisadas (ex: `PETR4.SA` e `VALE3.SA`);
   - Período de análise (ex: `1y`, `6mo`, etc).

2. **Pesquisa de Ações** (`pesquisar_acoes`):
   - Coleta dados históricos;
   - Calcula indicadores como **SMA 50**, **SMA 200** e **RSI**;
   - Extrai dados fundamentais (preço-alvo, dividend yield, P/E, beta etc);
   - Gera resumo técnico em formato markdown.

3. **Geração de Relatório com IA** (`gerar_relatorio`):
   - Envia os dados coletados como *prompt* para o modelo Mistral 7B;
   - Recebe de volta uma análise detalhada contendo:
     - Comparação técnica;
     - Avaliação fundamentalista;
     - Avaliação de risco;
     - Recomendações de investimento;
     - Perspectivas futuras.

4. **Visualização de Gráficos**:
   - Gráfico de preços com médias móveis;
   - Gráfico do RSI com faixas indicativas de sobrecompra/sobrevenda.

---

## 📊 Resultados

Ao rodar a aplicação, o usuário obtém:

- ✅ Um **relatório em linguagem natural**, com insights reais sobre ambas as ações comparadas;
- 📈 Gráficos interativos que facilitam a leitura visual da performance;
- 🔎 Dados fundamentais importantes para decisões de médio e longo prazo;
- 🤖 Uma experiência híbrida entre dados quantitativos e interpretação qualitativa via IA.

---

## ✅ Conclusões

Este projeto demonstra como é possível integrar múltiplas bibliotecas e APIs para construir um **agente financeiro inteligente**, capaz de automatizar análises e relatórios complexos. Ele é útil para:

- Investidores iniciantes e avançados;
- Analistas financeiros;
- Educadores e estudantes de finanças;
- Desenvolvedores interessados em IA e Finanças Quantitativas.
---

## 🚀 Como Executar

**Instalação dos pacotes necessários**
```bash
pip install streamlit openai langgraph langchain-core yfinance plotly
```

**Execução do app Streamlit**
```bash
streamlit run nome_do_arquivo.py
```

**Tela do Deploy**

![imagem](https://github.com/gustavoptavares/agente_financeiro/blob/main/Deploy.jpg)
