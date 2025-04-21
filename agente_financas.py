import streamlit as st
from langgraph.graph import Graph
from langchain_core.messages import HumanMessage, AIMessage
from typing import Dict, TypedDict, Optional
from openai import OpenAI
import yfinance as yf
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta

# Configuração do cliente Mistral 7B
def get_mistral_client(api_key: str) -> OpenAI:
    """Configura o cliente da API Mistral via OpenRouter"""
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

# Define o estado do agente
class AgentState(TypedDict):
    messages: list
    acao1: str
    acao2: str
    api_key: str
    dados_tecnicos: dict
    periodo: str

# Função para validar símbolos de ações
def validar_simbolo(ticker: str) -> bool:
    """Verifica se um símbolo de ação é válido no Yahoo Finance"""
    try:
        acao = yf.Ticker(ticker)
        # Tenta obter informações básicas
        info = acao.info
        return info is not None and len(info) > 0
    except Exception:
        return False

# Função para obter dados técnicos das ações com tratamento robusto de erros
def get_technical_data(ticker: str, period: str = '1y') -> Optional[dict]:
    """
    Obtém dados técnicos e fundamentais de uma ação com tratamento completo de erros
    
    Args:
        ticker: Símbolo da ação (ex: PETR4.SA)
        period: Período dos dados históricos (1mo, 3mo, 6mo, 1y, 2y, 5y)
    
    Returns:
        Dicionário com dados históricos, fundamentais e informações ou None em caso de erro
    """
    try:
        acao = yf.Ticker(ticker)
        
        # Obtém dados históricos com fallback para períodos menores se necessário
        try:
            hist = acao.history(period=period)
            if hist.empty:
                # Tenta um período menor se o solicitado não retornar dados
                fallback_period = '3mo' if period in ['1y', '2y', '5y'] else '1mo'
                hist = acao.history(period=fallback_period)
                if hist.empty:
                    print(f"Dados históricos vazios para {ticker} mesmo com fallback")
                    return None
        except Exception as e:
            print(f"Erro ao obter histórico para {ticker}: {e}")
            return None
        
        # Obtém informações fundamentais com tratamento de erro
        try:
            info = acao.info
            if not info:
                print(f"Nenhuma informação fundamental disponível para {ticker}")
                return None
        except Exception as e:
            print(f"Erro ao obter informações fundamentais para {ticker}: {e}")
            return None
        
        # Cálculo de indicadores técnicos com verificação de dados suficientes
        try:
            if len(hist) >= 50:
                hist['SMA_50'] = hist['Close'].rolling(window=50).mean()
            if len(hist) >= 200:
                hist['SMA_200'] = hist['Close'].rolling(window=200).mean()
            if len(hist) >= 14:
                hist['RSI'] = compute_rsi(hist['Close'])
        except Exception as e:
            print(f"Erro ao calcular indicadores técnicos para {ticker}: {e}")
        
        # Informações fundamentais com fallback para 'N/A' quando não disponível
        fundamentals = {
            'currentPrice': info.get('currentPrice', 'N/A'),
            'targetMeanPrice': info.get('targetMeanPrice', 'N/A'),
            'recommendationMean': info.get('recommendationMean', 'N/A'),
            'dividendYield': info.get('dividendYield', 'N/A'),
            'peRatio': info.get('trailingPE', 'N/A'),
            'beta': info.get('beta', 'N/A'),
            'marketCap': info.get('marketCap', 'N/A'),
            'volume': hist['Volume'].mean() if not hist['Volume'].empty else 'N/A'
        }
        
        return {
            'historical': hist,
            'fundamentals': fundamentals,
            'info': info
        }
    except Exception as e:
        print(f"Erro geral ao obter dados para {ticker}: {e}")
        return None

# Cálculo do RSI com tratamento de erro
def compute_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """Calcula o Relative Strength Index (RSI)"""
    try:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    except Exception as e:
        print(f"Erro ao calcular RSI: {e}")
        return pd.Series(index=prices.index, name='RSI')

# Nó de pesquisa com yfinance com validação e tratamento robusto
def pesquisar_acoes(state: AgentState) -> AgentState:
    """Pesquisa dados de ações com validação e tratamento completo de erros"""
    state.setdefault("messages", [])
    state.setdefault("dados_tecnicos", {})
    state.setdefault("periodo", "1y")
    
    # Validação de entrada
    if not state["acao1"] or not state["acao2"]:
        state["messages"].append(AIMessage(content="Erro: Símbolos das ações não fornecidos"))
        return state
    
    # Validação dos símbolos antes de pesquisar
    if not validar_simbolo(state["acao1"]):
        state["messages"].append(AIMessage(
            content=f"Erro: Símbolo {state['acao1']} inválido ou não encontrado. "
                   f"Verifique se o símbolo está correto (ex: PETR4.SA para Petrobras)."
        ))
        return state
    
    if not validar_simbolo(state["acao2"]):
        state["messages"].append(AIMessage(
            content=f"Erro: Símbolo {state['acao2']} inválido ou não encontrado. "
                   f"Verifique se o símbolo está correto (ex: VALE3.SA para Vale)."
        ))
        return state
    
    try:
        # Obter dados para ambas as ações
        dados_acao1 = get_technical_data(state["acao1"], state["periodo"])
        dados_acao2 = get_technical_data(state["acao2"], state["periodo"])
        
        # Verificar se os dados foram obtidos com sucesso
        if dados_acao1 is None:
            state["messages"].append(AIMessage(
                content=f"Erro: Não foi possível obter dados completos para {state['acao1']}. "
                       f"Alguns dados podem não estar disponíveis para esta ação."
            ))
            return state
            
        if dados_acao2 is None:
            state["messages"].append(AIMessage(
                content=f"Erro: Não foi possível obter dados completos para {state['acao2']}. "
                       f"Alguns dados podem não estar disponíveis para esta ação."
            ))
            return state
        
        # Armazenar dados técnicos
        state["dados_tecnicos"] = {
            state["acao1"]: dados_acao1,
            state["acao2"]: dados_acao2
        }
        
        # Preparar resumo dos dados com verificação de valores
        def format_value(value):
            if isinstance(value, float):
                return f"{value:.2f}" if not pd.isna(value) else "N/A"
            return str(value) if value is not None else "N/A"
        
        resumo = f"""
        ## Dados Técnicos Coletados (Período: {state['periodo']})
        
        **{state["acao1"]}**
        - Preço Atual: {format_value(dados_acao1['fundamentals']['currentPrice'])}
        - Média de Preço Alvo: {format_value(dados_acao1['fundamentals']['targetMeanPrice'])}
        - Recomendação Média: {format_value(dados_acao1['fundamentals']['recommendationMean'])}
        - Dividend Yield: {format_value(dados_acao1['fundamentals']['dividendYield'])}
        - P/E Ratio: {format_value(dados_acao1['fundamentals']['peRatio'])}
        - Volume Médio: {format_value(dados_acao1['fundamentals']['volume'])}
        
        **{state["acao2"]}**
        - Preço Atual: {format_value(dados_acao2['fundamentals']['currentPrice'])}
        - Média de Preço Alvo: {format_value(dados_acao2['fundamentals']['targetMeanPrice'])}
        - Recomendação Média: {format_value(dados_acao2['fundamentals']['recommendationMean'])}
        - Dividend Yield: {format_value(dados_acao2['fundamentals']['dividendYield'])}
        - P/E Ratio: {format_value(dados_acao2['fundamentals']['peRatio'])}
        - Volume Médio: {format_value(dados_acao2['fundamentals']['volume'])}
        """
        
        state["messages"].append(AIMessage(content=resumo))
        
    except Exception as e:
        state["messages"].append(AIMessage(
            content=f"Erro inesperado na pesquisa: {str(e)}. "
                   "Por favor, tente novamente com outros símbolos ou mais tarde."
        ))
    
    return state

# Nó de geração de relatório com tratamento de erro melhorado
def gerar_relatorio(state: AgentState) -> AgentState:
    """Gera relatório comparativo com tratamento robusto de erros"""
    state.setdefault("messages", [])
    
    if not state.get("api_key"):
        state["messages"].append(AIMessage(content="Erro: Chave da API não configurada"))
        return state
    
    if not state.get("dados_tecnicos"):
        state["messages"].append(AIMessage(content="Erro: Dados técnicos não disponíveis para gerar relatório"))
        return state
    
    try:
        client = get_mistral_client(state["api_key"])
        
        # Preparar prompt detalhado com fallback para dados ausentes
        dados_disponiveis = state["messages"][-1].content if state["messages"] else "Dados limitados disponíveis"
        
        prompt = f"""
        Você é um analista financeiro especializado. Analise comparativamente as ações {state["acao1"]} e {state["acao2"]} com base nos seguintes dados:
        
        {dados_disponiveis}
        
        Gere um relatório completo com:
        1. Análise Técnica Comparativa (incluindo tendências)
        2. Análise Fundamentalista Detalhada
        3. Avaliação de Risco
        4. Recomendações de Investimento
        5. Perspectivas Futuras
        
        Destaque:
        - Vantagens comparativas de cada ação
        - Riscos específicos
        - Cenários favoráveis e desfavoráveis
        
        Use formatação markdown com tabelas comparativas quando apropriado.
        Se algum dado estiver marcado como 'N/A', explique que essa informação não estava disponível.
        Responda em português brasileiro com terminologia financeira apropriada.
        """
        
        response = client.chat.completions.create(
            model="mistralai/mistral-7b-instruct:free",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=2000
        )
        
        relatorio = response.choices[0].message.content
        state["messages"].append(AIMessage(content=relatorio))
        
    except Exception as e:
        state["messages"].append(AIMessage(
            content=f"Erro na geração do relatório: {str(e)}. "
                   "Por favor, verifique sua chave de API e conexão com a internet."
        ))
    
    return state

# Criar visualizações gráficas com tratamento robusto
def criar_graficos(dados_tecnicos: dict) -> list:
    """Cria gráficos Plotly a partir dos dados técnicos"""
    graficos = []
    
    for ticker, dados in dados_tecnicos.items():
        if dados and 'historical' in dados and not dados['historical'].empty:
            df = dados['historical'].reset_index()
            
            try:
                # Gráfico de preços com médias móveis (se disponíveis)
                fig = px.line(df, x='Date', y=['Close'], 
                             title=f'Preço - {ticker}',
                             labels={'value': 'Preço', 'variable': 'Indicador'})
                
                # Adiciona médias móveis se calculadas
                if 'SMA_50' in df.columns:
                    fig.add_scatter(x=df['Date'], y=df['SMA_50'], name='SMA 50')
                if 'SMA_200' in df.columns:
                    fig.add_scatter(x=df['Date'], y=df['SMA_200'], name='SMA 200')
                
                graficos.append(fig)
                
                # Gráfico de RSI se calculado
                if 'RSI' in df.columns:
                    fig_rsi = px.line(df, x='Date', y=['RSI'], 
                                    title=f'RSI - {ticker}',
                                    labels={'value': 'RSI'})
                    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Sobrevendido")
                    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Sobrecomprado")
                    graficos.append(fig_rsi)
                    
            except Exception as e:
                print(f"Erro ao criar gráficos para {ticker}: {e}")
    
    return graficos if graficos else None

# Interface Streamlit melhorada
def main():
    """Interface principal do aplicativo"""
    st.set_page_config(page_title="Analisador de Ações", layout="wide")
    
    st.title("📈 Analisador Avançado de Ações")
    st.caption("""
    Compare ações com análise técnica e fundamentalista completa. 
    Dados fornecidos pelo Yahoo Finance. Relatórios gerados por IA.
    """)
    
    # Sidebar com configurações
    with st.sidebar:
        st.header("Configurações")
        api_key = st.text_input("Chave da API OpenRouter", type="password", 
                              help="Obtenha uma chave em https://openrouter.ai/")
        
        st.markdown("### Ajuda")
        st.info("""
        - Use símbolos completos (ex: PETR4.SA, VALE3.SA)
        - Ações brasileiras terminam com .SA
        - Para ETFs, use o símbolo completo (ex: BOVA11.SA)
        """)
        
        st.markdown("### Exemplos Válidos")
        st.code("""
        PETR4.SA - Petrobras
        VALE3.SA - Vale
        ITUB4.SA - Itaú
        BOVA11.SA - ETF Ibovespa
        AAPL - Apple (EUA)
        """)
    
    # Área principal
    col1, col2 = st.columns(2)
    with col1:
        acao1 = st.text_input("Primeira ação (ex: PETR4.SA)", "PETR4.SA",
                            help="Insira o símbolo completo da ação")
    with col2:
        acao2 = st.text_input("Segunda ação (ex: VALE3.SA)", "VALE3.SA",
                            help="Insira o símbolo completo da ação")
    
    periodo = st.selectbox("Período de análise", ['1mo', '3mo', '6mo', '1y', '2y', '5y'], 
                          index=3, help="Período histórico para análise")
    
    if st.button("🔍 Analisar Ações", type="primary"):
        if not api_key:
            st.error("Por favor, insira sua chave da API OpenRouter para gerar relatórios")
            st.stop()
            
        with st.spinner("Coletando dados e gerando análise..."):
            try:
                # Criar e executar workflow
                workflow = Graph()
                workflow.add_node("pesquisar", pesquisar_acoes)
                workflow.add_node("gerar_relatorio", gerar_relatorio)
                workflow.set_entry_point("pesquisar")
                workflow.add_edge("pesquisar", "gerar_relatorio")
                workflow.set_finish_point("gerar_relatorio")
                compiled_workflow = workflow.compile()
                
                estado_inicial = AgentState(
                    messages=[],
                    acao1=acao1,
                    acao2=acao2,
                    api_key=api_key,
                    dados_tecnicos={},
                    periodo=periodo
                )
                
                resultado = compiled_workflow.invoke(estado_inicial)
                
                # Exibir resultados
                if resultado["messages"]:
                    for msg in resultado["messages"]:
                        if isinstance(msg, AIMessage):
                            st.markdown(msg.content)
                
                # Adicionar visualizações gráficas se disponíveis
                if resultado.get("dados_tecnicos"):
                    graficos = criar_graficos(resultado["dados_tecnicos"])
                    if graficos:
                        st.subheader("📊 Visualizações Técnicas")
                        for grafico in graficos:
                            st.plotly_chart(grafico, use_container_width=True)
                    else:
                        st.warning("Não foi possível gerar gráficos para estas ações")
                        
            except Exception as e:
                st.error(f"""
                Erro durante a análise: {str(e)}
                
                Por favor:
                1. Verifique os símbolos das ações
                2. Confira sua conexão com a internet
                3. Tente novamente mais tarde
                """)
                st.stop()

if __name__ == "__main__":
    main()