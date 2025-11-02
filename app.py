import os
import re
import datetime
import pandas as pd
import yfinance as yf
import streamlit as st
from prophet import Prophet
from google import genai
from google.genai import types

from dotenv import load_dotenv

load_dotenv()

import os
import openai


openai.api_key = os.getenv("GROQ_API_KEY")
openai.api_base = "https://api.groq.com/openai/v1"



# =============================
# Configurações do Streamlit
# =============================
st.set_page_config(page_title="Histórico de Ações B3", layout="centered")
st.title("📊 Histórico de Ações da B3")
#st.markdown("#### O app lê automaticamente o arquivo local COTAHIST.")
st.markdown("#### Coleta o histórico atualizado do ticker via Yahoo Finance.")

modelo = "llama-3.3-70b-versatile"

inicio = "2025-01-01"
hoje = datetime.date.today().strftime("%Y-%m-%d")

st.markdown("### Periodo")
st.markdown("#### Inicio: "+ inicio)
st.markdown("#### Final: " + hoje)

# =============================
# Função para localizar arquivo
# =============================

pasta = "txt"

def localizar_arquivo_cotahist(pasta):
    """
    Localiza o arquivo mais recente COTAHIST_*.TXT dentro da pasta ./txt
    """
    if not os.path.exists(pasta):
        os.makedirs(pasta)
        return None

    arquivos = [arq for arq in os.listdir(pasta) if arq.upper().startswith("COTAHIST_") and arq.lower().endswith(".txt")]
    if not arquivos:
        return None

    # Pega o mais recente (pelo ano)
    arquivos.sort(reverse=True)
    st.write("Arquivo lido:", arquivos[0])
    return os.path.join(pasta, arquivos[0])

# =============================
# Função para extrair tickers
# =============================
@st.cache_data
def extrair_tickers_b3(caminho_txt):
    tickers = set()
    with open(caminho_txt, "r", encoding="latin1") as f:
        for linha in f:
            if linha.startswith("01"):
                # pega 12 caracteres da posição correta (12 a 23)
                ticker = linha[12:24].strip()
                #st.write("Ticker:", ticker)
                # valida: pelo menos 4 letras + pelo menos 1 número
                if re.match(r"^[A-Z]{2,5}\d{1,2}[A-Z]?$", ticker):
                    tickers.add(ticker)
                    #st.write("Tickers:", tickers)
    return sorted(tickers)

# --- Função de previsão com Prophet ---
def predict_stock(df_input):
    try:
        hist = df_input[['ds','y']].dropna()
        if hist.empty:
            return None, None, None
        hist['ds'] = pd.to_datetime(hist['ds'])
        m = Prophet(daily_seasonality=True)
        m.fit(hist)
        futuro = m.make_future_dataframe(periods=180)
        forecast = m.predict(futuro)
        return forecast,m,hist
    except Exception as e:
        st.error(f"Erro ao prever dados: {e}")
        return None, None, None
        


# --- Função de plotagem ---
def plot_predictions(ticker, forecast, model, hist):
    if forecast is None or model is None:
        st.warning("Nenhum dado de previsão disponível.")
        return
    st.markdown(f"#### Previsão de Preço para {ticker} (Próximos 6 meses)")
    fig1 = model.plot(forecast)
    st.pyplot(fig1)
    fig2 = model.plot_components(forecast)
    st.pyplot(fig2)       


def create_llm_forecast_agent(forecast_df, ticker):
    """
    Cria um agente Groq para interpretar apenas a previsão futura do Prophet (180 dias após a data atual).
    """
    if forecast_df.empty:
        return "Não há dados de previsão para interpretar."

    hoje = pd.Timestamp.today().normalize()
    previsao_futura = forecast_df[forecast_df['ds'] > hoje].copy()

    if previsao_futura.empty:
        return "Nenhuma previsão futura encontrada (verifique o forecast)."

    first_day_forecast = previsao_futura.iloc[0]
    last_day_forecast = previsao_futura.iloc[-1]
    max_yhat_row = previsao_futura.loc[previsao_futura['yhat'].idxmax()]
    min_yhat_row = previsao_futura.loc[previsao_futura['yhat'].idxmin()]

    yhat_max_date_str = max_yhat_row['ds'].strftime('%d/%m/%Y')
    yhat_min_date_str = min_yhat_row['ds'].strftime('%d/%m/%Y')

    trend_direction = "Estável"
    if last_day_forecast['yhat'] > first_day_forecast['yhat'] * 1.02:
        trend_direction = "Crescimento acentuado"
    elif last_day_forecast['yhat'] < first_day_forecast['yhat'] * 0.98:
        trend_direction = "Queda acentuada"

    previsao_futura['interval_width'] = (
        previsao_futura['yhat_upper'] - previsao_futura['yhat_lower']
    ).abs()
    avg_interval_width = previsao_futura['interval_width'].mean()
    max_interval_width = previsao_futura['interval_width'].max()

    # --- Construir prompt apenas com o período futuro ---
    prompt = f"""
    Você é um analista financeiro especializado em ações da B3.
    Analise as previsões futuras da ação {ticker} geradas pelo modelo Prophet.

    Período da previsão: {first_day_forecast['ds'].strftime('%d/%m/%Y')} a {last_day_forecast['ds'].strftime('%d/%m/%Y')}
    Tendência geral: {trend_direction}
    Máximo previsto: R$ {max_yhat_row['yhat']:.2f} em {yhat_max_date_str}
    Mínimo previsto: R$ {min_yhat_row['yhat']:.2f} em {yhat_min_date_str}
    Intervalo médio de confiança: R$ {avg_interval_width:.2f}
    Intervalo máximo de confiança: R$ {max_interval_width:.2f}

    Gere um relatório em português, com:
    1- Titulo do relatório: Análise da Ação - incluir o ticker.
    2. Uma tabela em Markdown com os valores acima. Colunas Categoria e Valor.
    3. Uma análise textual em 2-3 parágrafos explicando a tendência, possíveis riscos e incertezas.
    """

    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1"
        )

        response = client.chat.completions.create(
            model=f'{modelo}',
            messages=[
                {"role": "system", "content": "Você é um analista financeiro técnico e objetivo."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=700
        )

        result = response.choices[0].message.content
        st.markdown("### 📈 Interpretação da Previsão (Groq LLM)")
        st.markdown(f"### Modelo: {modelo}")
        st.markdown(result)
        st.warning("Disclaimer: Interpretação gerada por IA, não é aconselhamento financeiro.")
        
        result = response.choices[0].message.content

        # Gerar e oferecer download
        relatorio = gerar_relatorio_analise(ticker, modelo, result)
        #st.write('Data:', hoje.strftime('%d/%m/%Y'))

        st.download_button(
            label="📥 Baixar Relatório Completo",
            data=relatorio,
            file_name=f"analise_{ticker}_{hoje.strftime('%d/%m/%Y')}.md",
            mime="text/markdown",
            help="O relatório é gerado sob demanda e não fica armazenado no servidor"
        )


    except Exception as e:
        st.error(f"Erro ao gerar interpretação: {e}")


 
def gerar_relatorio_analise(ticker, modelo, resultado):
    """Gera conteúdo do relatório sem salvar em disco"""
    
    conteudo = f"""# 📊 Análise de Previsão - {ticker}
    
    data = f'{hoje.strftime('%d/%m/%Y')}'

    **Data:** {data}
    **Modelo:** {modelo}
    **Ticker:** {ticker}

    ## 📈 Interpretação da Previsão

    {resultado}

    ---
    *Relatório gerado automaticamente - Para fins educacionais*
    *Arquivo não é armazenado no servidor*
    """
    return conteudo 
    
    
# =============================
# Localiza e processa arquivo
# =============================
arquivo_txt = localizar_arquivo_cotahist(pasta)

if arquivo_txt is None:
    st.error("⚠️ Nenhum arquivo encontrado em `./txt`. Coloque o arquivo COTAHIST_AAAAA.TXT nessa pasta.")
    st.stop()
# Tickers extraidos do arquivo txt
#st.info(f"Usando o arquivo: **{pasta}\{os.path.basename(arquivo_txt)}**")

lista_tickers = extrair_tickers_b3(arquivo_txt)

if not lista_tickers:
    st.error("Não foi possível extrair tickers válidos do arquivo.")
    st.stop()

# =============================
# Interface principal
# =============================
ticker_escolhido = st.selectbox("Selecione o ticker:", lista_tickers)

if st.button("ANALISE"):

    if ticker_escolhido:
        st.info(f"Buscando histórico do ticker via Yahoo Finance...")

        inicio = "2025-01-01"
        hoje = datetime.date.today().strftime("%Y-%m-%d")
        dados = yf.download(f"{ticker_escolhido}.SA", start=inicio, end=hoje)

        if not dados.empty:
            # Se houver MultiIndex, "flatten" para usar só os nomes principais
            if isinstance(dados.columns, pd.MultiIndex):
                dados.columns = [col[0] for col in dados.columns]

            # Seleciona apenas colunas de interesse
            colunas_principais = ["Open", "High", "Low", "Close", "Volume"]
            dados_proph = dados[colunas_principais].copy()
            st.markdown("### 📅 Últimas cotações:")
            st.dataframe(dados_proph[colunas_principais].tail())
            
            tamanho = dados_proph.shape[0]
            
            if tamanho > 10:

                # Transforma o index (Date) em coluna 'ds'
                st.markdown("### Dataset Prophet")
                dados_proph.reset_index(inplace=True)
                dados_proph.rename(columns={"Date": "ds", 'Close': 'y'}, inplace=True)
                dados_proph['ticker'] = ticker_escolhido
                dados_proph = dados_proph[['ticker', 'ds','y']]
                inicio = dados_proph.iloc[0]
                st.write("Inicio:", inicio['ds'].strftime("%d-%m-%Y"))
                fim = dados_proph.iloc[-1]
                st.write("Fim:", fim['ds'].strftime("%d-%m-%Y"))
                st.dataframe(dados_proph.tail())
                
                forecast_df, model, hist = predict_stock(dados_proph)
                plot_predictions(ticker_escolhido, forecast_df, model, hist)
                
                # Previsao interpretada pelo agente
                create_llm_forecast_agent(forecast_df, ticker_escolhido)
            else:
                st.markdown("### Ticker com poucos dados.")
                st.error("Dados insuficientes")            
          
        else:
            st.error("⚠️ Não foi possível obter dados do Yahoo Finance para esse ticker.")
