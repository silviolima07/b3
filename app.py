import os
import streamlit as st
import pandas as pd
from PIL import Image
import zipfile
from prophet import Prophet
import matplotlib.pyplot as plt
from google import genai
from google.genai import types
from dotenv import load_dotenv

# --- Configurações ---
st.set_page_config(page_title="Previsões de Ações B3", layout="wide")
load_dotenv()
API_KEY = os.getenv('GEMINI_API_KEY')
client = genai.Client(api_key=API_KEY)

# --- Arquivos ---
txt_file_name = "COTAHIST_A2025.TXT"
zip_file_name = "COTAHIST_A2025.zip"
extract_to_path = './'

# --- Upload / Extração do TXT ---
if 'unzipped' not in st.session_state:
    st.session_state.unzipped = False

if not st.session_state.unzipped:
    os.makedirs(extract_to_path, exist_ok=True)
    try:
        if not os.path.exists(zip_file_name):
            st.error(f"Arquivo ZIP '{zip_file_name}' não encontrado.")
            st.stop()

        with zipfile.ZipFile(zip_file_name, 'r') as zip_ref:
            if txt_file_name in zip_ref.namelist():
                zip_ref.extract(txt_file_name, extract_to_path)
                st.session_state.unzipped = True
            else:
                st.error(f"Arquivo '{txt_file_name}' não encontrado no ZIP.")
                st.stop()
    except Exception as e:
        st.error(f"Erro ao descompactar ZIP: {e}")
        st.stop()

# --- Carregar e pré-processar dados da B3 ---
@st.cache_data
def load_and_preprocess_b3_data(file_path):
    try:
        from b3fileparser.b3parser import B3Parser
        parser = B3Parser.create_parser(engine='polars')
        dados_b3 = parser.read_b3_file(file_path)
        b3 = dados_b3.to_pandas()

        b3_stock_df = b3.loc[
            (b3.CODIGO_BDI == 'LOTE_PADRAO') & (b3.TIPO_DE_MERCADO == 'VISTA'),
            ['DATA_DO_PREGAO','PRECO_ULTIMO_NEGOCIO','CODIGO_DE_NEGOCIACAO','NOME_DA_EMPRESA']
        ].copy()
        b3_stock_df.rename(columns={
            'DATA_DO_PREGAO':'ds',
            'PRECO_ULTIMO_NEGOCIO':'y',
            'CODIGO_DE_NEGOCIACAO':'ticker',
            'NOME_DA_EMPRESA':'empresa'
        }, inplace=True)

        b3_stock_df['ds'] = pd.to_datetime(b3_stock_df['ds'], format='%Y%m%d', errors='coerce')
        b3_stock_df['y'] = pd.to_numeric(b3_stock_df['y'], errors='coerce')
        b3_stock_df['ticker'] = b3_stock_df['ticker'].str.strip()
        b3_stock_df.dropna(subset=['ds','y'], inplace=True)

        unique_tickers = sorted(list(b3_stock_df.ticker.unique()))
        return b3_stock_df, unique_tickers

    except Exception as e:
        st.error(f"Erro ao processar dados da B3: {e}")
        return pd.DataFrame(), []

b3_stock, lista_stocks_unique = load_and_preprocess_b3_data(txt_file_name)
if b3_stock.empty:
    st.error("Não foi possível carregar dados históricos da B3.")
    st.stop()

# --- Função de previsão com Prophet ---
def predict_stock(df_input):
    try:
        hist = df_input[['ds','y']].dropna()
        if hist.empty:
            return None, None, None
        hist['ds'] = pd.to_datetime(hist['ds'])
        m = Prophet(daily_seasonality=True)
        m.fit(hist)
        future = m.make_future_dataframe(periods=6, freq='M')
        forecast = m.predict(future)
        return forecast, m, hist
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

# --- Função de interpretação IA usando Gemini Developer ---
def create_llm_forecast_agent(forecast_df, ticker):
    if forecast_df.empty:
        return "Não há dados de previsão para interpretar."

    first_day_forecast = forecast_df.iloc[0]
    last_day_forecast = forecast_df.iloc[-1]

    max_yhat_row = forecast_df.loc[forecast_df['yhat'].idxmax()]
    min_yhat_row = forecast_df.loc[forecast_df['yhat'].idxmin()]

    yhat_max_date_str = max_yhat_row['ds'].strftime('%d/%m/%Y') if isinstance(max_yhat_row['ds'], pd.Timestamp) else 'N/A'
    yhat_min_date_str = min_yhat_row['ds'].strftime('%d/%m/%Y') if isinstance(min_yhat_row['ds'], pd.Timestamp) else 'N/A'

    trend_direction = "Estável"
    if last_day_forecast['yhat'] > first_day_forecast['yhat'] * 1.02:
        trend_direction = "Crescimento acentuado"
    elif last_day_forecast['yhat'] < first_day_forecast['yhat'] * 0.98:
        trend_direction = "Queda acentuada"

    forecast_df['interval_width'] = (forecast_df['yhat_upper'].fillna(0) - forecast_df['yhat_lower'].fillna(0)).abs()
    avg_interval_width = forecast_df['interval_width'].mean()
    max_interval_width = forecast_df['interval_width'].max()

    prompt = f"""
    Analise os dados de previsão da ação {ticker} para os próximos 6 meses.
    Forneça sumário em tabela Markdown e interpretação em 2-3 parágrafos em Português.
    """

    try:
        #response = client.models.generate_content(
        #    model="gemini-1.5-flash-8b",
        #    contents=[prompt],
        #    config=types.GenerateContentConfig(
        #        temperature=0.3,
        #        max_output_tokens=600
        #    )
        #)
        # Chame o modelo Gemini 1.5 Flash-8B
        response = client.chat.completions.create(
            model="gemini-1.5-flash-8b",
            messages=[
                {"role": "system", "content": "Você é um assistente financeiro."},
               {"role": "user", "content": prompt}
    ]
)

        
        return response.choices[0].message.content
    except Exception as e:
        return f"Erro ao gerar interpretação: {str(e)}"

# --- Streamlit UI ---
st.sidebar.title("Menu")
choice = st.sidebar.radio("Opções", ["Previsões", "Sobre"])

if choice == "Previsões":
    selected_stock = st.sidebar.selectbox("Selecione uma ação", lista_stocks_unique)
    b3_periodo = b3_stock[b3_stock.ticker == selected_stock].sort_values(by='ds')

    if st.sidebar.button("Processar"):
        forecast, model, hist = predict_stock(b3_periodo)
        plot_predictions(selected_stock, forecast, model, hist)

        st.markdown("### Interpretação da IA")
        interpretation = create_llm_forecast_agent(forecast, selected_stock)
        st.write(interpretation)
        st.warning("Disclaimer: Interpretação gerada por IA, não é aconselhamento financeiro.")

elif choice == "Sobre":
    st.markdown("#### Dados históricos coletados em b3.com.br")
    st.markdown("#### Modelo Prophet + Gemini Developer API")
