import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt #pyplot já importado
import warnings
from datetime import date
from prophet import Prophet
import pytz
import os
import yaml
import zipfile # Importar zipfile
import google.generativeai as genai

st.set_page_config(page_title="Previsões de Ações B3", layout="wide")

# --- Variável para o nome do arquivo TXT e ZIP ---
txt_file_name = "COTAHIST_A2025.TXT"
zip_file_name = "COTAHIST_A2025.zip"
extract_to_path = './' # Diretório onde o TXT será descompactado (mesmo diretório do script)

# --- DESCOMPACTAÇÃO DO ARQUIVO ZIP ---
# Esta parte precisa ser executada ANTES de carregar os dados B3
# Garante que o arquivo TXT esteja disponível.
# A descompactação deve acontecer apenas uma vez por sessão do Streamlit
# para evitar desnecessário reprocessamento e erros.
if 'unzipped' not in st.session_state:
    st.session_state.unzipped = False

if not st.session_state.unzipped:
    # Crie o diretório de destino se não existir
    os.makedirs(extract_to_path, exist_ok=True)

    try:
        if not os.path.exists(zip_file_name):
            st.error(f"Erro: Arquivo ZIP '{zip_file_name}' não encontrado no diretório do script.")
            st.stop()

        with zipfile.ZipFile(zip_file_name, 'r') as zip_ref:
            # Verifica se o arquivo TXT está dentro do ZIP
            if txt_file_name in zip_ref.namelist():
                zip_ref.extract(txt_file_name, extract_to_path)
                #st.success(f"Arquivo '{txt_file_name}' descompactado com sucesso para '{extract_to_path}'")
                st.session_state.unzipped = True # Marca como descompactado
            else:
                st.error(f"Erro: Arquivo '{txt_file_name}' não encontrado dentro de '{zip_file_name}'.")
                st.stop()

    except zipfile.BadZipFile:
        st.error(f"Erro: '{zip_file_name}' não é um arquivo ZIP válido. Verifique se o arquivo não está corrompido.")
        st.stop()
    except Exception as e:
        st.error(f"Ocorreu um erro ao descompactar o arquivo ZIP: {e}")
        st.stop()

# --- CONFIGURAÇÃO DA API GEMINI ---
api_key = None

# 1. Tentar ler a chave de API dos segredos do Streamlit Cloud
try:
    if "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]
        #st.success("Chave GEMINI_API_KEY carregada dos segredos do Streamlit Cloud.")
except Exception as e:
        st.error(f"Chaves GEMINI_API_KEY nao encontrada: {str(e)}")
        st.stop()

def create_llm_forecast_agent(forecast_df, ticker):
    """
    Cria um agente de IA (LLM) para interpretar o forecast de ações,
    apresentando os dados chave em uma tabela e uma interpretação separada.
    """
    if forecast_df.empty:
        return "Não há dados de previsão para interpretar."

    first_day_forecast = forecast_df.iloc[0]
    last_day_forecast = forecast_df.iloc[-1]

    max_yhat_row = forecast_df.loc[forecast_df['yhat'].idxmax()]
    min_yhat_row = forecast_df.loc[forecast_df['yhat'].idxmin()]

    # Garantir que as datas são objetos de data válidos antes de formatar
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

    # --- Construir o prompt para o LLM ---
    prompt = f"""
    Você é um analista financeiro experiente, com foco em ações do mercado brasileiro.
    Sua tarefa é analisar os dados de previsão de preço para a ação {ticker} nos próximos 6 meses e fornecer uma interpretação concisa e profissional.

    Primeiro, apresente os dados chave da previsão em uma tabela formatada em Markdown. Em seguida, forneça uma interpretação dos dados em 2-3 parágrafos.

    **Dados da Previsão para {ticker} (para a tabela e interpretação):**
    - Período da Previsão: {first_day_forecast['ds'].strftime('%d/%m/%Y')} a {last_day_forecast['ds'].strftime('%d/%m/%Y')}
    - Preço Previsto no Início: {first_day_forecast['yhat']}
    - Intervalo de Confiança no Início: {first_day_forecast['yhat_lower']} a {first_day_forecast['yhat_upper']}
    - Preço Previsto no Fim: {last_day_forecast['yhat']}
    - Intervalo de Confiança no Fim: {last_day_forecast['yhat_lower']} a {last_day_forecast['yhat_upper']}
    - Preço Máximo Previsto: {max_yhat_row['yhat']} (Data: {yhat_max_date_str})
    - Preço Mínimo Previsto: {min_yhat_row['yhat']} (Data: {yhat_min_date_str})
    - Tendência Geral Prevista: {trend_direction}
    - Amplitude Média do Intervalo de Confiança: {avg_interval_width}
    - Amplitude Máxima do Intervalo de Confiança: {max_interval_width}

    **Formato da Resposta:**
    Sua resposta deve seguir estritamente o formato abaixo:

    ### Sumário da Previsão
    | Métrica | Valor | Data (se aplicável) |
    |---|---|---|
    | Período da Previsão | {first_day_forecast['ds'].strftime('%d/%m/%Y')} a {last_day_forecast['ds'].strftime('%d/%m/%Y')} | |
    | Preço Previsto no Início | R$ {first_day_forecast['yhat']:.2f} | {first_day_forecast['ds'].strftime('%d/%m/%Y')} |
    | Intervalo de Confiança (Início) | R$ {first_day_forecast['yhat_lower']:.2f} a R$ {first_day_forecast['yhat_upper']:.2f} | |
    | Preço Previsto no Fim | R$ {last_day_forecast['yhat']:.2f} | {last_day_forecast['ds'].strftime('%d/%m/%Y')} |
    | Intervalo de Confiança (Fim) | R$ {last_day_forecast['yhat_lower']:.2f} a R$ {last_day_forecast['yhat_upper']:.2f} | |
    | Preço Máximo Previsto | R$ {max_yhat_row['yhat']:.2f} | {yhat_max_date_str} |
    | Preço Mínimo Previsto | R$ {min_yhat_row['yhat']:.2f} | {yhat_min_date_str} |
    | Tendência Geral | {trend_direction} | |
    | Amplitude Média do Intervalo | R$ {avg_interval_width:.2f} | |
    | Amplitude Máxima do Intervalo | R$ {max_interval_width:.2f} | |

    ### Análise da Previsão
    [2-3 parágrafos de interpretação profissional, seguindo estas regras, **escrito em Português do Brasil fluente e natural**:]
    1.  Comece com uma frase clara sobre a tendência geral esperada para o preço da ação {ticker}, referenciando o período da previsão.
    2.  Comente sobre a evolução do preço previsto (se é um crescimento ou queda significativa, ou estabilidade), mencionando o comportamento geral, como a diferença entre o preço inicial e final, e a presença de pontos de mínimo/máximo. **Não repita os valores numéricos exatos que já estão na tabela.**
    3.  Analise a incerteza da previsão, comentando sobre a amplitude média e máxima do intervalo de confiança e o que isso implica para a volatilidade da projeção. Classifique a incerteza como baixa, moderada ou alta.
    4.  Se houver valores negativos na previsão (especialmente para o preço mínimo ou final), comente sobre o que isso pode indicar sobre a aplicabilidade ou limitações do modelo para essa ação/período, e a necessidade de cautela.
    5.  Conclua com uma ressalva sobre a natureza das previsões de mercado e a importância de fatores externos.
    6.  Não dê conselhos financeiros diretos. Apenas interprete os dados fornecidos.
    7.  Use linguagem profissional e objetiva, adequada para um relatório financeiro, sem abreviações ou jargões excessivos.
    """

    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Ocorreu um erro ao gerar a interpretação da IA: {str(e)}"

def plot_stock_data(df):
    empresa = df['empresa'].iloc[0] if 'empresa' in df.columns and not df.empty else "Empresa Desconhecida"
    ticker = df['ticker'].iloc[0] if 'ticker' in df.columns and not df.empty else "Ticker Desconhecido"

    df_diario = df.set_index('ds')[['y']]
    df_semanal = df_diario.resample('W').last()
    st.write(" ")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df_semanal.index, df_semanal['y'], marker='o', linestyle='-')

    for i, txt in enumerate(df_semanal['y']):
        ax.text(df_semanal.index[i], df_semanal['y'].iloc[i], f'{txt:.2f}',
                ha='center', va='bottom',
                fontsize=11, color='red')

    first_date = str(df['ds'].min()).split(' ')[0]
    last_date = str(df['ds'].max()).split(' ')[0]

    st.markdown(f"### Empresa: {empresa}")
    st.markdown("### Período de Dados Históricos")
    st.write(f"{first_date} a {last_date}")

    ax.set_title(f'\n{ticker}\nEvolução Semanal\n', fontsize=14, fontweight='bold')
    ax.set_xlabel('Dia da Semana')
    ax.set_ylabel('Preço de Fechamento (R$)')
    ax.grid(True)

    st.pyplot(fig)


def predict_stock(df_input):
    """Make predictions for a given stock ticker"""
    try:
        hist = df_input[['ds', 'y']].dropna(subset=['y'])
        if hist.empty:
            ticker_name = df_input['ticker'].iloc[0] if 'ticker' in df_input.columns and not df_input.empty else "a ação selecionada"
            st.error(f"Não há dados históricos válidos para {ticker_name} após a remoção de NaNs. Não é possível fazer previsões.")
            return None, None, None

        hist['ds'] = pd.to_datetime(hist['ds'])
        if hist['ds'].dt.tz is not None:
            hist['ds'] = hist['ds'].dt.tz_localize(None)

        m = Prophet(daily_seasonality=True)
        m.fit(hist)

        future = m.make_future_dataframe(periods=6, freq='M')
        forecast = m.predict(future)
        return forecast, m, hist

    except Exception as e:
        ticker_name = df_input['ticker'].iloc[0] if 'ticker' in df_input.columns and not df_input.empty else "a ação selecionada"
        st.error(f"Erro ao processar {ticker_name} para previsão: {str(e)}")
        return None, None, None

def plot_predictions(ticker, forecast, model, hist):
    """Display the forecast results and components"""
    if forecast is None or model is None:
        st.error("Nenhum dado de previsão disponível")
        return

    try:
        st.markdown(f"#### Previsão de Preço para {ticker} (Próximos 6 meses)")
        # Plotar o forecast diretamente com Matplotlib
        fig1 = model.plot(forecast)
        st.pyplot(fig1)

        st.markdown("### Componentes da Previsão: Tendências, Semanal e Diário")
        # Plotar os componentes diretamente com Matplotlib
        fig2 = model.plot_components(forecast)
        st.pyplot(fig2)

    except Exception as e:
        st.error(f"Erro ao gerar gráficos do Prophet: {str(e)}")

# Usa a variável txt_file_name aqui
serie = txt_file_name

# A função load_and_preprocess_b3_data agora só precisa do nome do arquivo txt
@st.cache_data
def load_and_preprocess_b3_data(file_path):
    try:
        from b3fileparser.b3parser import B3Parser # Importe aqui dentro da função se B3Parser for o único lugar que precisa dela
        parser = B3Parser.create_parser(engine='polars')
        dados_b3 = parser.read_b3_file(file_path)
        b3 = dados_b3.to_pandas()

        codigo = 'LOTE_PADRAO'
        preco = 'PRECO_ULTIMO_NEGOCIO'
        nome = "CODIGO_DE_NEGOCIACAO"
        data = "DATA_DO_PREGAO"
        tipo = 'VISTA'
        empresa= 'NOME_DA_EMPRESA'

        b3_stock_df = b3.loc[(b3.CODIGO_BDI == codigo) & (b3.TIPO_DE_MERCADO == tipo)][[data, preco, nome, empresa]].copy()
        b3_stock_df = b3_stock_df.rename(columns={data: 'ds', preco: 'y', nome: 'ticker', empresa :'empresa'})

        b3_stock_df['ds'] = pd.to_datetime(b3_stock_df['ds'], format='%Y%m%d', errors='coerce')

        b3_stock_df['ticker'] = b3_stock_df['ticker'].str.strip()

        b3_stock_df['y'] = pd.to_numeric(b3_stock_df['y'], errors='coerce')

        b3_stock_df.dropna(subset=['ds', 'y'], inplace=True)

        unique_tickers = sorted(list(b3_stock_df.ticker.unique()))

        return b3_stock_df, unique_tickers

    except Exception as e:
        st.error(f"Erro ao carregar e pré-processar os dados da B3: {str(e)}")
        return pd.DataFrame(), []

def main():

    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = False
    if 'current_stock_selection' not in st.session_state:
        st.session_state.current_stock_selection = None
    if 'forecast_result' not in st.session_state:
        st.session_state.forecast_result = None
    if 'model_result' not in st.session_state:
        st.session_state.model_result = None
    if 'hist_result' not in st.session_state:
        st.session_state.hist_result = None
    if 'b3_periodo_for_display' not in st.session_state:
        st.session_state.b3_periodo_for_display = pd.DataFrame()

    logo = Image.open('Logo.png')
    st.sidebar.image(logo)

    st.markdown("""
    <div style="background-color:blue;padding:10px">
        <h1 style='text-align:center;color:white;'>Previsões do Mercado de Ações Brasileiro</h1>
    </div>
    """, unsafe_allow_html=True)
    st.write(" ")
    st.write(" ")
    st.markdown("### 📅 Evolução Semanal desde Janeiro/2025")
    st.markdown("### 📈 Previsão para os Próximos Seis Meses (usando Prophet)")

    activities = ["Previsões", "Sobre"]
    choice = st.sidebar.radio("Menu", activities)

    # Verifica se os dados B3 foram carregados com sucesso
    b3_stock, lista_stocks_unique = load_and_preprocess_b3_data(serie)

    if b3_stock.empty:
        st.error("Não foi possível carregar os dados históricos da B3. Verifique o arquivo COTAHIST_A2025.TXT e a configuração.")
        st.stop()


    if choice == "Previsões":
        st.sidebar.markdown("### Selecione uma Ação")
        selected_stock = st.sidebar.selectbox("Selecione uma ação", lista_stocks_unique, index=0, label_visibility='collapsed', key='stock_selector')

        if selected_stock != st.session_state.current_stock_selection:
            st.session_state.processed_data = False
            st.session_state.current_stock_selection = selected_stock
            st.session_state.forecast_result = None
            st.session_state.model_result = None
            st.session_state.hist_result = None
            st.session_state.b3_periodo_for_display = pd.DataFrame()

        if st.sidebar.button("Processar", key='process_button_main'):
            st.session_state.processed_data = True
            st.session_state.current_stock_selection = selected_stock

            b3_periodo = b3_stock.loc[b3_stock.ticker == selected_stock].copy()
            b3_periodo = b3_periodo.sort_values(by='ds')

            b3_periodo['ds'] = pd.to_datetime(b3_periodo['ds'])
            if b3_periodo['ds'].dt.tz is None:
                b3_periodo['ds'] = (
                        b3_periodo['ds']
                        .dt.tz_localize('America/Sao_Paulo', ambiguous='NaT', nonexistent='NaT')
                        .dt.tz_convert('UTC')
                        .dt.tz_localize(None)
                    )
            else:
                b3_periodo['ds'] = (
                        b3_periodo['ds']
                        .dt.tz_convert('UTC')
                        .dt.tz_localize(None)
                    )
            b3_periodo['y'] = pd.to_numeric(b3_periodo['y'], errors='coerce')
            b3_periodo.dropna(subset=['y'], inplace=True)

            st.session_state.b3_periodo_for_display = b3_periodo

            if not b3_periodo.empty:
                with st.spinner("Processando dados e gerando previsões..."):
                    forecast, model, hist = predict_stock(b3_periodo.copy())
                    st.session_state.forecast_result = forecast
                    st.session_state.model_result = model
                    st.session_state.hist_result = hist
            else:
                st.warning(f"Não há dados históricos válidos para '{selected_stock}'.")
                st.session_state.processed_data = False
                st.session_state.forecast_result = None

        if st.session_state.processed_data and st.session_state.forecast_result is not None:
            plot_stock_data(st.session_state.b3_periodo_for_display)
            plot_predictions(
                st.session_state.current_stock_selection,
                st.session_state.forecast_result,
                st.session_state.model_result,
                st.session_state.hist_result
            )

            st.markdown("---")
            st.markdown("### Interpretação da Previsão por IA")

            with st.spinner("A IA está analisando a previsão..."):
                interpretation = create_llm_forecast_agent(
                        st.session_state.forecast_result,
                        st.session_state.current_stock_selection
                    )
                st.write(interpretation)
                st.warning("Disclaimer: Esta interpretação é gerada por uma IA com base nos dados de previsão e não constitui aconselhamento financeiro.")
        elif st.session_state.processed_data and st.session_state.forecast_result is None:
            st.warning("Não foi possível gerar previsões para a ação selecionada após o processamento.")


    elif choice == "Sobre":
        st.write(" ")
        st.markdown("#### Dados históricos coletados em b3.com.br.")
        st.markdown("#### - Algoritmo: Prophet")
        st.markdown("#### - Modelo: gemini-1.5-flash")
        st.markdown("#### Base de Dados:")
        st.markdown("##### https://www.b3.com.br/pt_br/market-data-e-indices/servicos-de-dados/market-data/historico/mercado-a-vista/series-historicas/")

        #if st.button("Linkedin"):
        st.markdown("[Linkedin: Silvio Lima](https://www.linkedin.com/in/silviocesarlima/)", unsafe_allow_html=True)

if __name__ == '__main__':
    main()
