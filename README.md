# 📈 B3 App - Consulta de Dados da Bolsa com IA

Este é um app interativo construído com [Streamlit](https://streamlit.io/) para consultar dados financeiros da B3 e gerar análises com apoio de inteligência artificial via Gemini API (Google AI).

A série histórica esta disponível no site da b3:

https://www.b3.com.br/pt_br/market-data-e-indices/servicos-de-dados/market-data/historico/mercado-a-vista/series-historicas/

---

## 🚀 Funcionalidades

- Consulta de ativos da bolsa brasileira (B3)
- Análise automática com IA (Google Gemini)
- Interface simples e interativa via navegador (Streamlit)

---

## 🛠️ Tecnologias

- Python 3.10+
- Streamlit
- Google Gemini API
- Docker (opcional)
- `.env` ou `secrets.toml` para chaves de API

---

## ⚙️ Como rodar localmente

### 1. Clone o repositório

git clone https://github.com/seu-usuario/b3-app.git
cd b3-app

### 2. Crie o arquivo .env com sua chave da API Gemini
GEMINI_API_KEY=sua-chave-aqui ! Não seu aspas ao redor da chave

### 3. Instale as dependências
pip install -r requirements.txt

### 4. Execute o app
streamlit run app.py

🐳 Rodando com Docker
### 1. Crie o arquivo .env com a chave da API (mesmo formato acima)
### 2. Construa a imagem Docker
docker build -t b3-app .
### 3. Execute o container
docker run -d -p 8501:8501 --name b3-app --env-file .env b3-app
### 4. Acesse em: 
http://localhost:8501

☁️ Implantação no Streamlit Cloud
Suba o projeto no GitHub

Acesse Streamlit Cloud

Crie um novo app apontando para seu repositório

Vá em Settings > Secrets e adicione:
GEMINI_API_KEY = "sua-chave-aqui"



