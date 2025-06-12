FROM python:3.10-slim

# Instalar a biblioteca libgomp1 necessária para LightGBM e outras bibliotecas de ML
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*


WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Criar o diretório .streamlit e copiar um secrets.toml vazio para satisfazer o Streamlit
#RUN mkdir -p /app/.streamlit
#COPY ./.streamlit/secrets.toml /app/.streamlit/secrets.toml

COPY . .

CMD ["streamlit", "run", "app.py"]
EXPOSE 8501