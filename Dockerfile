# Utilize uma imagem base com Python 3.10.12
FROM python:3.10.12-slim
RUN pip install --upgrade pip

# Defina o diretório de trabalho
WORKDIR /app

# Copie o requirements.txt para a imagem
COPY requirements.txt .

# Instale as dependências do Python
RUN pip install --no-cache-dir -r requirements.txt

# Copie o seu código Python para a imagem
COPY ./src .

# Argumentos de build
ARG CLIENT_ID
ARG CLIENT_SECRET
ARG REDIRECT_URI
ARG NEO4J_URL
ARG NEO4J_USER
ARG NEO4J_PASSWORD

# Define as variáveis de ambiente para as credenciais do Google Auth
ENV CLIENT_ID=$CLIENT_ID
ENV CLIENT_SECRET=$CLIENT_SECRET
ENV REDIRECT_URI=$REDIRECT_URI

# Define as variáveis de ambiente para as credenciais do Neo4j
ENV NEO4J_URL=$NEO4J_URL
ENV NEO4J_USER=$NEO4J_USER
ENV NEO4J_PASSWORD=$NEO4J_PASSWORD

# Exponha a porta para a aplicação Streamlit
EXPOSE 8501

# Comando para iniciar a aplicação Streamlit
CMD ["streamlit", "run", "main.py"]
