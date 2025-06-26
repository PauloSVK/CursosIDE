import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity
from llama_index.embeddings.ollama import OllamaEmbedding
import json


# --- CONFIGURAÇÃO ---
# URLs ou caminhos para os dados pré-calculados
CURSOS_EMBEDDINGS_URL = r'vectors_nome_do_curso_nome.json'
DISCIPLINAS_EMBEDDINGS_URL = r'vectors_nome_desc.json'


# --- FUNÇÕES DE CARREGAMENTO ---
@st.cache_data
def load_data():
    with open(CURSOS_EMBEDDINGS_URL) as f:
        vectors_cursos = json.load(f)
    embeddings_cursos = np.array(list(vectors_cursos.values()))
    with open(DISCIPLINAS_EMBEDDINGS_URL) as f:
        vectors_disciplinas = json.load(f)
    embeddings_disciplinas = np.array(list(vectors_disciplinas.values()))
    df_cursos = pd.read_excel(r'Cursos e disciplinas IDE completo.xlsx', sheet_name="Cursos")
    df_disc = pd.read_excel(r'Cursos e disciplinas IDE completo.xlsx', sheet_name="Disciplinas")
    meta_cursos_umap = pd.read_excel(r'meta_cursos_umap.xlsx')
    meta_disciplinas_tsne = pd.read_excel(r'meta_disciplinas_tsne.xlsx')
    return df_cursos, df_disc, embeddings_cursos, embeddings_disciplinas, meta_cursos_umap, meta_disciplinas_tsne

# --- CARREGA RECURSOS ---
@st.cache_resource
def load_model():
    return OllamaEmbedding("paraphrase-multilingual:latest", base_url='http://10.61.49.233:11434/')

# --- UI PRINCIPAL ---
st.title("Visualização de Cursos e Disciplinas")


# --- SIDEBAR ---
def render_sidebar():
    with st.sidebar:
        st.title("Cursos IDE")

        st.header("Gráfico")
        selected_chart = st.selectbox(
            "Selecione o gráfico a ser renderizado",
            options=['Cursos','Disciplinas']
        )

        texto_busca = st.text_input("Digite sua descrição ou interesse:")
    return selected_chart, texto_busca


# --- CALCULAR SIMILARIDADE ---
def compute_similarity(texto_busca, model, embeddings):
    user_emb = model.get_text_embedding(texto_busca)
    distances = cosine_similarity([user_emb],embeddings)[0]
    return distances

# --- PLOTAR GRÁFICO ---
def plot_chart(meta_df, id_col):
    fig = px.scatter(
        meta_df,
        x="X",
        y="Y",
        color= 'distance',
        hover_name=id_col,
        title='Teste',
        labels={'X':'Componente 1', 'Y':'Componente 2'},
        color_continuous_scale=px.colors.sequential.Bluered
    )
    fig.update_traces(marker=dict(size=6, opacity=0.8))
    x_margin = meta_df["X"].max() * 0.1
    y_margin = meta_df["Y"].max() * 0.1
    fig.update_xaxes(range=[meta_df["X"].min() - x_margin, meta_df["X"].max() + x_margin])
    fig.update_yaxes(range=[meta_df["Y"].min() - y_margin, meta_df["Y"].max() + y_margin])
    st.plotly_chart(fig, use_container_width=True)

# --- PLOTAR TABELA
def plot_dataframe(df, meta_df, selected_chart, key):
#    df_results = meta_df[[key,'Descrição','distance']].merge(
#        df, on=[[key,'Descrição']], how='left'
#    )
#    if selected_chart=='Cursos':
#        df_results = df_results[['Nome do curso', 'Programa','Área de Conhecimento','distance']]
#    else:
#        df_results = df_results[['Nome','Descrição','Autor','distance']]
#    df_results.sort_values('distance',ascending=True)
    meta_df.sort_values('distance',ascending=False,inplace=True)
    meta_df = meta_df[[key,'Descrição']]
    st.subheader("Disciplinas mais próximas do tema de interesse")
    st.dataframe(meta_df, height = 400)

    


def main():
    df_cursos, df_disc, embeddings_cursos, embeddings_disciplinas, meta_cursos_umap, meta_disciplinas_tsne = load_data()
    selected_chart, texto_busca = render_sidebar()
    model = load_model()

    if selected_chart == 'Cursos':
        meta_df = meta_cursos_umap
        vecs = embeddings_cursos
        df = df_cursos
        id_col = 'Nome do curso'
    else:
        meta_df = meta_disciplinas_tsne
        vecs = embeddings_disciplinas
        df = df_disc
        id_col = 'Nome'
    
    distances = None
    if texto_busca:
        distances = compute_similarity(texto_busca, model, vecs)
    meta_df['distance'] = distances
   

    plot_chart(meta_df, id_col)
    plot_dataframe(df, meta_df, selected_chart, id_col)

if __name__ == "__main__":
    main()
