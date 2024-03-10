from neo4j import GraphDatabase
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.colors as colors
import matplotlib.patches as mpatches
from pyvis.network import Network
from pyvis import network as net
import streamlit as st
import streamlit.components.v1 as components
import community
import pandas as pd
import numpy as np
import altair as alt
import base64
from pathlib import Path
import re
from collections import defaultdict
import scipy.sparse
import scipy.sparse.csgraph
from functools import reduce
from datetime import datetime
import os

NEO4J_URL = os.environ['NEO4J_URL']
NEO4J_USER  = os.environ['NEO4J_USER']
NEO4J_PASSWORD = os.environ['NEO4J_PASSWORD']

def run():
    tipo_ind = None
    email_unicos = None
    all_options_comm = None
    rod = None
    node_indegrees = None

    tab1, tab2, tab3 = st.tabs(
            ["Redes Geral", "Redes Guilda", "Redes Cliente"])  # cria as tabs
        
    with tab1:
        
        col1, col2 = st.columns([1, 3])

        st.markdown("<br>"*2, unsafe_allow_html=True)

        col3, col4 = st.columns([1, 3])

        with col1:
            # cria o driver python
            driver = GraphDatabase.driver(
            	NEO4J_URL, auth=(NEO4J_USER, NEO4J_PASSWORD))

            # função neo4j para importar a rede do localhost neo4j
            def get_eua3(tx, dimensao):
                result = tx.run("""
                    MATCH (c1)-[b1 {dimensao: $dimensao}]->(c2)
                    RETURN c1, b1, b1.rodada, c2
                """,
                                dimensao=dimensao)

                records = list(result)
                summary = result.consume()

                return records, summary

            with driver.session() as session:

                dimension_list = [record["b1_dimensao"] for record in session.run(
                    f"MATCH ()-[b1:BUSCOU]->() RETURN DISTINCT b1.dimensao AS b1_dimensao")]
                dim = st.selectbox("Selecione a dimensão:", sorted(dimension_list), key = 'a')

                records, summary = session.execute_read(
                    get_eua3, dimensao=dim)
                
            G = nx.MultiDiGraph()

            # pega todas as caracteristicas laços e nodos de 'cada vez'
            # adiciona nodos e laços a partir dos resultados do query da rede neo4j
            for record in records:
                source_id = record["c1"].element_id
                target_id = record["c2"].element_id
                G.add_node(source_id, label=record["c1"]["email"])
                G.add_node(target_id, label=record["c2"]["email"])
                G.add_edge(source_id, target_id, label=record["b1"], rodada=record["b1"]["rodada"])

            # rodada = nx.get_edge_attributes(G, "rodada")
            # rodada = {str(key): value for key, value in rodada.items()}

            rodada_values = nx.get_edge_attributes(G, 'rodada')
            graphs_by_rodada = defaultdict(nx.DiGraph)
            for (u, v, k), rodada in rodada_values.items():
                graphs_by_rodada[rodada].add_edge(u, v)

            # Cria a caixa lateral das medidas de centralidade
            with col1:
                centrality = st.selectbox(
                    'Selecione uma medida de centralização:', [
                        'InDegree médio', 'OutDegree médio', 'Betweenness médio', 'Closeness médio', 'Page rank médio',
                        'Densidade'], key='centrality a')
                if centrality != 'Densidade':
                    if centrality == 'InDegree médio':
                        medida = {rodada: nx.in_degree_centrality(
                            graph) for rodada, graph in graphs_by_rodada.items()}
                    elif centrality == 'OutDegree médio':
                        medida = {rodada: nx.out_degree_centrality(
                            graph) for rodada, graph in graphs_by_rodada.items()}
                    elif centrality == 'Betweenness médio':
                        medida = {rodada: nx.betweenness_centrality(
                            graph) for rodada, graph in graphs_by_rodada.items()}

                    # elif centrality == 'Closeness médio':
                    #     medida = {rodada: nx.closeness_centrality(
                    #         graph) for rodada, graph in graphs_by_rodada.items()}

                    elif centrality == 'Closeness médio':

                        import scipy.sparse
                        import scipy.sparse.csgraph

                        medida = {}

                        # computa centralidade closeness para cada graph
                        for rodada, graph in graphs_by_rodada.items():
                            A = nx.adjacency_matrix(graph).tolil()
                            D = scipy.sparse.csgraph.floyd_warshall(
                                A, directed=False, unweighted=False)

                            n = D.shape[0]
                            closeness_centrality = {}
                            for r in range(0, n):
                                cc = 0.0

                                possible_paths = list(enumerate(D[r, :]))
                                shortest_paths = dict(filter(
                                    lambda x: not x[1] == np.inf, possible_paths))

                                total = sum(shortest_paths.values())
                                n_shortest_paths = len(shortest_paths) - 1.0
                                if total > 0.0 and n > 1:
                                    s = n_shortest_paths / (n - 1)
                                    cc = (n_shortest_paths / total) * s
                                closeness_centrality[r] = cc

                            medida[rodada] = closeness_centrality

                    elif centrality == 'Page rank médio':
                        medida = {rodada: nx.pagerank(
                            graph) for rodada, graph in graphs_by_rodada.items()}

                    flat_data = [(category, key, value) for category, inner_dict in medida.items(
                    ) for key, value in inner_dict.items()]
                    df = pd.DataFrame(flat_data, columns=[
                                    'Category', 'Key', 'Value'])
                    df = df.drop(columns=['Key'])

                    # cria data frame
                    df_results = df.groupby('Category')['Value'].agg(
                        ['mean', 'median', 'std'])

                    # ordernar por mes e ano
                    month_to_num = {'JAN': '01', 'FEV': '02', 'MAR': '03', 'ABR': '04', 'MAI': '05', 'JUN': '06', 'JUL': '07', 'AGO': '08', 'SET': '09', 'OUT': '10', 'NOV': '11', 'DEZ': '12'
                                    }
                    df_results['date'] = df_results.index.map(
                        lambda x: month_to_num[x[:3]] + x[3:])
                    df_results['date'] = pd.to_datetime(
                        df_results['date'], format='%m%Y')
                    df_results = df_results.sort_values('date')
                    df_results = df_results.drop(columns='date')

                    df_results = df_results.rename(columns={
                        'mean': 'média',
                        'median': 'mediana',
                        'std': 'desvio padrão'
                    })

                    with col2:

                        # transforma o df em formato long
                        source = df_results.reset_index().melt(
                            'Category', var_name='category', value_name='y')

                        colors = ['#cb0e15', '#0d5cd3', '#1c9937']

                        line_chart = alt.Chart(source).mark_line(point=True).encode(
                            alt.X('Category', title='Rodada', sort=None),
                            alt.Y('y', title=''),
                            color=alt.Color('category', title='Medida')
                        ).properties(
                            title='Séries temporais de grupo',
                            height=450
                        ).configure_range(
                            category=alt.RangeScheme(colors)
                        ).interactive()

                        st.altair_chart(line_chart, use_container_width=True)

                else:
                    densidade = {rodada: nx.density(
                        graph) for rodada, graph in graphs_by_rodada.items()}

                    densidade

                    with col2:
                        df = pd.DataFrame(list(densidade.items()), columns=[
                                        'Category', 'Value'])

                        month_to_num = {'JAN': '01', 'FEV': '02', 'MAR': '03', 'ABR': '04', 'MAI': '05', 'JUN': '06', 'JUL': '07', 'AGO': '08', 'SET': '09', 'OUT': '10', 'NOV': '11', 'DEZ': '12'
                                        }
                        df['date'] = df['Category'].map(
                            lambda x: month_to_num[x[:3]] + x[3:])
                        df['date'] = pd.to_datetime(df['date'], format='%m%Y')

                        df = df.sort_values('date')

                        df = df.drop(columns=['date'])

                        chart = alt.Chart(df).mark_line(point=True).encode(
                            alt.X('Category', title='Rodada', sort=None),
                            alt.Y('Value', title=''),
                        ).properties(
                            title='Séries temporais de grupo',
                            height=450
                        ).interactive()

                        st.altair_chart(chart, use_container_width=True)
    #####################################
    ### Guilda ####################
    with tab2:
        col1, col2 = st.columns([1, 3])

        st.markdown("<br>"*2, unsafe_allow_html=True)

        col3, col4 = st.columns([1, 3])

        with col1:

            # cria o driver python
            driver = GraphDatabase.driver(
            	NEO4J_URL, auth=(NEO4J_USER, NEO4J_PASSWORD))

            # função neo4j para importar a rede do localhost neo4j
            def get_eua3(tx, dimensao, rodada, email):
                result = tx.run("""
                    MATCH (c1)-[b1 {rodada: $rodada, dimensao: $dimensao}]->(c2)
                    WHERE c1.email IN $email OR c2.email IN $email
                    RETURN c1, b1, c2
                """,
                                email=email,
                                dimensao=dimensao,
                                rodada=rodada)

                records = list(result)
                summary = result.consume()

                return records, summary

            # Inicia a sessão do driver
            with driver.session() as session:

                # caixas seleção dimensao e rodada
                with col1:
                    rodada1_list = [record["b1_rodada"] for record in session.run(
                        "MATCH ()-[b1:BUSCOU]->() RETURN DISTINCT b1.rodada AS b1_rodada")]
                    
                    def to_date(s):    
                        months = ["JAN", "FEV", "MAR", "ABR", "MAI", "JUN", "JUL", "AGO", "SET", "OUT", "NOV", "DEZ"]
                        month, year = s[:3], s[3:]
                        return datetime(int(year), months.index(month) + 1, 1)
                    rodada1_list.sort(key=to_date, reverse=True)

                    rod = st.selectbox("Selecione a rodada:", rodada1_list)

                with col1:
                    dimension_list = [record["b1_dimensao"] for record in session.run(
                        f"MATCH ()-[b1:BUSCOU {{rodada: '{rod}'}}]->() RETURN DISTINCT b1.dimensao AS b1_dimensao")]
                    dim = st.selectbox("Selecione a dimensão:",sorted(dimension_list), key = 'b')

                # adiciona os botões
                e_list = [(record["c1_email"], record["c2_email"]) for record in session.run(  # pega todos os nomes em todas as relações existentes para cada dimensao e rodada
                    "MATCH (c1)-[b1 {rodada: $rodada, dimensao: $dimensao}]->(c2) RETURN DISTINCT c1.email AS c1_email, c2.email AS c2_email",
                    dimensao=dim,
                    rodada=rod)]
                email_list = {name for tuple_ in e_list for name in tuple_}
                email_list = list(email_list)

                # executa a query apos construida todas as variaveis de entrada
                records, summary = session.execute_read(
                    get_eua3, dimensao=dim, rodada=rod, email=email_list)

                text_data = []
                for record in records:
                    data = record.data()  # Extrai os dados do registro como um dicionário
                    # Converte o dicionário para uma string e adiciona à lista
                    text_data.append(str(data))

                # Converte a lista em uma única string
                text = " ".join(text_data)
                valores_email = re.findall("'email': '([^']+)", text)

                # Converte a lista para um conjunto para obter valores únicos de email
                email_unicos = set(valores_email)
                # 'email_unicos', email_unicos

                import json
                # converte em lista
                email_unicos = list(email_unicos)
                # formata a lista como array json (mesma sintaxe da lista cypher)
                email_unicos = json.dumps(email_unicos)

                records_ind, summary_ind = session.execute_read(
                    get_eua3, dimensao=dim, rodada=rod, email=email_list)

                text_data_ind = []
                for record in records_ind:
                    data = record.data()  # Extrai os dados do registro como um dicionário
                    # Converte o dicionário para uma string e adiciona à lista
                    text_data_ind.append(str(data))

                # Converte a lista em uma única string
                text_ind = " ".join(text_data_ind)
                valores_email_ind = re.findall("'email': '([^']+)", text_ind)

                # Converte a lista para um conjunto para obter valores únicos de email
                email_unicos_ind = set(valores_email_ind)
                # 'email_unicos', email_unicos

                import json
                # converte em lista
                email_unicos_ind = list(email_unicos_ind)
                # formata a lista como array json (mesma sintaxe da lista cypher)
                email_unicos_ind = json.dumps(email_unicos_ind)

                query_cg_2 = f"MATCH (c:Colaborador)-[r:PERTENCE_{rod}]->(c1:Guilda) RETURN DISTINCT c1.nome AS c1_nome"
                rede_list_guilda = [record["c1_nome"]
                                    for record in session.run(query_cg_2)]

                query = f"MATCH (e)-[r:PERTENCE_{rod}]->(g1) RETURN DISTINCT e.email AS email, g1.nome AS g1_nome"
                guilda = [(record["email"], record['g1_nome'])
                        for record in session.run(query)]

                # cria lista para usar para criar propriedades de guilda
                modified_list_g = []
                for sublist in guilda:
                    modified_dict = {
                        "email": sublist[0],
                        "cliente_guilda": sublist[1]
                    }
                    modified_list_g.append(modified_dict)

                G = nx.DiGraph()

                # pega todas as caracteristicas laços e nodos de 'cada vez'
                # adiciona nodos e laços a partir dos resultados do query da rede neo4j
                for record in records:
                    # 'record', record
                    source_id = record["c1"].element_id
                    target_id = record["c2"].element_id
                    G.add_node(source_id, label=record["c1"]["email"])
                    G.add_node(target_id, label=record["c2"]["email"])
                    G.add_edge(source_id, target_id, label=record["b1"])

                graph_dict = {}
                original_G3 = G.copy()

                for guilda in rede_list_guilda:
                    # Cria um novo grafo para cada guilda
                    G3 = original_G3.copy()
                    for i in range(len(modified_list_g)):
                        if modified_list_g[i]['cliente_guilda'] == guilda:
                            for node in G3.nodes():
                                if G3.nodes[node]['label'] == modified_list_g[i]['email']:
                                    G3.nodes[node]['color'] = '#11d63f'
                                    G3.nodes[node]['guilda'] = guilda

                    nodes_to_remove = [node for node, data in G3.nodes(
                        data=True) if data.get('color') != '#11d63f']
                    G3.remove_nodes_from(nodes_to_remove)
                    graph_dict[guilda] = G3

                centrality_dict = {}
                node_count_dict = {}

                with col1:
                    centrality = st.selectbox(
                        'Selecione uma medida de centralização:', [
                            'Degree', 'Betweenness', 'Closeness', 'Page rank',
                            'Densidade'
                        ])
                    if centrality == 'Degree':
                        for guilda, graph in graph_dict.items():
                            measure_dict = nx.in_degree_centrality(graph)
                            centrality_dict[guilda] = pd.Series(measure_dict).mean()
                            node_count_dict[guilda] = graph.number_of_nodes()
                    elif centrality == 'Betweenness':
                        for guilda, graph in graph_dict.items():
                            measure_dict = nx.betweenness_centrality(graph)
                            centrality_dict[guilda] = pd.Series(measure_dict).mean()
                            node_count_dict[guilda] = graph.number_of_nodes()
                    elif centrality == 'Closeness':
                        for guilda, graph in graph_dict.items():
                            D = nx.to_numpy_array(graph)
                            n = D.shape[0]
                            closeness_centrality = {}
                            for r in range(0, n):
                                cc = 0.0
                                possible_paths = list(enumerate(D[r, :]))

                                shortest_paths = dict(
                                    filter(lambda x: not x[1] == np.inf, possible_paths))

                                total = sum(shortest_paths.values())
                                n_shortest_paths = len(shortest_paths) - 1.0

                                if total > 0.0 and n > 1:
                                    s = n_shortest_paths / (n - 1)
                                    cc = (n_shortest_paths / total) * s
                                closeness_centrality[r] = cc
                            centrality_dict[guilda] = pd.Series(closeness_centrality).mean()
                            node_count_dict[guilda] = graph.number_of_nodes()
                    elif centrality == 'Page rank':
                        for guilda, graph in graph_dict.items():
                            measure_dict = nx.pagerank(graph)
                            centrality_dict[guilda] = pd.Series(measure_dict).mean()
                            node_count_dict[guilda] = graph.number_of_nodes()
                    elif centrality == 'Densidade':
                        for guilda, graph in graph_dict.items():
                            measure_dict = nx.density(graph)
                            centrality_dict[guilda] = pd.Series(measure_dict).mean()
                            node_count_dict[guilda] = graph.number_of_nodes()

                    node_count_series = pd.Series(node_count_dict)

                    df = pd.DataFrame(centrality_dict.items(),columns=['Guilda', 'Measure'])
                    # Exclui 'Guilda' como 'Unknown'
                    df = df.loc[df['Guilda'] != 'Unknown']
                    df = df.sort_values('Measure', ascending=True)

                    df['Num_nodes'] = df['Guilda'].map(node_count_series)
                    df = df.loc[~((df['Num_nodes'] == 1) & (df['Measure'] == 1))]# exclui o caso de redes com 1 (total de medida) e 1 nós apenas.

                    fig, ax = plt.subplots()

                    with col2:
                        chart = alt.Chart(df.reset_index()).mark_bar().encode(
                            x=alt.X('Measure:Q', title='Centrality'),
                            y=alt.Y('Guilda:N', sort='-x', title=' '),
                            tooltip=[alt.Tooltip('Guilda:N'), alt.Tooltip('Measure:Q'), alt.Tooltip('Num_nodes:Q')]
                        ).properties(
                            title='Grupos de destaque'
                        ).configure_axisY(
                            labelLimit=200  # Increase this value to allow for wider labels on the y-axis    
                        ).configure_title(
                            offset=20
                        )

                        st.altair_chart(chart, use_container_width=True)

    ########################################
                    with col3:

                        # cria o driver python
                        driver = GraphDatabase.driver(
           			NEO4J_URL, auth=(NEO4J_USER, NEO4J_PASSWORD))

                        # função neo4j para importar a rede do localhost neo4j
                        # o substring retira de PERTENCE_MES/ANO MES/ANO para fazer o filtro com rodada
                        def get_eua3(tx, nome, dimensao):
                            result = tx.run("""
                                MATCH (n)-[r]->(g:Guilda {nome: $nome})
                                WITH n, SUBSTRING(type(r), 9) AS rodadaValue
                                WITH COLLECT({email: n.email, rodada: rodadaValue}) AS emailData

                                UNWIND emailData AS data1
                                UNWIND emailData AS data2

                                MATCH (n1)-[r2 {dimensao: $dimensao, rodada: data1.rodada}]->(n2)
                                WHERE n1.email = data1.email AND n2.email = data2.email AND data1.rodada = data2.rodada
                                RETURN n1, r2, r2.rodada AS rodada, n2
                                """,
                                            nome=nome,
                                            dimensao=dimensao)

                            records = list(result)
                            summary = result.consume()

                            return records, summary
                        
                            records

                        with driver.session() as session:

                            query_cg_3 = f"MATCH ()-[]->(c1:Guilda) RETURN DISTINCT c1.nome AS c1_nome"
                            rede_listuilda = [record["c1_nome"]
                                            for record in session.run(query_cg_3)]

                            nome = st.selectbox(
                                "Selecione a guilda:", sorted(rede_listuilda))

                            query_dimensao = f"MATCH (n)-[r]->(g:Guilda) WHERE g.nome IN ['{nome}'] WITH collect(n) AS GuildaNodes UNWIND GuildaNodes as n1 UNWIND GuildaNodes as n2 MATCH (n1)-[r2]->(n2) WHERE id(n1) < id(n2) AND r2.dimensao <> 'None' RETURN DISTINCT r2.dimensao AS r2_dim"

                            dim_g = [record["r2_dim"]for record in session.run(query_dimensao)]

                            dim = st.selectbox("Selecione a dimensão", sorted(dim_g), key='dim')

                            records, summary = session.execute_read(
                                get_eua3, nome=nome, dimensao=dim)

                            G2 = nx.MultiDiGraph()

                            for record in records:
                                source_id = record["n1"].element_id
                                target_id = record["n2"].element_id
                                G2.add_node(
                                    source_id, label=record["n1"]["email"])
                                G2.add_node(
                                    target_id, label=record["n2"]["email"])
                                G2.add_edge(
                                    source_id, target_id, label=record["r2"], rodada=record["r2"]['rodada'])

                            G2_separated = {}

                            # pecorre cada edge no grafo para extrai o valor rodada de cada edge
                            for u, v, data in G2.edges(data=True):
                                rodada = data['rodada']

                                # verifica se um grafo para este valor rodada já existe. cria um novo grafo caso contrário
                                if rodada not in G2_separated:
                                    G2_separated[rodada] = nx.MultiDiGraph()

                                # Add um edge para o grafo para este valor 'rodada'
                                G2_separated[rodada].add_edge(u, v, **data)

                            centrality = st.selectbox(
                                'Selecione uma medida de centralização:', [
                                    'InDegree', 'OutDegree', 'Betweenness', 'Closeness', 'Page rank',
                                    'Densidade'], key='centrality ab')

                            if centrality == 'InDegree':
                                medida = {rodada: nx.in_degree_centrality(
                                    graph) for rodada, graph in G2_separated.items()}
                            elif centrality == 'OutDegree':
                                medida = {rodada: nx.out_degree_centrality(
                                    graph) for rodada, graph in G2_separated.items()}
                            elif centrality == 'Betweenness':
                                medida = {rodada: nx.betweenness_centrality(
                                    graph) for rodada, graph in G2_separated.items()}

                            # elif centrality == 'Closeness':
                            #     medida = {rodada: nx.closeness_centrality(
                            #         graph) for rodada, graph in G2_separated.items()}

                            elif centrality == 'Closeness':

                                import scipy.sparse
                                import scipy.sparse.csgraph

                                medida = {}

                                for rodada, graph in G2_separated.items():
                                    A = nx.adjacency_matrix(graph).tolil()
                                    D = scipy.sparse.csgraph.floyd_warshall(
                                        A, directed=False, unweighted=False)

                                    n = D.shape[0]
                                    closeness_centrality = {}
                                    for r in range(0, n):
                                        cc = 0.0

                                        possible_paths = list(
                                            enumerate(D[r, :]))
                                        shortest_paths = dict(
                                            filter(lambda x: not x[1] == np.inf, possible_paths))

                                        total = sum(shortest_paths.values())
                                        n_shortest_paths = len(
                                            shortest_paths) - 1.0
                                        if total > 0.0 and n > 1:
                                            s = n_shortest_paths / (n - 1)
                                            cc = (n_shortest_paths / total) * s
                                        closeness_centrality[r] = cc

                                    medida[rodada] = closeness_centrality

                            elif centrality == 'Page rank':
                                medida = {rodada: nx.pagerank(
                                    graph) for rodada, graph in G2_separated.items()}
                            elif centrality == 'Densidade':
                                medida_d = {rodada: nx.density(
                                    graph) for rodada, graph in G2_separated.items()}

                            with col4:
                                if centrality != 'Densidade':
                                    df = pd.DataFrame(
                                        columns=["rodada", "média", "mediana", "desvio padrão"])

                                    for key, values in medida.items():
                                        mean_val = np.mean(
                                            list(values.values()))
                                        median_val = np.median(
                                            list(values.values()))
                                        std_dev_val = np.std(
                                            list(values.values()))

                                        new_data = pd.DataFrame({"rodada": [key], "média": [mean_val], "mediana": [
                                                                median_val], "desvio padrão": [std_dev_val]})

                                        df = pd.concat(
                                            [df, new_data], ignore_index=True)

                                    month_to_num = {'JAN': '01', 'FEV': '02', 'MAR': '03', 'ABR': '04', 'MAI': '05', 'JUN': '06', 'JUL': '07', 'AGO': '08', 'SET': '09', 'OUT': '10', 'NOV': '11', 'DEZ': '12'
                                                    }
                                    df['date'] = df['rodada'].map(
                                        lambda x: month_to_num[x[:3]] + x[3:])
                                    df['date'] = pd.to_datetime(
                                        df['date'], format='%m%Y')

                                    df = df.sort_values('date')

                                    df = df.drop(columns=['date'])

                                    source = df.melt(
                                        'rodada', var_name='measure', value_name='value')

                                    colors = ['#cb0e15', '#0d5cd3', '#1c9937']

                                    line_chart = alt.Chart(source).mark_line(point=True).encode(
                                        alt.X('rodada', title='Rodada',
                                            sort=None),
                                        alt.Y('value', title=''),
                                        color=alt.Color(
                                            'measure', title='Medida')
                                    ).properties(
                                        title='Séries temporais de grupo',
                                        height=450
                                    ).configure_range(
                                        category=alt.RangeScheme(colors)
                                    ).interactive()

                                    st.altair_chart(
                                        line_chart, use_container_width=True)

                            with col4:
                                if centrality == 'Densidade':
                                    df = pd.DataFrame(list(medida_d.items()), columns=[
                                        'Category', 'Value'])

                                    month_to_num = {'JAN': '01', 'FEV': '02', 'MAR': '03', 'ABR': '04', 'MAI': '05', 'JUN': '06', 'JUL': '07', 'AGO': '08', 'SET': '09', 'OUT': '10', 'NOV': '11', 'DEZ': '12'
                                                    }
                                    df['date'] = df['Category'].map(
                                        lambda x: month_to_num[x[:3]] + x[3:])
                                    df['date'] = pd.to_datetime(
                                        df['date'], format='%m%Y')

                                    df = df.sort_values('date')

                                    df = df.drop(columns=['date'])

                                    chart = alt.Chart(df).mark_line(point=True).encode(
                                        alt.X('Category',
                                            title='Rodada', sort=None),
                                        alt.Y('Value', title=''),
                                    ).properties(
                                        title='Séries temporais de grupo',
                                        height=450
                                    ).interactive()

                                    st.altair_chart(
                                        chart, use_container_width=True)


    ##############################################
    ###### cliente ##################
    with tab3:
        col1, col2 = st.columns([1, 3])

        st.markdown("<br>"*2, unsafe_allow_html=True)

        col3, col4 = st.columns([1, 3])

        with col1:
            # cria o driver python
            driver = GraphDatabase.driver(
            	NEO4J_URL, auth=(NEO4J_USER, NEO4J_PASSWORD))

            # função neo4j para importar a rede do localhost neo4j
            def get_eua3(tx, dimensao, rodada, email):
                result = tx.run("""
                    MATCH (c1)-[b1 {rodada: $rodada, dimensao: $dimensao}]->(c2)
                    WHERE c1.email IN $email OR c2.email IN $email
                    RETURN c1, b1, c2
                """,
                                email=email,
                                dimensao=dimensao,
                                rodada=rodada)

                records = list(result)
                summary = result.consume()

                return records, summary

            # Inicia a sessão do driver
            with driver.session() as session:

                # caixas seleção dimensao e rodada
                with col1:
                    rodada1_list = [record["b1_rodada"] for record in session.run(
                        "MATCH ()-[b1:BUSCOU]->() RETURN DISTINCT b1.rodada AS b1_rodada")]
                    
                    def to_date(s):    
                        months = ["JAN", "FEV", "MAR", "ABR", "MAI", "JUN", "JUL", "AGO", "SET", "OUT", "NOV", "DEZ"]
                        month, year = s[:3], s[3:]
                        return datetime(int(year), months.index(month) + 1, 1)
                    rodada1_list.sort(key=to_date, reverse=True)

                    rod = st.selectbox("Selecione a rodada:",rodada1_list, key='selectbox_rod')

                with col1:
                    dimension_list = [record["b1_dimensao"] for record in session.run(
                        f"MATCH ()-[b1:BUSCOU {{rodada: '{rod}'}}]->() RETURN DISTINCT b1.dimensao AS b1_dimensao")]
                    dim = st.selectbox("Selecione a dimensão:", sorted(dimension_list), key='selectbox_dim')

                # adiciona os botões
                e_list = [(record["c1_email"], record["c2_email"]) for record in session.run(  # pega todos os nomes em todas as relações existentes para cada dimensao e rodada
                    "MATCH (c1)-[b1 {rodada: $rodada, dimensao: $dimensao}]->(c2) RETURN DISTINCT c1.email AS c1_email, c2.email AS c2_email",
                    dimensao=dim,
                    rodada=rod)]
                email_list = {name for tuple_ in e_list for name in tuple_}
                email_list = list(email_list)

                # executa a query apos construida todas as variaveis de entrada
                records, summary = session.execute_read(
                    get_eua3, dimensao=dim, rodada=rod, email=email_list)

                text_data = []
                for record in records:
                    data = record.data()  # Extrai os dados do registro como um dicionário
                    # Converte o dicionário para uma string e adiciona à lista
                    text_data.append(str(data))

                # Converte a lista em uma única string
                text = " ".join(text_data)
                valores_email = re.findall("'email': '([^']+)", text)

                # Converte a lista para um conjunto para obter valores únicos de email
                email_unicos = set(valores_email)
                # 'email_unicos', email_unicos

                import json
                # converte em lista
                email_unicos = list(email_unicos)
                # formata a lista como array json (mesma sintaxe da lista cypher)
                email_unicos = json.dumps(email_unicos)

                records_ind, summary_ind = session.execute_read(
                    get_eua3, dimensao=dim, rodada=rod, email=email_list)

                text_data_ind = []
                for record in records_ind:
                    data = record.data()  # Extrai os dados do registro como um dicionário
                    # Converte o dicionário para uma string e adiciona à lista
                    text_data_ind.append(str(data))

                # Converte a lista em uma única string
                text_ind = " ".join(text_data_ind)
                valores_email_ind = re.findall("'email': '([^']+)", text_ind)

                # Converte a lista para um conjunto para obter valores únicos de email
                email_unicos_ind = set(valores_email_ind)
                # 'email_unicos', email_unicos

                import json
                # converte em lista
                email_unicos_ind = list(email_unicos_ind)
                # formata a lista como array json (mesma sintaxe da lista cypher)
                email_unicos_ind = json.dumps(email_unicos_ind)

                query_cg_2 = f"MATCH (c:Colaborador)-[:ALOCADO_{rod}]->(c1:Cliente) RETURN DISTINCT c1.nome AS c1_nome"
                rede_list_cliente = [record["c1_nome"]
                                    for record in session.run(query_cg_2)]
                query = f"MATCH (e)-[:ALOCADO_{rod}]->(c1) RETURN DISTINCT e.email AS email, c1.nome AS c1_nome"
                cliente = [(record["email"], record['c1_nome'])
                        for record in session.run(query)]
                # cria lista para usar para criar propriedades de cliente
                modified_list_c = []
                for sublist in cliente:
                    modified_dict = {
                        "email": sublist[0],
                        "cliente_guilda": sublist[1]
                    }
                    modified_list_c.append(modified_dict)

                G = nx.DiGraph()

                # pega todas as caracteristicas laços e nodos de 'cada vez'
                # adiciona nodos e laços a partir dos resultados do query da rede neo4j
                for record in records:
                    # 'record', record
                    source_id = record["c1"].element_id
                    target_id = record["c2"].element_id
                    G.add_node(source_id, label=record["c1"]["email"])
                    G.add_node(target_id, label=record["c2"]["email"])
                    G.add_edge(source_id, target_id, label=record["b1"])

                graph_dict = {}
                original_G3 = G.copy()

                for cliente in rede_list_cliente:
                    G3 = original_G3.copy() 
                    for i in range(len(modified_list_c)):
                        if modified_list_c[i]['cliente_guilda'] == cliente:
                            for node in G3.nodes():
                                if G3.nodes[node]['label'] == modified_list_c[i]['email']:
                                    G3.nodes[node]['color'] = '#11d63f'
                                    G3.nodes[node]['cliente'] = cliente

                    nodes_to_remove = [node for node, data in G3.nodes(
                        data=True) if data.get('color') != '#11d63f']
                    G3.remove_nodes_from(nodes_to_remove)
                    graph_dict[cliente] = G3

                centrality_dict_c = {}
                node_count_dict = {}

                with col1:
                    centrality = st.selectbox(
                        'Selecione uma medida de centralização:', [
                            'Degree', 'Betweenness', 'Closeness', 'Page rank',
                            'Densidade'
                        ], key = 'centrality = bc')
                    if centrality == 'Degree':
                        for cliente, graph in graph_dict.items():
                            measure_dict = nx.in_degree_centrality(graph)
                            centrality_dict_c[cliente] = pd.Series(measure_dict).mean()
                            node_count_dict[cliente] = graph.number_of_nodes()
                    elif centrality == 'Betweenness':
                        for cliente, graph in graph_dict.items():
                            measure_dict = nx.betweenness_centrality(graph)
                            centrality_dict_c[cliente] = pd.Series(measure_dict).mean()
                            node_count_dict[cliente] = graph.number_of_nodes()
                    elif centrality == 'Closeness':
                        for cliente, graph in graph_dict.items():

                            D = nx.to_numpy_array(graph)
                            n = D.shape[0]
                            closeness_centrality = {}
                            for r in range(0, n):
                                cc = 0.0
                                possible_paths = list(enumerate(D[r, :]))

                                shortest_paths = dict(
                                    filter(lambda x: not x[1] == np.inf, possible_paths))

                                total = sum(shortest_paths.values())
                                n_shortest_paths = len(shortest_paths) - 1.0

                                if total > 0.0 and n > 1:
                                    s = n_shortest_paths / (n - 1)
                                    cc = (n_shortest_paths / total) * s
                                closeness_centrality[r] = cc
                            centrality_dict_c[cliente] = pd.Series(closeness_centrality).mean()
                            node_count_dict[cliente] = graph.number_of_nodes()
                    elif centrality == 'Page rank':
                        for cliente, graph in graph_dict.items():
                            measure_dict = nx.pagerank(graph)
                            centrality_dict_c[cliente] = pd.Series(measure_dict).mean()
                            node_count_dict[cliente] = graph.number_of_nodes()
                    elif centrality == 'Densidade':
                        for cliente, graph in graph_dict.items():
                            measure_dict = nx.density(graph)
                            centrality_dict_c[cliente] = pd.Series(measure_dict).mean()
                            node_count_dict[cliente] = graph.number_of_nodes()

                    node_count_series = pd.Series(node_count_dict)

                    df = pd.DataFrame(centrality_dict_c.items(), columns=['Cliente', 'Measure'])
                    df = df.loc[df['Cliente'] != 'Unknown']
                    df = df.sort_values('Measure', ascending=True)

                    df['Num_nodes'] = df['Cliente'].map(node_count_series)
                    df = df.loc[~((df['Num_nodes'] == 1) & (df['Measure'] == 1))]# exclui o caso de redes com 1 (total de medida) e 1 nós apenas.

                    fig, ax = plt.subplots()

                    with col2:
                        chart = alt.Chart(df.reset_index()).mark_bar().encode(
                            x=alt.X('Measure:Q', title='Centrality'),
                            y=alt.Y('Cliente:N', sort='-x', title=' '),
                            tooltip=[alt.Tooltip('Cliente:N'), alt.Tooltip('Measure:Q'), alt.Tooltip('Num_nodes:Q')]
                        ).properties(
                            title='Grupos de destaque'
                        ).configure_axisY(
                            labelLimit=200  # Increase this value to allow for wider labels on the y-axis    
                        ).interactive()

                        st.altair_chart(chart, use_container_width=True)

    ##############
        with col3:

            # cria o driver python
            driver = GraphDatabase.driver(
            	NEO4J_URL, auth=(NEO4J_USER, NEO4J_PASSWORD))

            # função neo4j para importar a rede do localhost neo4j
            # o substring retira de ALOCADO_MES/ANO para fazer o filtro com rodada
            def get_eua3(tx, nome, dimensao):
                result = tx.run("""
                    MATCH (n)-[r]->(g:Cliente {nome: $nome})
                    WITH n, SUBSTRING(type(r), 8) AS rodadaValue
                    WITH COLLECT({email: n.email, rodada: rodadaValue}) AS emailData

                    UNWIND emailData AS data1
                    UNWIND emailData AS data2

                    MATCH (n1)-[r2 {dimensao: $dimensao, rodada: data1.rodada}]->(n2)
                    WHERE n1.email = data1.email AND n2.email = data2.email AND data1.rodada = data2.rodada
                    RETURN n1, r2, r2.rodada AS rodada, n2
                    """,
                                nome=nome,
                                dimensao=dimensao)

                records = list(result)
                summary = result.consume()

                return records, summary

            with driver.session() as session:

                query_cg_3 = f"MATCH ()-[]->(c1:Cliente) RETURN DISTINCT c1.nome AS c1_nome"
                rede_listuilda = [record["c1_nome"]
                                for record in session.run(query_cg_3)]

                nome = st.selectbox(
                    "Selecione um cliente:", sorted(rede_listuilda))

                query_dimensao = f"MATCH (n)-[r]->(g:Cliente) WHERE g.nome IN ['{nome}'] WITH collect(n) AS clienteNodes UNWIND clienteNodes as n1 UNWIND clienteNodes as n2 MATCH (n1)-[r2]->(n2) WHERE id(n1) < id(n2) AND r2.dimensao <> 'None' RETURN DISTINCT r2.dimensao AS r2_dim"

                dim_g = [record["r2_dim"]
                        for record in session.run(query_dimensao)]

                dim = st.selectbox("Selecione a dimensão", sorted(dim_g))

                records, summary = session.execute_read(
                    get_eua3, nome=nome, dimensao=dim)

                G2 = nx.MultiDiGraph()

                for record in records:
                    source_id = record["n1"].element_id
                    target_id = record["n2"].element_id
                    G2.add_node(source_id, label=record["n1"]["email"])
                    G2.add_node(target_id, label=record["n2"]["email"])
                    G2.add_edge(source_id, target_id, label=record["r2"], rodada=record["r2"]['rodada'])

                G2_separated = {}

                for u, v, data in G2.edges(data=True):
                    rodada = data['rodada']

                    if rodada not in G2_separated:
                        G2_separated[rodada] = nx.MultiDiGraph()

                    G2_separated[rodada].add_edge(u, v, **data)

                centrality_dict = []

                centrality = st.selectbox(
                    'Selecione uma medida de centralização:', [
                        'InDegree', 'OutDegree', 'Betweenness', 'Closeness', 'Page rank',
                        'Densidade'], key='c')

                if centrality == 'InDegree':
                    medida = {rodada: nx.in_degree_centrality(
                        graph) for rodada, graph in G2_separated.items()}
                elif centrality == 'OutDegree':
                    medida = {rodada: nx.out_degree_centrality(
                        graph) for rodada, graph in G2_separated.items()}
                elif centrality == 'Betweenness':
                    medida = {rodada: nx.betweenness_centrality(
                        graph) for rodada, graph in G2_separated.items()}
                # elif centrality == 'Closeness':
                #     medida = {rodada: nx.closeness_centrality(
                #         graph) for rodada, graph in G2_separated.items()}

                elif centrality == 'Closeness':

                    import scipy.sparse
                    import scipy.sparse.csgraph

                    medida = {}

                    for rodada, graph in G2_separated.items():
                        A = nx.adjacency_matrix(graph).tolil()
                        D = scipy.sparse.csgraph.floyd_warshall(
                            A, directed=False, unweighted=False)

                        n = D.shape[0]
                        closeness_centrality = {}
                        for r in range(0, n):
                            cc = 0.0

                            possible_paths = list(enumerate(D[r, :]))
                            shortest_paths = dict(
                                filter(lambda x: not x[1] == np.inf, possible_paths))

                            total = sum(shortest_paths.values())
                            n_shortest_paths = len(shortest_paths) - 1.0
                            if total > 0.0 and n > 1:
                                s = n_shortest_paths / (n - 1)
                                cc = (n_shortest_paths / total) * s
                            closeness_centrality[r] = cc

                        medida[rodada] = closeness_centrality

                elif centrality == 'Page rank':
                    medida = {rodada: nx.pagerank(
                        graph) for rodada, graph in G2_separated.items()}
                elif centrality == 'Densidade':
                    medida_d = {rodada: nx.density(
                        graph) for rodada, graph in G2_separated.items()}

                with col4:
                    if centrality != 'Densidade':
                        df = pd.DataFrame(
                            columns=["rodada", "média", "mediana", "desvio padrão"])

                        for key, values in medida.items():
                            mean_val = np.mean(list(values.values()))
                            median_val = np.median(list(values.values()))
                            std_dev_val = np.std(list(values.values()))

                            new_data = pd.DataFrame({"rodada": [key], "média": [mean_val], "mediana": [
                                                    median_val], "desvio padrão": [std_dev_val]})
                            df = pd.concat(
                                [df, new_data], ignore_index=True)

                        month_to_num = {'JAN': '01', 'FEV': '02', 'MAR': '03', 'ABR': '04', 'MAI': '05', 'JUN': '06', 'JUL': '07', 'AGO': '08', 'SET': '09', 'OUT': '10', 'NOV': '11', 'DEZ': '12'
                                        }
                        df['date'] = df['rodada'].map(
                            lambda x: month_to_num[x[:3]] + x[3:])
                        df['date'] = pd.to_datetime(
                            df['date'], format='%m%Y')

                        df = df.sort_values('date')

                        df = df.drop(columns=['date'])

                        source = df.melt(
                            'rodada', var_name='measure', value_name='value')

                        colors = ['#cb0e15','#0d5cd3' ,'#1c9937']

                        line_chart = alt.Chart(source).mark_line(point=True).encode(
                            alt.X('rodada', title='Rodada', sort=None),
                            alt.Y('value', title=''),
                            color=alt.Color('measure', title='Medida')
                        ).properties(
                            title='Séries temporais de grupo',
                            height=450
                        ).configure_range(
                            category=alt.RangeScheme(colors)
                        ).interactive()

                        st.altair_chart(
                            line_chart, use_container_width=True)

                with col4:
                    if centrality == 'Densidade':
                        df = pd.DataFrame(list(medida_d.items()), columns=[
                            'Category', 'Value'])

                        month_to_num = {'JAN': '01', 'FEV': '02', 'MAR': '03', 'ABR': '04', 'MAI': '05', 'JUN': '06', 'JUL': '07', 'AGO': '08', 'SET': '09', 'OUT': '10', 'NOV': '11', 'DEZ': '12'
                                        }
                        df['date'] = df['Category'].map(
                            lambda x: month_to_num[x[:3]] + x[3:])
                        df['date'] = pd.to_datetime(
                            df['date'], format='%m%Y')

                        df = df.sort_values('date')

                        df = df.drop(columns=['date'])

                        chart = alt.Chart(df).mark_line(point=True).encode(
                            alt.X('Category', title='Rodada', sort=None),
                            alt.Y('Value', title=''),
                        ).properties(
                            title='Séries temporais de grupo',
                            height=450
                        ).interactive()

                        st.altair_chart(chart, use_container_width=True)
