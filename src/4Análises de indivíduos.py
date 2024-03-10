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
            def get_eua3(tx, dimensao, rodada):
                result = tx.run("""
                    MATCH (c1)-[b1 {rodada: $rodada, dimensao: $dimensao}]->(c2)
                    RETURN c1, b1, c2
                """,
                                dimensao=dimensao,
                                rodada=rodada)

                records = list(result)
                summary = result.consume()

                return records, summary

            # Inicia a sessão do driver
            with driver.session() as session:

                # caixas seleção dimensao e rodada

                rodada1_list = [record["b1_rodada"] for record in session.run(
                    "MATCH ()-[b1:BUSCOU]->() RETURN DISTINCT b1.rodada AS b1_rodada")]
                def to_date(s):    
                    months = ["JAN", "FEV", "MAR", "ABR", "MAI", "JUN", "JUL", "AGO", "SET", "OUT", "NOV", "DEZ"]
                    month, year = s[:3], s[3:]
                    return datetime(int(year), months.index(month) + 1, 1)
                rodada1_list.sort(key=to_date, reverse=True)
                rod = st.selectbox("Selecione a rodada:", rodada1_list)

                dimension_list = [record["b1_dimensao"] for record in session.run(
                    f"MATCH ()-[b1:BUSCOU {{rodada: '{rod}'}}]->() RETURN DISTINCT b1.dimensao AS b1_dimensao")]
                dim = st.selectbox("Selecione a dimensão:", sorted(dimension_list))

                # executa a query apos construida todas as variaveis de entrada
                records, summary = session.execute_read(
                    get_eua3, dimensao=dim, rodada=rod)

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

                centrality = st.selectbox('Selecione uma medida de centralização:', [
                    'InDegree', 'OutDegree', 'Betweenness', 'Closeness', 'Page rank'])

                if centrality == 'InDegree':
                    medida = nx.in_degree_centrality(G)
                elif centrality == 'OutDegree':
                    medida = nx.out_degree_centrality(G)
                elif centrality == 'Betweenness':
                    medida = nx.betweenness_centrality(G)
                elif centrality == 'Closeness':
                    A = nx.adjacency_matrix(G)
                    nodes = A.shape[0]
                    node_order = list(G.nodes())

                    D = scipy.sparse.csgraph.floyd_warshall(
                        A, directed=False, unweighted=False)
                    closeness_centrality = {}
                    for r in range(0, nodes):
                        cc = 0.0
                        possible_paths = list(enumerate(D[r, :]))
                        shortest_paths = dict(
                            filter(lambda x: not x[1] == np.inf, possible_paths))
                        total = sum(shortest_paths.values())
                        n_shortest_paths = len(shortest_paths) - 1.0
                        if total > 0.0 and nodes > 1:
                            s = n_shortest_paths / (nodes - 1)
                            cc = (n_shortest_paths / total) * s
                        closeness_centrality[node_order[r]] = cc

                    id_to_email = {node: data["label"]
                                for node, data in G.nodes(data=True)}

                    medida = {id_to_email[node]: centrality_value for node,
                            centrality_value in closeness_centrality.items() if node in id_to_email}

                    df = pd.DataFrame(list(medida.items()),
                                    columns=['Email', centrality])
                    df_cl = df.sort_values(centrality, ascending=False)

                elif centrality == 'Page rank':
                    medida = nx.pagerank(G)

                # mapeia node ids para os labels dos emails
                id_to_email = {node: data["label"]for node, data in G.nodes(data=True)}
                email_centrality = {id_to_email[node]: centrality_value for node,
                                    centrality_value in medida.items() if node in id_to_email}
                df = pd.DataFrame(list(email_centrality.items()),columns=['Email', centrality])
                df = df.sort_values(centrality, ascending=False)

                if centrality != 'Closeness':
                    with col2:
                        chart = alt.Chart(df[:20].reset_index()).mark_bar().encode(
                            x=alt.X(f"{centrality}:Q", title=centrality),
                            y=alt.Y('Email:N', sort='-x', title=' ')
                        ).properties(
                            title='Destaques individuais'
                        ).configure_title(
                            offset=20 
                        )

                        st.altair_chart(chart, use_container_width=True)
                elif centrality == 'Closeness':
                    with col2:
                        chart = alt.Chart(df_cl[:20].reset_index()).mark_bar().encode(
                            x=alt.X(f"{centrality}:Q", title=centrality),
                            y=alt.Y('Email:N', sort='-x', title=' ')
                        ).properties(
                            title='Destaques individuais'
                        ).configure_title(
                            offset=20
                        ).configure_axisY(
                            labelLimit=200
                        )

                        st.altair_chart(chart, use_container_width=True)
    ######################
        with col3:

            # Python driver
            driver = GraphDatabase.driver(
            	NEO4J_URL, auth=(NEO4J_USER, NEO4J_PASSWORD))

            def get_eua3(tx, dimensao):
                result = tx.run("""
                    MATCH (c1)-[b1 {dimensao: $dimensao}]->(c2)
                    RETURN c1, b1, c2
                """,
                                dimensao=dimensao)

                records = list(result)
                summary = result.consume()

                return records, summary

            with driver.session() as session:
                dimension_list = [record["b1_dimensao"] for record in session.run(
                    f"MATCH ()-[b1:BUSCOU]->() RETURN DISTINCT b1.dimensao AS b1_dimensao")]
                dim = st.selectbox("Selecione a dimensão:", sorted(dimension_list), key='geral2')

                e_list = [(record["c1_email"], record["c2_email"]) for record in session.run(
                    "MATCH (c1)-[b1 {dimensao: $dimensao}]->(c2) RETURN DISTINCT c1.email AS c1_email, c2.email AS c2_email",
                    dimensao=dim)]
                email_list = {name for tuple_ in e_list for name in tuple_ if name.strip()}
                email_list = list(email_list)

                source = st.multiselect("Selecione um ou mais EuA3:", sorted(email_list), key='ind')

                records, summary = session.execute_read(
                    get_eua3, dimensao=dim)
                
                G = nx.MultiDiGraph()

                for record in records:
                    source_id = record["c1"].element_id
                    target_id = record["c2"].element_id
                    G.add_node(source_id, label=record["c1"]["email"])
                    G.add_node(target_id, label=record["c2"]["email"])
                    G.add_edge(source_id, target_id, label=record["b1"], rodada=record["b1"]["rodada"])

                rodada_values = nx.get_edge_attributes(G, 'rodada')
                graphs_by_rodada = defaultdict(nx.MultiDiGraph)

                for (u, v, k), rodada in rodada_values.items():
                    if u not in graphs_by_rodada[rodada]:
                        graphs_by_rodada[rodada].add_node(u, **G.nodes[u])
                    if v not in graphs_by_rodada[rodada]:
                        graphs_by_rodada[rodada].add_node(v, **G.nodes[v])
                    
                    graphs_by_rodada[rodada].add_edge(u, v)

                centrality = st.selectbox(
                    'Selecione uma medida de centralidade:', [
                        'InDegree', 'OutDegree', 'Betweenness', 'Closeness', 'Page rank'], key='a')

                centrality_measures_by_rodada = {}
                for rodada, graph in graphs_by_rodada.items():
                    if centrality == 'InDegree':
                        medida = nx.in_degree_centrality(graph)
                    elif centrality == 'OutDegree':
                        medida = nx.out_degree_centrality(graph)
                    elif centrality == 'Betweenness':
                        medida = nx.betweenness_centrality(graph)
                    elif centrality == 'Closeness':
                        medida = nx.closeness_centrality(graph)
                    elif centrality == 'Page rank':
                        medida = nx.pagerank(graph)

                    medida_with_labels = {f"{node_id}|{graph.nodes[node_id]['label']}": value for node_id, value in medida.items()}
                    centrality_measures_by_rodada[rodada] = medida_with_labels
                
                records = []
                for rodada, measures in centrality_measures_by_rodada.items():
                    for key, value in measures.items():
                        # Extract email part using string splitting
                        email = key.split('|')[-1]
                        if email in source:
                            records.append((email, value, rodada))


                df = pd.DataFrame(records, columns=['label', 'values', 'rodada'])

                if source: 
                    month_to_num = {'JAN': '01', 'FEV': '02', 'MAR': '03', 'ABR': '04', 'MAI': '05',
                                    'JUN': '06', 'JUL': '07', 'AGO': '08', 'SET': '09', 'OUT': '10', 'NOV': '11', 'DEZ': '12'}
                    df['date'] = df['rodada'].map(
                        lambda x: month_to_num[x[:3]] + x[3:])
                    df['date'] = pd.to_datetime(df['date'], format='%m%Y')
                    df = df.sort_values('date')

                    df = df.drop(columns=['date'])

                    with col4:
                        line_chart = alt.Chart(df).mark_line(point=True).encode(
                            alt.X('rodada:O', title='Rodada', sort=None),
                            alt.Y('values:Q', title=''),
                            alt.Color('label:N', title='EuA3')
                        ).properties(
                            title='Série temporal individual',
                            height=450
                        ).interactive()

                        st.altair_chart(line_chart, use_container_width=True)

    ###############################
    # Guilda ########################
    ##############
    with tab2:

        col1, col2 = st.columns([1, 3])

        st.markdown("<br>"*2, unsafe_allow_html=True)

        col3, col4 = st.columns([1, 3])

        with col1:
            # cria o driver python
            driver = GraphDatabase.driver(
                NEO4J_URL, auth=(NEO4J_USER, NEO4J_PASSWORD))
                
            # função neo4j para importar a rede do localhost neo4j
            def get_eua3(session, nome, dimensao, rodada):
                query = f"""
                    MATCH (n)-[r:PERTENCE_{rodada}]->(guilda:Guilda {{nome: $nome}})
                    WITH collect(n) AS GuildaNodes, guilda
                    UNWIND GuildaNodes as n1
                    UNWIND GuildaNodes as n2

                    MATCH (n1)-[r2 {{rodada: $rodada, dimensao: $dimensao}}]->(n2)
                    RETURN DISTINCT n1, r2, n2
                """
                def read_tx(tx):
                    result = tx.run(query, nome=nome, rodada=rodada, dimensao=dimensao)
                    records = list(result)
                    summary = result.consume()
                    return records, summary

                records, summary = session.execute_read(read_tx)
                return records, summary
            
            with driver.session() as session:

                with col1:

                    query_rodada = query_rodada = f"MATCH (n)-[r]->(n2) WHERE r.rodada IS NOT NULL RETURN DISTINCT r.rodada AS r_rod"
                    rod_g = [record["r_rod"]for record in session.run(query_rodada)]
                    def to_date(s):    
                        months = ["JAN", "FEV", "MAR", "ABR", "MAI", "JUN", "JUL", "AGO", "SET", "OUT", "NOV", "DEZ"]
                        month, year = s[:3], s[3:]
                        return datetime(int(year), months.index(month) + 1, 1)
                    rod_g.sort(key=to_date, reverse=True)
                    rod = st.selectbox("Selecione a rodada", rod_g)
                    
                    query_dimensao = f"MATCH (n1)-[r]->(n2) WHERE r.dimensao IS NOT NULL RETURN DISTINCT r.dimensao AS r_dim"
                    dim_g = [record["r_dim"]for record in session.run(query_dimensao)]
                    dim = st.selectbox("Selecione a dimensão", sorted(dim_g))

                    query_cg_3 = f"MATCH ()-[:PERTENCE_{rod}]->(g:Guilda) RETURN DISTINCT g.nome AS g_nome"
                    rede_list_cliente = [record["g_nome"]for record in session.run(query_cg_3)]
                    nome = st.selectbox("Selecione a Guilda:", sorted(rede_list_cliente))

                    records, summary = get_eua3(session, nome=nome, dimensao=dim, rodada=rod)

                    G = nx.DiGraph()

                    for record in records:
                        source_id = record["n1"].element_id
                        target_id = record["n2"].element_id
                        G.add_node(source_id, label=record["n1"]["email"])
                        G.add_node(target_id, label=record["n2"]["email"])
                        G.add_edge(source_id, target_id, label=record["r2"])

                    centrality = st.selectbox(
                        'Selecione uma medida de centralidade:',
                        ['InDegree', 'OutDegree', 'Betweenness', 'Closeness', 'Page rank'], key='centrality abc')

                    if centrality == 'InDegree':
                        medida = nx.in_degree_centrality(G)
                    elif centrality == 'OutDegree':
                        medida = nx.out_degree_centrality(G)
                    elif centrality == 'Betweenness':
                        medida = nx.betweenness_centrality(G)
                    elif centrality == 'Closeness':
                        medida = nx.closeness_centrality(G)
                    elif centrality == 'Page rank':
                        medida = nx.pagerank(G)

                    df = pd.DataFrame(list(medida.items()), columns=['NodeId', centrality])

                    node_dict = nx.get_node_attributes(G, 'label')
                    df['Email'] = df['NodeId'].map(node_dict)

                    df = df.sort_values(by=centrality, ascending=False)

                    with col2:
                        chart = alt.Chart(df[:20].reset_index()).mark_bar().encode(
                            x=alt.X(f"{centrality}:Q", title=centrality),
                            y=alt.Y('Email:N', sort='-x', title=' ')
                        ).properties(
                            title='Destaques individuais'
                        ).configure_title(
                            offset=20
                        ).configure_axisY(
                            labelLimit=200
                        )

                        st.altair_chart(chart, use_container_width=True)

    ##############
            # cria o driver python
            driver = GraphDatabase.driver(
                NEO4J_URL, auth=(NEO4J_USER, NEO4J_PASSWORD))

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

            with driver.session() as session:

                with col3:

                    query_cg_3 = f"MATCH ()-[]->(c1:Guilda) RETURN DISTINCT c1.nome AS c1_nome"
                    rede_list_guilda = [record["c1_nome"]for record in session.run(query_cg_3)]
                    nome = st.selectbox("Selecione a guilda:", sorted(rede_list_guilda), key='nome_b')

                    query_dimensao = f"MATCH (n)-[r]->(g:Guilda) WHERE g.nome IN ['{nome}'] WITH collect(n) AS GuildaNodes UNWIND GuildaNodes as n1 UNWIND GuildaNodes as n2 MATCH (n1)-[r2]->(n2) WHERE id(n1) < id(n2) AND r2.dimensao <> 'None' RETURN DISTINCT r2.dimensao AS r2_dim"
                    dim_g = [record["r2_dim"]for record in session.run(query_dimensao)]
                    dim = st.selectbox("Selecione a dimensão", sorted(dim_g), key='dim_b')
                    
                    # adiciona os botões
                    e_list = [
                        (record["n1_email"], record["n2_email"])
                        for record in session.run(
                            f"""
                            MATCH (n)-[r]->(guilda:Guilda {{nome: $nome}})
                            WITH collect(n) AS GuildaNodes, guilda
                            UNWIND GuildaNodes as n1
                            UNWIND GuildaNodes as n2

                            MATCH (n1)-[r2 {{dimensao: $dimensao}}]->(n2)
                            RETURN DISTINCT n1.email AS n1_email, n2.email AS n2_email
                            """,
                            dimensao = dim,
                            nome=nome
                        )]
                    
                    email_list = {name for tuple_ in e_list for name in tuple_}
                    source = st.multiselect("Selecione um ou mais EuA3:", sorted(email_list), key ='source_b')
                    
                    records, summary = session.execute_read(
                                get_eua3, nome=nome, dimensao=dim)

                    G = nx.MultiDiGraph()

                    for record in records:
                        source_id = record["n1"].element_id
                        target_id = record["n2"].element_id
                        G.add_node(source_id, label=record["n1"]["email"])
                        G.add_node(target_id, label=record["n2"]["email"])
                        G.add_edge(source_id, target_id, label=record["r2"], rodada=record["r2"]["rodada"])

                    rodada_values = nx.get_edge_attributes(G, 'rodada')
                    graphs_by_rodada = defaultdict(nx.MultiDiGraph)

                    for (u, v, k), rodada in rodada_values.items():
                        if u not in graphs_by_rodada[rodada]:
                            graphs_by_rodada[rodada].add_node(u, **G.nodes[u])
                        if v not in graphs_by_rodada[rodada]:
                            graphs_by_rodada[rodada].add_node(v, **G.nodes[v])
                        
                        graphs_by_rodada[rodada].add_edge(u, v)

                    centrality = st.selectbox(
                        'Selecione uma medida de centralidade:', [
                            'InDegree', 'OutDegree', 'Betweenness', 'Closeness', 'Page rank'], key='abc')

                    centrality_measures_by_rodada = {}
                    for rodada, graph in graphs_by_rodada.items():
                        if centrality == 'InDegree':
                            medida = nx.in_degree_centrality(graph)
                        elif centrality == 'OutDegree':
                            medida = nx.out_degree_centrality(graph)
                        elif centrality == 'Betweenness':
                            medida = nx.betweenness_centrality(graph)
                        elif centrality == 'Closeness':
                            medida = nx.closeness_centrality(graph)
                        elif centrality == 'Page rank':
                            medida = nx.pagerank(graph)

                        medida_with_labels = {f"{node_id}|{graph.nodes[node_id]['label']}": value for node_id, value in medida.items()}
                        centrality_measures_by_rodada[rodada] = medida_with_labels
                    
                    records = []
                    for rodada, measures in centrality_measures_by_rodada.items():
                        for key, value in measures.items():
                            # Extract email part using string splitting
                            email = key.split('|')[-1]
                            if email in source:
                                records.append((email, value, rodada))

                    # Convert list to DataFrame
                    df = pd.DataFrame(records, columns=['label', 'values', 'rodada'])

                    if source: 
                        month_to_num = {'JAN': '01', 'FEV': '02', 'MAR': '03', 'ABR': '04', 'MAI': '05',
                                        'JUN': '06', 'JUL': '07', 'AGO': '08', 'SET': '09', 'OUT': '10', 'NOV': '11', 'DEZ': '12'}
                        df['date'] = df['rodada'].map(
                            lambda x: month_to_num[x[:3]] + x[3:])
                        df['date'] = pd.to_datetime(df['date'], format='%m%Y')
                        df = df.sort_values('date')

                        df = df.drop(columns=['date'])

                        with col4:
                            line_chart = alt.Chart(df).mark_line(point=True).encode(
                                alt.X('rodada:O', title='Rodada', sort=None),
                                alt.Y('values:Q', title=''),
                                alt.Color('label:N', title='EuA3')
                            ).properties(
                                title='Série temporal individual',
                                height=450
                            ).interactive()

                            st.altair_chart(line_chart, use_container_width=True)

################################
## Cliente ############################
##########################
    with tab3:

        col1, col2 = st.columns([1, 3])

        st.markdown("<br>"*2, unsafe_allow_html=True)

        col3, col4 = st.columns([1, 3])

        with col1:
            # cria o driver python
            driver = GraphDatabase.driver(
                NEO4J_URL, auth=(NEO4J_USER, NEO4J_PASSWORD))

            # função neo4j para importar a rede do localhost neo4j
            def get_eua3(session, nome, dimensao, rodada):
                query = f"""
                    MATCH (n)-[r:ALOCADO_{rodada}]->(cliente:Cliente {{nome: $nome}})
                    WITH collect(n) AS ClienteNodes, cliente
                    UNWIND ClienteNodes as n1
                    UNWIND ClienteNodes as n2

                    MATCH (n1)-[r2 {{rodada: $rodada, dimensao: $dimensao}}]->(n2)
                    RETURN DISTINCT n1, r2, n2
                """
                def read_tx(tx):
                    result = tx.run(query, nome=nome, rodada=rodada, dimensao=dimensao)
                    records = list(result)
                    summary = result.consume()
                    return records, summary

                records, summary = session.execute_read(read_tx)
                return records, summary
            
            with driver.session() as session:

                with col1:
               
                    query_rodada = f"MATCH (n)-[r]->(n2) WHERE r.rodada IS NOT NULL RETURN DISTINCT r.rodada AS r_rod"
                    rod_g = [record["r_rod"]for record in session.run(query_rodada)]
                    def to_date(s):    
                        months = ["JAN", "FEV", "MAR", "ABR", "MAI", "JUN", "JUL", "AGO", "SET", "OUT", "NOV", "DEZ"]
                        month, year = s[:3], s[3:]
                        return datetime(int(year), months.index(month) + 1, 1)
                    rod_g.sort(key=to_date, reverse=True)
                    rod = st.selectbox("Selecione a rodada", rod_g, key = "ind_2")
                    
                    query_dimensao = f"MATCH (n1)-[r]->(n2) WHERE r.dimensao IS NOT NULL RETURN DISTINCT r.dimensao AS r_dim"
                    dim_g = [record["r_dim"]for record in session.run(query_dimensao)]
                    dim = st.selectbox("Selecione a dimensão", sorted(dim_g), key = "ind_1")

                    query_cg_3 = f"MATCH ()-[:ALOCADO_{rod}]->(c:Cliente) RETURN DISTINCT c.nome AS g_nome"
                    rede_list_cliente = [record["g_nome"]for record in session.run(query_cg_3)]
                    nome = st.selectbox("Selecione a Guilda:", sorted(rede_list_cliente), key ="ind_3")

                    records, summary = get_eua3(session, nome=nome, dimensao=dim, rodada=rod)


                    G = nx.DiGraph()

                    for record in records:
                        source_id = record["n1"].element_id
                        target_id = record["n2"].element_id
                        G.add_node(source_id, label=record["n1"]["email"])
                        G.add_node(target_id, label=record["n2"]["email"])
                        G.add_edge(source_id, target_id, label=record["r2"])

                    centrality = st.selectbox(
                        'Selecione uma medida de centralidade:',
                        ['InDegree', 'OutDegree', 'Betweenness', 'Closeness', 'Page rank'], key='centrality abcd')

                    if centrality == 'InDegree':
                        medida = nx.in_degree_centrality(G)
                    elif centrality == 'OutDegree':
                        medida = nx.out_degree_centrality(G)
                    elif centrality == 'Betweenness':
                        medida = nx.betweenness_centrality(G)
                    elif centrality == 'Closeness':
                        medida = nx.closeness_centrality(G)
                    elif centrality == 'Page rank':
                        medida = nx.pagerank(G)

                    df = pd.DataFrame(list(medida.items()), columns=['NodeId', centrality])

                    node_dict = nx.get_node_attributes(G, 'label')
                    df['Email'] = df['NodeId'].map(node_dict)

                    df = df.sort_values(by=centrality, ascending=False)

                    with col2:
                        chart = alt.Chart(df[:20].reset_index()).mark_bar().encode(
                            x=alt.X(f"{centrality}:Q", title=centrality),
                            y=alt.Y('Email:N', sort='-x', title=' ')
                        ).properties(
                            title='Destaques individuais'
                        ).configure_title(
                            offset=20
                        ).configure_axisY(
                            labelLimit=200
                        )

                        st.altair_chart(chart, use_container_width=True)

    ##############
            # cria o driver python
            driver = GraphDatabase.driver(
                NEO4J_URL, auth=(NEO4J_USER, NEO4J_PASSWORD))

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

                with col3:

                    query_cg_3 = f"MATCH ()-[]->(c1:Cliente) RETURN DISTINCT c1.nome AS c1_nome"
                    rede_list_cliente = [record["c1_nome"]
                                        for record in session.run(query_cg_3)]
                    nome = st.selectbox(
                        "Selecione o cliente:", sorted(rede_list_cliente), key='nome_bc')

                    query_dimensao = f"MATCH (n)-[r]->(g:Cliente) WHERE g.nome IN ['{nome}'] WITH collect(n) AS ClienteNodes UNWIND ClienteNodes as n1 UNWIND ClienteNodes as n2 MATCH (n1)-[r2]->(n2) WHERE id(n1) < id(n2) AND r2.dimensao <> 'None' RETURN DISTINCT r2.dimensao AS r2_dim"
                    dim_g = [record["r2_dim"]
                                for record in session.run(query_dimensao)]
                    dim = st.selectbox("Selecione a dimensão", sorted(dim_g), key='dim_bc')
                    
                    # adiciona os botões
                    e_list = [
                        (record["n1_email"], record["n2_email"])
                        for record in session.run(
                            f"""
                            MATCH (n)-[r]->(cliente:Cliente {{nome: $nome}})
                            WITH collect(n) AS ClienteNodes, cliente
                            UNWIND ClienteNodes as n1
                            UNWIND ClienteNodes as n2

                            MATCH (n1)-[r2 {{dimensao: $dimensao}}]->(n2)
                            RETURN DISTINCT n1.email AS n1_email, n2.email AS n2_email
                            """,
                            dimensao = dim,
                            nome=nome
                        )]
                    
                    email_list = {name for tuple_ in e_list for name in tuple_}
                    source = st.multiselect("Selecione um ou mais EuA3:", sorted(email_list), key ='source_bc_papel')
                    
                    records, summary = session.execute_read(
                                get_eua3, nome=nome, dimensao=dim)

                    G = nx.MultiDiGraph()

                    for record in records:
                        source_id = record["n1"].element_id
                        target_id = record["n2"].element_id
                        G.add_node(source_id, label=record["n1"]["email"])
                        G.add_node(target_id, label=record["n2"]["email"])
                        G.add_edge(source_id, target_id, label=record["r2"], rodada=record["r2"]["rodada"])

                    rodada_values = nx.get_edge_attributes(G, 'rodada')
                    graphs_by_rodada = defaultdict(nx.MultiDiGraph)

                    for (u, v, k), rodada in rodada_values.items():
                        if u not in graphs_by_rodada[rodada]:
                            graphs_by_rodada[rodada].add_node(u, **G.nodes[u])
                        if v not in graphs_by_rodada[rodada]:
                            graphs_by_rodada[rodada].add_node(v, **G.nodes[v])
                        
                        graphs_by_rodada[rodada].add_edge(u, v)

                    centrality = st.selectbox(
                        'Selecione uma medida de centralidade:', [
                            'InDegree', 'OutDegree', 'Betweenness', 'Closeness', 'Page rank'], key='abcd')

                    centrality_measures_by_rodada = {}
                    for rodada, graph in graphs_by_rodada.items():
                        if centrality == 'InDegree':
                            medida = nx.in_degree_centrality(graph)
                        elif centrality == 'OutDegree':
                            medida = nx.out_degree_centrality(graph)
                        elif centrality == 'Betweenness':
                            medida = nx.betweenness_centrality(graph)
                        elif centrality == 'Closeness':
                            medida = nx.closeness_centrality(graph)
                        elif centrality == 'Page rank':
                            medida = nx.pagerank(graph)

                        medida_with_labels = {f"{node_id}|{graph.nodes[node_id]['label']}": value for node_id, value in medida.items()}
                        centrality_measures_by_rodada[rodada] = medida_with_labels
                    
                    records = []
                    for rodada, measures in centrality_measures_by_rodada.items():
                        for key, value in measures.items():
                            # Extract email part using string splitting
                            email = key.split('|')[-1]
                            if email in source:
                                records.append((email, value, rodada))

                    # Convert list to DataFrame
                    df = pd.DataFrame(records, columns=['label', 'values', 'rodada'])

                    if source: 
                        month_to_num = {'JAN': '01', 'FEV': '02', 'MAR': '03', 'ABR': '04', 'MAI': '05',
                                        'JUN': '06', 'JUL': '07', 'AGO': '08', 'SET': '09', 'OUT': '10', 'NOV': '11', 'DEZ': '12'}
                        df['date'] = df['rodada'].map(
                            lambda x: month_to_num[x[:3]] + x[3:])
                        df['date'] = pd.to_datetime(df['date'], format='%m%Y')
                        df = df.sort_values('date')

                        df = df.drop(columns=['date'])

                        with col4:
                            line_chart = alt.Chart(df).mark_line(point=True).encode(
                                alt.X('rodada:O', title='Rodada', sort=None),
                                alt.Y('values:Q', title=''),
                                alt.Color('label:N', title='EuA3')
                            ).properties(
                                title='Série temporal individual',
                                height=450
                            ).interactive()

                            st.altair_chart(line_chart, use_container_width=True)

