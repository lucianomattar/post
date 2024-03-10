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

        with col1:

            driver = GraphDatabase.driver(
                NEO4J_URL, auth=(NEO4J_USER, NEO4J_PASSWORD))

            # função neo4j para importar a rede do localhost neo4j
            def get_eua3(tx, dimensao, rodada):
                result = tx.run("""
                    MATCH (c1)-[b1 {dimensao: $dimensao, rodada: $rodada}]->(c2)
                    RETURN c1, b1, c2
                """,
                                rodada=rodada,
                                dimensao=dimensao)

                records = list(result)
                summary = result.consume()

                return records, summary

            with driver.session() as session:

                rodada_list = [record["b1_rodada"] for record in session.run(
                    f"MATCH ()-[b1:BUSCOU]->() RETURN DISTINCT b1.rodada AS b1_rodada")]
                def to_date(s):    
                    months = ["JAN", "FEV", "MAR", "ABR", "MAI", "JUN", "JUL", "AGO", "SET", "OUT", "NOV", "DEZ"]
                    month, year = s[:3], s[3:]
                    return datetime(int(year), months.index(month) + 1, 1)
                rodada_list.sort(key=to_date, reverse=True)
                rod = st.selectbox("Selecione a rodada:", rodada_list)
                     
                dimension_list = [record["b1_dimensao"] for record in session.run(
                    f"MATCH ()-[b1:BUSCOU]->() RETURN DISTINCT b1.dimensao AS b1_dimensao")]
                dim = st.selectbox("Selecione a dimensão:", sorted(dimension_list))
                
                records, summary = session.execute_read(get_eua3, dimensao=dim, rodada=rod)

            G = nx.DiGraph()

            for record in records:
                # 'record', record
                source_id = record["c1"].element_id
                target_id = record["c2"].element_id
                G.add_node(source_id, label=record["c1"]["email"])
                G.add_node(target_id, label=record["c2"]["email"])
                G.add_edge(source_id, target_id,label=record["b1"])

            centrality_a = st.selectbox(
                        'Selecione uma medida de centralidade para o eixo x:',
                        ['InDegree', 'OutDegree', 'Betweenness', 'Closeness', 'Page rank'], key='primeiro')

            if centrality_a == 'InDegree':
                medida_a = nx.in_degree_centrality(G)
            elif centrality_a == 'OutDegree':
                medida_a = nx.out_degree_centrality(G)
            elif centrality_a == 'Betweenness':
                medida_a = nx.betweenness_centrality(G)
            elif centrality_a == 'Closeness':
                A = nx.adjacency_matrix(G)
                nodes = A.shape[0]
                node_order = list(G.nodes())

                D = scipy.sparse.csgraph.floyd_warshall(
                    A, directed=False, unweighted=False)
                medida_a = {}
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
                    medida_a[node_order[r]] = cc

            elif centrality_a == 'Page rank':
                medida_a = nx.pagerank(G)

            centrality_b = st.selectbox(
                        'Selecione uma medida de centralidade para o eixo y:',
                        ['InDegree', 'OutDegree', 'Betweenness', 'Closeness', 'Page rank'], key='segundo')

            if centrality_b == 'InDegree':
                medida_b = nx.in_degree_centrality(G)
            elif centrality_b == 'OutDegree':
                medida_b = nx.out_degree_centrality(G)
            elif centrality_b == 'Betweenness':
                medida_b = nx.betweenness_centrality(G)
            elif centrality_b == 'Closeness':
                A = nx.adjacency_matrix(G)
                nodes = A.shape[0]
                node_order = list(G.nodes())

                D = scipy.sparse.csgraph.floyd_warshall(
                    A, directed=False, unweighted=False)
                medida_b = {}
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
                    medida_b[node_order[r]] = cc

            elif centrality_b == 'Page rank':
                medida_b = nx.pagerank(G)

            #numero de nodos
            st.write('\n')
            st.write('\n')
            num_nodes = G.number_of_nodes()
            st.write("Número de EuA3:", str(num_nodes))

            email_dict = nx.get_node_attributes(G, 'label')
            df_email = pd.DataFrame.from_dict(email_dict, orient='index', columns=['email']).reset_index()
            df_a = pd.DataFrame.from_dict(medida_a, orient='index', columns=[f"{centrality_a}_x"]).reset_index()
            df_a = df_a.merge(df_email, how='left', left_on='index', right_on='index')

            df_email = pd.DataFrame.from_dict(email_dict, orient='index', columns=['email']).reset_index()
            df_b = pd.DataFrame.from_dict(medida_b, orient='index', columns=[f"{centrality_b}_y"]).reset_index()
            df_b = df_b.merge(df_email, how='left', left_on='index', right_on='index')

            df = pd.merge(df_a, df_b, on='index')
            
            scatter_plot = alt.Chart(df).mark_circle(size=200).encode(
                x=alt.X(f"{centrality_a}_x:Q", title=f"{centrality_a}_x", scale=alt.Scale(padding=1)),
                y=alt.Y(f"{centrality_b}_y:Q", title=f"{centrality_b}_y", scale=alt.Scale(padding=1)),
                tooltip=[f"{centrality_a}_x:Q", f"{centrality_b}_y:Q", 'email_x']
            ).properties(
                title='Cruzamento entre métricas de centralidade',
                height=700
            )

            #corr
            corl = df.iloc[:, [1, 3]].corr()
            corl_f = corl.iloc[1, 0]

            corl_df = pd.DataFrame({
                'corl': ['Correlação: ' + str(round(corl_f, 2))],        
                'x': [df[f"{centrality_a}_x"].min()],
                'y': [df[f"{centrality_b}_y"].max()]
            })

            text = alt.Chart(corl_df).mark_text(
                align='left',
                baseline='top',
                dx=5, 
                fontSize=20  
            ).encode(
                x=alt.value(5),
                y=alt.value(5),
                text='corl'
            )
            
            # linha
            reg = scatter_plot.transform_regression(f"{centrality_a}_x", f"{centrality_b}_y").mark_line(color="red")

            char_sp = (scatter_plot + text + reg).interactive()

            with col2:
                st.altair_chart(char_sp, use_container_width=True)

    ###################
    #GUILDA
    ###################
    with tab2:

        col1, col2 = st.columns([1, 3])

        with col1:

            # cria o driver python
            driver = GraphDatabase.driver(
                NEO4J_URL, auth=(NEO4J_USER, NEO4J_PASSWORD))

            def get_eua3(session, nome, dimensao, rodada):
                query = f"""
                    MATCH (n)-[r:PERTENCE_{rodada}]->(guilda:Guilda {{nome: $nome}})
                    WITH collect(n) AS GuildaNodes, guilda
                    UNWIND GuildaNodes as n1
                    UNWIND GuildaNodes as n2

                    MATCH (n1)-[r2 {{rodada: $rodada, dimensao: $dimensao}}]->(n2)
                    RETURN DISTINCT n1, r2, n2, r2.rodada AS rodada
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

                    centrality_a = st.selectbox(
                        'Selecione uma medida de centralidade para o eixo x:',
                        ['InDegree', 'OutDegree', 'Betweenness', 'Closeness', 'Page rank'], key='centrality terceiro')

                    if centrality_a == 'InDegree':
                        medida_a = nx.in_degree_centrality(G)
                    elif centrality_a == 'OutDegree':
                        medida_a = nx.out_degree_centrality(G)
                    elif centrality_a == 'Betweenness':
                        medida_a = nx.betweenness_centrality(G)
                    elif centrality_a == 'Closeness':
                        A = nx.adjacency_matrix(G)
                        nodes = A.shape[0]
                        node_order = list(G.nodes())

                        D = scipy.sparse.csgraph.floyd_warshall(
                            A, directed=False, unweighted=False)
                        medida_a = {}
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
                            medida_a[node_order[r]] = cc

                    elif centrality_a == 'Page rank':
                        medida_a = nx.pagerank(G)

                    centrality_b = st.selectbox(
                                'Selecione uma medida de centralidade para o eixo y:',
                                ['InDegree', 'OutDegree', 'Betweenness', 'Closeness', 'Page rank'], key='centrality segundo')

                    if centrality_b == 'InDegree':
                        medida_b = nx.in_degree_centrality(G)
                    elif centrality_b == 'OutDegree':
                        medida_b = nx.out_degree_centrality(G)
                    elif centrality_b == 'Betweenness':
                        medida_b = nx.betweenness_centrality(G)
                    elif centrality_b == 'Closeness':
                        A = nx.adjacency_matrix(G)
                        nodes = A.shape[0]
                        node_order = list(G.nodes())

                        D = scipy.sparse.csgraph.floyd_warshall(
                            A, directed=False, unweighted=False)
                        medida_b = {}
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
                            medida_b[node_order[r]] = cc

                    elif centrality_b == 'Page rank':
                        medida_b = nx.pagerank(G)

                    #numero de nodos
                    st.write('\n')
                    st.write('\n')
                    num_nodes = G.number_of_nodes()
                    st.write("Número de EuA3:", str(num_nodes))

                    email_dict = nx.get_node_attributes(G, 'label') 
                    df_email = pd.DataFrame.from_dict(email_dict, orient='index', columns=['email']).reset_index()
                    df_a = pd.DataFrame.from_dict(medida_a, orient='index', columns=[f"{centrality_a}_x"]).reset_index()
                    df_a = df_a.merge(df_email, how='left', left_on='index', right_on='index')  

                    df_email = pd.DataFrame.from_dict(email_dict, orient='index', columns=['email']).reset_index()
                    df_b = pd.DataFrame.from_dict(medida_b, orient='index', columns=[f"{centrality_b}_y"]).reset_index()
                    df_b = df_b.merge(df_email, how='left', left_on='index', right_on='index')

                    df = pd.merge(df_a, df_b, on='index')

                    if len(G) != 0:

                        scatter_plot = alt.Chart(df).mark_circle(size=200).encode(
                            x=alt.X(f"{centrality_a}_x:Q", title=f"{centrality_a}_x", scale=alt.Scale(padding=1)),
                            y=alt.Y(f"{centrality_b}_y:Q", title=f"{centrality_b}_y", scale=alt.Scale(padding=1)),
                            tooltip=[f"{centrality_a}_x:Q", f"{centrality_b}_y:Q", 'email_x']
                        ).properties(
                            title='Cruzamento entre métricas de centralidade',
                            height=700
                        )

                        #corr
                        corl = df.iloc[:, [1, 3]].corr()
                        corl_f = corl.iloc[0, 1]

                        corl_df = pd.DataFrame({
                            'corl': ['Correlação: ' + str(round(corl_f, 2))],        
                            'x': [df[f"{centrality_a}_x"].min()],
                            'y': [df[f"{centrality_b}_y"].max()]
                        })

                        text = alt.Chart(corl_df).mark_text(
                            align='left',
                            baseline='top',
                            dx=5, 
                            fontSize=20  
                        ).encode(
                            x=alt.value(5),
                            y=alt.value(5),
                            text='corl'
                        )
                        
                        # linha
                        reg = scatter_plot.transform_regression(f"{centrality_a}_x", f"{centrality_b}_y").mark_line(color="red")

                        char_sp = (scatter_plot + text + reg).interactive()

                        with col2:
                            st.altair_chart(char_sp, use_container_width=True)
                    else:
                        pass

    ########################
    # CLIENTE
    ########################
    with tab3:
        col1, col2 = st.columns([1, 3])

        with col1:

            # cria o driver python
            driver = GraphDatabase.driver(
                NEO4J_URL, auth=(NEO4J_USER, NEO4J_PASSWORD))

            def get_eua3(session, nome, dimensao, rodada):
                query = f"""
                    MATCH (n)-[r:ALOCADO_{rodada}]->(cliente:Cliente {{nome: $nome}})
                    WITH collect(n) AS ClienteNodes, cliente
                    UNWIND ClienteNodes as n1
                    UNWIND ClienteNodes as n2

                    MATCH (n1)-[r2 {{rodada: $rodada, dimensao: $dimensao}}]->(n2)
                    RETURN DISTINCT n1, r2, n2, r2.rodada AS rodada
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

                    centrality_a = st.selectbox(
                        'Selecione uma medida de centralidade para o eixo x:',
                        ['InDegree', 'OutDegree', 'Betweenness', 'Closeness', 'Page rank'], key='centrality terceiro b')

                    if centrality_a == 'InDegree':
                        medida_a = nx.in_degree_centrality(G)
                    elif centrality_a == 'OutDegree':
                        medida_a = nx.out_degree_centrality(G)
                    elif centrality_a == 'Betweenness':
                        medida_a = nx.betweenness_centrality(G)
                    elif centrality_a == 'Closeness':
                        A = nx.adjacency_matrix(G)
                        nodes = A.shape[0]
                        node_order = list(G.nodes())

                        D = scipy.sparse.csgraph.floyd_warshall(
                            A, directed=False, unweighted=False)
                        medida_a = {}
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
                            medida_a[node_order[r]] = cc

                    elif centrality_a == 'Page rank':
                        medida_a = nx.pagerank(G)

                    centrality_b = st.selectbox(
                                'Selecione uma medida de centralidade para o eixo y:',
                                ['InDegree', 'OutDegree', 'Betweenness', 'Closeness', 'Page rank'], key='centrality segundo b')

                    if centrality_b == 'InDegree':
                        medida_b = nx.in_degree_centrality(G)
                    elif centrality_b == 'OutDegree':
                        medida_b = nx.out_degree_centrality(G)
                    elif centrality_b == 'Betweenness':
                        medida_b = nx.betweenness_centrality(G)
                    elif centrality_b == 'Closeness':
                        A = nx.adjacency_matrix(G)
                        nodes = A.shape[0]
                        node_order = list(G.nodes())

                        D = scipy.sparse.csgraph.floyd_warshall(
                            A, directed=False, unweighted=False)
                        medida_b = {}
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
                            medida_b[node_order[r]] = cc

                    elif centrality_b == 'Page rank':
                        medida_b = nx.pagerank(G)

                    #numero de nodos
                    st.write('\n')
                    st.write('\n')
                    num_nodes = G.number_of_nodes()
                    st.write("Número de EuA3:", str(num_nodes))

                    email_dict = nx.get_node_attributes(G, 'label') 
                    df_email = pd.DataFrame.from_dict(email_dict, orient='index', columns=['email']).reset_index()
                    df_a = pd.DataFrame.from_dict(medida_a, orient='index', columns=[f"{centrality_a}_x"]).reset_index()
                    df_a = df_a.merge(df_email, how='left', left_on='index', right_on='index')  

                    df_email = pd.DataFrame.from_dict(email_dict, orient='index', columns=['email']).reset_index()
                    df_b = pd.DataFrame.from_dict(medida_b, orient='index', columns=[f"{centrality_b}_y"]).reset_index()
                    df_b = df_b.merge(df_email, how='left', left_on='index', right_on='index')

                    df = pd.merge(df_a, df_b, on='index')

                    if len(G) != 0:

                        scatter_plot = alt.Chart(df).mark_circle(size=200).encode(
                            x=alt.X(f"{centrality_a}_x:Q", title=f"{centrality_a}_x", scale=alt.Scale(padding=1)),
                            y=alt.Y(f"{centrality_b}_y:Q", title=f"{centrality_b}_y", scale=alt.Scale(padding=1)),
                            tooltip=[f"{centrality_a}_x:Q", f"{centrality_b}_y:Q", 'email_x']
                        ).properties(
                            title='Cruzamento entre métricas de centralidade',
                            height=700
                        )

                        #corr
                        corl = df.iloc[:, [1, 3]].corr()
                        corl_f = corl.iloc[0, 1]

                        corl_df = pd.DataFrame({
                            'corl': ['Correlação: ' + str(round(corl_f, 2))],        
                            'x': [df[f"{centrality_a}_x"].min()],
                            'y': [df[f"{centrality_b}_y"].max()]
                        })

                        text = alt.Chart(corl_df).mark_text(
                            align='left',
                            baseline='top',
                            dx=5, 
                            fontSize=20  
                        ).encode(
                            x=alt.value(5),
                            y=alt.value(5),
                            text='corl'
                        )
                        
                        # linha
                        reg = scatter_plot.transform_regression(f"{centrality_a}_x", f"{centrality_b}_y").mark_line(color="red")

                        char_sp = (scatter_plot + text + reg).interactive()

                        with col2:
                            st.altair_chart(char_sp, use_container_width=True)
                    else:
                        pass
