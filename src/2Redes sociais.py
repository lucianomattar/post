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
from streamlit_extras.app_logo import add_logo
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

        # cria o driver python
        driver = GraphDatabase.driver(
            NEO4J_URL, auth=(NEO4J_USER, NEO4J_PASSWORD))

        # função neo4j para importar a rede do localhost neo4j
        def perform_eua3(session, dimensao, rodada, email):
            
            main_query = f"""
                MATCH (c1)-[b1 {{rodada: $rodada, dimensao: $dimensao}}]->(c2)
                WHERE c1.email IN $email OR c2.email IN $email
                RETURN c1, b1, c2, id(c1) AS c1_id, id(c2) AS c2_id
            """

            def read_tx(tx):
                result = tx.run(main_query, email=email, dimensao=dimensao, rodada=rodada)
                records = list(result)
                summary = result.consume()
                return records, summary

            records, summary = session.execute_read(read_tx)
            
            # Extrair node ids do nós da rede
            node_ids = node_ids = [record['c1_id'] for record in records] + [record['c2_id'] for record in records]
            unique_node_ids = list(set(node_ids))

            # cria papel como propriedade
            update_query = f"""
                MATCH (c:Colaborador)-[:ATUA_COMO_{rodada}]->(p:Papel)
                WHERE id(c) IN $node_ids
                SET c.papel = p.nome
            """
            
            session.execute_write(lambda tx: tx.run(update_query, node_ids=node_ids))

            return records, summary

        # título
        title = st.title(f"Rede EuA3:")
        st.markdown('<style>h1{font-size: 24px;}</style>',
                    unsafe_allow_html=True)

        # Inicia a sessão do driver
        with driver.session() as session:

            col1, col31, col2, col3 = st.columns(4)

            # caixas seleção dimensao e rodada
            with col2:
                rodada1_list = [record["b1_rodada"] for record in session.run(
                    "MATCH ()-[b1:BUSCOU]->() RETURN DISTINCT b1.rodada AS b1_rodada")]

                def to_date(s):    
                    months = ["JAN", "FEV", "MAR", "ABR", "MAI", "JUN", "JUL", "AGO", "SET", "OUT", "NOV", "DEZ"]
                    month, year = s[:3], s[3:]
                    return datetime(int(year), months.index(month) + 1, 1)
                rodada1_list.sort(key=to_date, reverse=True)
                
                rod = st.selectbox("Selecione a rodada:", rodada1_list)

            with col3:
                dimension_list = [record["b1_dimensao"] for record in session.run(
                    f"MATCH ()-[b1:BUSCOU {{rodada: '{rod}'}}]->() RETURN DISTINCT b1.dimensao AS b1_dimensao")]
                dim = st.selectbox("Selecione a dimensão:",
                                    sorted(dimension_list))

            # adiciona os botões
            e_list = [(record["c1_email"], record["c2_email"]) for record in session.run(  # pega todos os nomes em todas as relações existentes para cada dimensao e rodada
                "MATCH (c1)-[b1 {rodada: $rodada, dimensao: $dimensao}]->(c2) RETURN DISTINCT c1.email AS c1_email, c2.email AS c2_email",
                dimensao=dim,
                rodada=rod)]
            email_list = {name for tuple_ in e_list for name in tuple_}
            email_list = list(email_list)

            # caixa de multiseleção rede total
            with col1:
                source = st.multiselect(
                    "Selecione um ou mais EuA3:", sorted(email_list))
                source_ind = source

            with col31:
                st.write('\n')
                st.write('\n')
                all_options_eua3 = st.checkbox("Selecione todos os EuA3")
                if all_options_eua3:
                    source = email_list

            # executa a query apos construida todas as variaveis de entrada
            records, summary = perform_eua3(session, dimensao=dim, rodada=rod, email=source)

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

            records_ind, summary = perform_eua3(session, dimensao=dim, rodada=rod, email=source)

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

            # selecionar guilda e cliente para colorir como caracteristicas o nós
            col51, col52, col54, col55 = st.columns(4)

            query_cg = None
            selected_option = None
            selected_property = None

            with col51:
                selected_atr_cg = st.selectbox('Cor de acordo com uma característica do EuA3:', [
                    'Caracteríticas não selecionadas', 'Atributos dos Colaboradores', 'Clientes e Guildas'])

            # caixa cliente guilda e none
            with col52:
                if selected_atr_cg == 'Atributos dos Colaboradores':
                    selected_property = st.selectbox('Atributos dos Colaboradores:', [
                                                        'Papel', 'Gênero', 'Tempo de casa'])
                    if selected_property == 'Papel':
                        selected_property = 'papel'
                    elif selected_property == 'Gênero':
                        selected_property = 'genero'
                    else:
                        selected_property = 'tempoDeCasa'

                if selected_atr_cg == 'Clientes e Guildas':
                    selected_option = st.selectbox('Clientes e Guildas:', [
                        'Cliente', 'Guilda'])
                    if selected_option == 'Cliente':
                        query_cg = f"MATCH (c:Colaborador)-[:ALOCADO_{rod}]->(c1:Cliente) WHERE c.email IN {email_unicos} RETURN DISTINCT c1.nome AS c1_nome"
                    else:
                        query_cg = f"MATCH (c:Colaborador)-[r:PERTENCE_{rod}]->(c1:Guilda) WHERE c.email IN {email_unicos} RETURN DISTINCT c1.nome AS c1_nome"

            # habilita caixa cliente guilda individual
            with col54:
                if selected_atr_cg == 'Clientes e Guildas':
                    rede_list = [record["c1_nome"]
                                    for record in session.run(query_cg)]

                    if selected_option == 'Cliente':
                        tipo_ind = st.selectbox(
                            "Selecione um cliente:", sorted(rede_list))
                    else:
                        tipo_ind = st.selectbox(
                            "Selecione a guilda:", sorted(rede_list))
                else:
                    st.empty()

                query = f"MATCH (e)-[r:PERTENCE_{rod}]->(g1) RETURN DISTINCT e.email AS email, g1.nome AS g1_nome"
                guilda = [(record["email"], record['g1_nome'])
                            for record in session.run(query)]
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

            # cria lista para usar para criar propriedades de guilda
            modified_list_g = []
            for sublist in guilda:
                modified_dict = {
                    "email": sublist[0],
                    "cliente_guilda": sublist[1]
                }
                modified_list_g.append(modified_dict)

            # selecione todas as guildas e clientes
            all_options_gc = None
            with col55:
                if query_cg is not None:
                    if selected_option == 'Cliente':
                        st.write('\n')
                        st.write('\n')
                        all_options_gc = st.checkbox(
                            "Selecione todos os clientes principais")
                        if all_options_gc:
                            source_c = rede_list

                    else:
                        st.write('\n')
                        st.write('\n')
                        all_options_gc = st.checkbox(
                            "Selecione todas as guildas principais")
                        if all_options_gc:
                            source_g = rede_list
                else:
                    st.empty()

                query = f"MATCH (n1:Colaborador)-[:PERTENCE_{rod}]->(g) RETURN n1.email AS email, COLLECT(g.nome) AS todas_guildas"
                guilda_t = [(record["email"], record['todas_guildas'])
                            for record in session.run(query)]

                # guilda_t[0][0]
                # guilda_t[1][1][0]
                query = f"MATCH (n1:Colaborador)-[:ALOCADO_{rod}]->(g) RETURN n1.email AS email, COLLECT(g.nome) AS todos_clientes"
                cliente_t = [(record["email"], record['todos_clientes'])
                                for record in session.run(query)]

            guilda_t_d = {}
            # Loop sobre cada tupla nos dados
            for i in range(len(guilda_t)):
                # o nome é o primeiro item da tupla
                name = guilda_t[i][0]
                # A posição é o primeiro item da lista dentro da tupla
                position = guilda_t[i][1][0]
                # Add ao dicionário
                guilda_t_d[name] = position

            new_guilda = [{"email": k, "guilda": v}
                            for i, (k, v) in enumerate(guilda_t_d.items())]

            cliente_t_c = {}
            for i in range(len(cliente_t)):
                name = cliente_t[i][0]
                position = cliente_t[i][1][0]
                cliente_t_c[name] = position

            new_cliente = [{"email": k, "cliente": v}
                            for i, (k, v) in enumerate(cliente_t_c.items())]

            # 'new_cliente', new_cliente
            # print(type(new_cliente))
            # new_cliente[0]['cliente']

            # Cria o objeto de grafo direcionado do networkx
            G = nx.DiGraph()

            # pega todas as caracteristicas laços e nós de 'cada vez'
            # adiciona nós e laços a partir dos resultados do query da rede neo4j
            for record in records:
                source_id = record["c1"].element_id
                target_id = record["c2"].element_id
                G.add_node(source_id, label=record["c1"]["email"], genero=record["c1"]["genero"], papel=record["c1"]
                            ["papel"], tempoDeCasa=record["c1"]["tempoDeCasa"], idade=record["c1"]["idade"])
                G.add_node(target_id, label=record["c2"]["email"], genero=record["c2"]["genero"], papel=record["c2"]
                            ["papel"], tempoDeCasa=record["c2"]["tempoDeCasa"], idade=record["c2"]["idade"])
                G.add_edge(source_id, target_id, label=record["b1"])

            # for node in G.nodes(data=True):
            #     'node', node

            # define o tipo de visualização do networkx
            pos = nx.fruchterman_reingold_layout(G)

            # exclui edge label '= None' que vem do neo4j para não ser visualizado no networkx para melhorar a vizualização geral do grafo
            nx.set_edge_attributes(G, None, "label")
            nx.draw_networkx_nodes(G, pos)
            nx.draw_networkx_edges(G, pos)
            nx.draw_networkx_labels(G, pos)

        # Define a caixa com as medidas de centralidade
        centrality_measures = {
            'InDegree': nx.in_degree_centrality(G),
            'OutDegree': nx.out_degree_centrality(G),
            'Betweenness': nx.betweenness_centrality(G),
            'Closeness': nx.closeness_centrality(G),
            'Page Rank': nx.pagerank(G)
        }

        # Cria a caixa lateral das medidas de centralidade
        col01, col02, col03, col04 = st.columns(4)

        with col01:
            centrality = st.selectbox(
                'Selecione uma medida de centralização:', list(centrality_measures.keys()))

        # visualizacao
        # Calcula as medidas descritiva

        # número de laços
        num_ties = G.number_of_edges()

        # número de nós
        num_nodes = G.number_of_nodes()

        # densidade
        density = nx.density(G)
    
        # calcula o diametro do maior componente
        def effective_diameter(Gu):
            subG = Gu.subgraph([node for node, degree in Gu.degree() if degree > 0])
            if not nx.is_connected(subG):
                diameters = [nx.diameter(subG.subgraph(c)) for c in nx.connected_components(subG)]
                return max(diameters)
            else:
                return nx.diameter(subG)
        if all(degree == 0 for _, degree in G.degree()):
            diameter = 0
        else:
            Gu = nx.to_undirected(G)
            diameter = effective_diameter(Gu) 

        # degree médio
        if G.number_of_nodes() > 0:
            average_degree = num_ties/num_nodes
        else:
            average_degree = 0
            

        # título
        title.title(
            f"Rede EuA3: Dimensão {dim} em {rod} - Medida de Centralidade: {centrality}")
        st.markdown('<style>h1{font-size: 24px;}</style>',
                    unsafe_allow_html=True)

        st.write('\n')

        # Pega as medidas de centralização do networkx
        nx.set_node_attributes(G, 10, 'size')
        nc = nx.get_node_attributes(G, 'size')

        ns = centrality_measures[centrality]
        ns = {node: ns[node]*50 + 10 for node in G.nodes()}
        nx.set_node_attributes(G, ns, 'size')
        nc = nx.get_node_attributes(G, 'size')

        # colorir os nós mais centrais
        sorted_nodes = sorted(nc, key=nc.get, reverse=True)
        # Cria um botão para selecionar o numéro de nós mais centrais
        num_highlight = None

        with col02:
            if selected_atr_cg == 'Caracteríticas não selecionadas':
                num_highlight = st.selectbox("Escolha a quantidade de EuA3 mais centrais", [
                    i for i in range(0, len(sorted_nodes) + 1)])

        # Defini as cores do nós da rede e pela centralidade
        # Defini as cores do nós da rede e pela centralidade
        #if  len(source_ind) != 0:

        node_color_dict = {}
        for node in G.nodes():
            # rede apos a escolha de medidas de centralidade
            if node in sorted_nodes[:num_highlight]:
                node_color_dict[node] = '#dd4b39'  # cor mais centrais
            else:
                node_color_dict[node] = '#1779e1'  # cor menos centrais
        nx.set_node_attributes(G, node_color_dict, 'color')

        for node in G.nodes():
            # checar se os labels não estão no 'email_unicos_ind' list
            if G.nodes[node]['label'] not in email_unicos_ind:
                G.nodes[node]['color'] = 'lightgray'


        # cor propriedades
        # cor genero
        node_color_dict = {}
        if selected_property == 'genero':
            for node in G.nodes():
                if G.nodes[node]['genero'] == 'f':
                    node_color_dict[node] = '#00CC00'
                elif G.nodes[node]['genero'] == 'm':
                    node_color_dict[node] = '#d3d020e8'
                elif G.nodes[node]['genero'] == 'feminino':
                    node_color_dict[node] = '#00CC00'
                elif G.nodes[node]['genero'] == 'masculino':
                    node_color_dict[node] = '#d3d020e8'
                elif G.nodes[node]['genero'] == 'Unknown':
                    node_color_dict[node] = '#000000'
                else:
                    pass
        nx.set_node_attributes(G, node_color_dict, 'color')

        node_color_dict = {}
        if all_options_eua3:
            if len(source_ind) != 0:
                if selected_property == 'genero':
                    for node in G.nodes():
                        if G.nodes[node]['genero'] == 'f':
                            node_color_dict[node] = '#00CC00'
                        elif G.nodes[node]['genero'] == 'm':
                            node_color_dict[node] = '#d3d020e8'
                        elif G.nodes[node]['genero'] == 'feminino':
                            node_color_dict[node] = '#00CC00'
                        elif G.nodes[node]['genero'] == 'masculino':
                            node_color_dict[node] = '#d3d020e8'
                        elif G.nodes[node]['genero'] == 'Unknown':
                            node_color_dict[node] = '#000000'
                        else:
                            pass
                nx.set_node_attributes(G, node_color_dict, 'color')
                for node in G.nodes():
                    # checar se os labels não estão no 'email_unicos_ind' list
                    if G.nodes[node]['label'] not in email_unicos_ind:
                        G.nodes[node]['color'] = 'lightgray'

        # cor tempo de casa
        node_color_dict = {}
        if selected_property == 'tempoDeCasa':
            for node in G.nodes():
                if G.nodes[node]['tempoDeCasa'] == 'De 1 a 6 meses':
                    node_color_dict[node] = '#ca98ec'
                elif G.nodes[node]['tempoDeCasa'] == 'De 6 meses a 1 ano':
                    node_color_dict[node] = '#ca9800'
                elif G.nodes[node]['tempoDeCasa'] == 'Recém ingressante':
                    node_color_dict[node] = '#097c20'
                elif G.nodes[node]['tempoDeCasa'] == 'Mais que um ano':
                    node_color_dict[node] = '#ca0000'
                elif G.nodes[node]['tempoDeCasa'] == 'Unknown':
                    node_color_dict[node] = '#000000'
                else:
                    pass
        nx.set_node_attributes(G, node_color_dict, 'color')

        node_color_dict = {}
        if all_options_eua3:
            if len(source_ind) != 0:
                if selected_property == 'tempoDeCasa':
                    for node in G.nodes():
                        if G.nodes[node]['tempoDeCasa'] == 'De 1 a 6 meses':
                            node_color_dict[node] = '#ca98ec'
                        elif G.nodes[node]['tempoDeCasa'] == 'De 6 meses a 1 ano':
                            node_color_dict[node] = '#ca9800'
                        elif G.nodes[node]['tempoDeCasa'] == 'Recém ingressante':
                            node_color_dict[node] = '#097c20'
                        elif G.nodes[node]['tempoDeCasa'] == 'Mais que um ano':
                            node_color_dict[node] = '#ca0000'
                        elif G.nodes[node]['tempoDeCasa'] == 'Unknown':
                            node_color_dict[node] = '#000000'
                        else:
                            pass
                nx.set_node_attributes(G, node_color_dict, 'color')
                for node in G.nodes():
                    # checar se os labels não estão no 'email_unicos_ind' list
                    if G.nodes[node]['label'] not in email_unicos_ind:
                        G.nodes[node]['color'] = 'lightgray'

        # cor papel
        color_mapping = {}
        if selected_property == 'papel':
            unique_values = list(set(node['papel']
                                    for node in G.nodes().values()))
            num_unique_values = len(unique_values)

            # 50 cores para funçoes
            distinct_colors = [
                (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
                (128, 0, 128), (0, 128, 128), (128, 128, 128), (255, 128, 0), (255, 0, 128), (0, 255, 128), (128, 0, 255), (0, 128, 255),
                (255, 128, 128), (128, 255, 128), (128, 128, 255), (255, 255, 128), (255, 128, 255), (128, 255, 255), (192, 0, 0), (0, 192, 0),
                (0, 0, 192), (192, 192, 0), (192, 0, 192), (0, 192, 192), (192, 192, 192), (255, 192, 0), (255, 0, 192), (0, 255, 192),
                (192, 0, 255), (0, 192, 255), (255, 192, 192), (192, 255, 192), (192, 192, 255), (255, 255, 192), (255, 192, 255), (192, 255, 255),
                (64, 0, 0), (0, 64, 0), (0, 0, 64), (64, 64, 0), (64, 0, 64), (0, 64, 64), (64, 64, 64), (128, 64, 0),
                (128, 0, 64), (0, 128, 64), (64, 0, 128), (0, 64, 128), (128, 64, 64), (64, 128, 64), (64, 64, 128), (128, 128, 64),
                (128, 64, 128), (64, 128, 128), (128, 128, 128), (255, 255, 255)
            ]

            # Normaliza o rgb 0 to 1
            distinct_colors_norm = [(r / 255, g / 255, b / 255)
                                    for r, g, b in distinct_colors]

            # Cria dicionario de cores
            color_mapping = {value: colors.rgb2hex(color) for color, value in zip(
                distinct_colors_norm, unique_values)}

            for node in G.nodes():
                papel = G.nodes[node]['papel']
                if papel == 'Unknown':
                    # cinza para desconhecidos
                    G.nodes[node]['color'] = '#000000'
                else:
                    color = color_mapping.get(papel)
                    if color is not None:
                        G.nodes[node]['color'] = color

            if all_options_eua3:
                if len(source_ind) != 0:
                    for node in G.nodes():
                        papel = G.nodes[node]['papel']
                        if papel == 'Unknown':
                            # cinza para desconhecidos
                            G.nodes[node]['color'] = '#000000'
                        else:
                            color = color_mapping.get(papel)
                            if color is not None:
                                G.nodes[node]['color'] = color

                        for node in G.nodes():
                            # checar se os labels não estão no 'email_unicos_ind' list
                            if G.nodes[node]['label'] not in email_unicos_ind:
                                G.nodes[node]['color'] = 'lightgray'

        # 'modified_list_c', modified_list_c
        # cor atributo guilda_cliente e cor
        if selected_option == 'Cliente':
            for i in range(len(modified_list_c)):
                # tipo_ind é o nome da guilda e cliente individual
                if modified_list_c[i]['cliente_guilda'] == tipo_ind:
                    for node in G.nodes():
                        if G.nodes[node]['label'] == modified_list_c[i]['email']:
                            G.nodes[node]['cliente'] = tipo_ind
                            G.nodes[node]['color'] = '#11d63f'
                        else:
                            if 'cliente' not in G.nodes[node]:
                                G.nodes[node]['cliente'] = 'Unknown'
                                G.nodes[node]['color'] = '#eae4e4'

        elif selected_option == 'Guilda':
            for i in range(len(modified_list_g)):
                if modified_list_g[i]['cliente_guilda'] == tipo_ind:
                    for node in G.nodes():
                        if G.nodes[node]['label'] == modified_list_g[i]['email']:
                            G.nodes[node]['guilda'] = tipo_ind
                            G.nodes[node]['color'] = '#11d63f'
                        else:
                            if 'guilda' not in G.nodes[node]:
                                G.nodes[node]['guilda'] = 'Unknown'
                                G.nodes[node]['color'] = '#eae4e4'

        else:
            pass

        if len(source) > 0:
            with col03:
                st.write('\n')
                st.write('\n')
                all_options_comm = st.checkbox("Detectar comunidades")

            with col04:
                resolution = st.slider("Escolha o nível de modularidade:", min_value=0.5,
                                        max_value=1.5, step=0.1, value=1.0, disabled=not all_options_comm, key = "slider_a")
                
                if all_options_comm:
                    num_edges = G.number_of_edges()
                    num_edges
                    if num_edges == 0:
                        community2color = matplotlib.colors.rgb2hex[0]
                        # atribui cores aos nós
                        nx.set_node_attributes(G, community2color, 'color')
                    else:
                        # Detecta comunidades
                        communities = nx.community.greedy_modularity_communities(
                            G, resolution=resolution)
                        
                        # Gera cores distintas para as comunidades
                        num_communities = len(communities)
                        color_palette = plt.cm.hsv(
                            np.linspace(0, 1, num_communities))

                        # Cria dicionario para mapear as cores
                        community2color = {i: matplotlib.colors.rgb2hex(
                            color_palette[i]) for i in range(num_communities)}
                        
                        # Adiciona cores para as comunidades
                        node2color = {}
                        for i, community in enumerate(communities):
                            for node in community:
                                node2color[node] = community2color[i]

                        # atribui cores aos nós
                        nx.set_node_attributes(G, node2color, 'color')

        # colorir todos os individuos selecionados
        # seleciona ind na rede geral todos e nao seleciona ind na rede todos cliente/guida
        # colorir com cores da guilda e cliente principal a rede total
        distinct_colors = [
                (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
                (128, 0, 128), (0, 128, 128), (128, 128, 128), (255, 128, 0), (255, 0, 128), (0, 255, 128), (128, 0, 255), (0, 128, 255),
                (255, 128, 128), (128, 255, 128), (128, 128, 255), (255, 255, 128), (255, 128, 255), (128, 255, 255), (192, 0, 0), (0, 192, 0),
                (0, 0, 192), (192, 192, 0), (192, 0, 192), (0, 192, 192), (192, 192, 192), (255, 192, 0), (255, 0, 192), (0, 255, 192),
                (192, 0, 255), (0, 192, 255), (255, 192, 192), (192, 255, 192), (192, 192, 255), (255, 255, 192), (255, 192, 255), (192, 255, 255),
                (64, 0, 0), (0, 64, 0), (0, 0, 64), (64, 64, 0), (64, 0, 64), (0, 64, 64), (64, 64, 64), (128, 64, 0),
                (128, 0, 64), (0, 128, 64), (64, 0, 128), (0, 64, 128), (128, 64, 64), (64, 128, 64), (64, 64, 128), (128, 128, 64),
                (128, 64, 128), (64, 128, 128), (128, 128, 128), (255, 255, 255)
            ]
        
        if 'color_count' not in st.session_state:
            st.session_state.color_count = 0

        def get_next_color():
            global color_count

            color = '#%02X%02X%02X' % distinct_colors[st.session_state.color_count]
            st.session_state.color_count += 1

            if st.session_state.color_count >= len(distinct_colors):
                st.session_state.color_count = 0

            return color

        cg_colors = {}

        if all_options_gc:
            if selected_option == 'Cliente':
                for node in G.nodes():
                    found = False
                    for i in range(len(new_cliente)):
                        if G.nodes[node]['label'] == new_cliente[i]['email']:
                            cliente = new_cliente[i]['cliente']

                            if cliente not in cg_colors:
                                cg_colors[cliente] = get_next_color()

                            G.nodes[node]['cliente'] = cliente
                            G.nodes[node]['color'] = cg_colors[cliente]
                            found = True
                            break

                    if G.nodes[node]['cliente'] == 'Unknown':
                        G.nodes[node]['color'] = '#000000'

            elif selected_option == 'Guilda':
                for node in G.nodes():
                    found = False
                    for i in range(len(new_guilda)):
                        if G.nodes[node]['label'] == new_guilda[i]['email']:
                            guilda = new_guilda[i]['guilda']

                            if guilda not in cg_colors:
                                cg_colors[guilda] = get_next_color()

                            G.nodes[node]['guilda'] = guilda
                            G.nodes[node]['color'] = cg_colors[guilda]
                            found = True
                            break

                    if G.nodes[node]['guilda'] == 'Unknown':
                        G.nodes[node]['color'] = '#000000'

        # colorir indiv especificos quando 'selecione todos os EuA3' estão selecionados
        ### seleciona ind na rede geral e nao seleciona todos rede cliente guilda principal
        if all_options_gc:
            if selected_option == 'Cliente':
                if all_options_eua3:
                    if len(source_ind) != 0:
                        for node in G.nodes():
                            found = False
                            for i in range(len(new_cliente)):
                                if G.nodes[node]['label'] == new_cliente[i]:
                                    cliente = new_cliente[i]['cliente']

                                    if cliente not in cg_colors:
                                        cg_colors[cliente] = get_next_color()

                                    G.nodes[node]['cliente'] = cliente
                                    G.nodes[node]['color'] = cg_colors[cliente]
                                    found = True
                                    break

                                if G.nodes[node]['cliente'] == 'Unknown':
                                    G.nodes[node]['color'] = '#000000'

                                # checar se os labels não estão no 'email_unicos_ind' list
                                if G.nodes[node]['label'] not in email_unicos_ind:
                                    G.nodes[node]['color'] = 'lightgray'

            elif selected_option == 'Guilda':
                if all_options_eua3:
                    if len(source_ind) != 0:
                        for node in G.nodes():
                            found = False
                            for i in range(len(new_guilda)):
                                if G.nodes[node]['label'] == new_guilda[i]:
                                    guilda = new_guilda[i]['guilda']

                                    if guilda not in cg_colors:
                                        cg_colors[guilda] = get_next_color()

                                    G.nodes[node]['guilda'] = guilda
                                    G.nodes[node]['color'] = cg_colors[guilda]
                                    found = True
                                    break 

                                if G.nodes[node]['guilda'] == 'Unknown':
                                    G.nodes[node]['color'] = '#000000'

                                # checar se os labels não estão no 'email_unicos_ind'
                                if G.nodes[node]['label'] not in email_unicos_ind:
                                    G.nodes[node]['color'] = 'lightgray'

        # pyvis no streamlit
        net = Network(directed=True, notebook=True,
                        cdn_resources='in_line', neighborhood_highlight=True)
        # net.repulsion()  # necessário para estabilizar a visualização
        # net.show_buttons(filter_= ['nodes'])
        net.set_options("""var options = {
                            "edges": {
                                "color": {
                                "inherit": true
                                },
                                "smooth": false
                            },

                            "layout": {
                                    
                                "set_separation": 200

                            },

                            "physics": {
                                "hierarchicalRepulsion": {
                                "centralGravity": 0,
                                "springConstant": 0.05,
                                "nodeDistance": 200
                                },
                                "minVelocity": 0.75,
                                "solver": "hierarchicalRepulsion"
                            }
                            }""")
        net.from_nx(G)

        # adiciona nós que vieram do networkx ao pyvis já com as pŕopriedades criadas e armazenadas no objeto networkx
        for node in G.nodes():
            size = G.nodes[node]['size']
            net.add_node(node, size=size)

        # criar o arquivo htlml utilizado para visualização no pyvix
        html = net.generate_html()
        with open("ona.html", mode='w', encoding='utf-8') as fp:
            fp.write(html)

        col111, col1123, col222 = st.columns([2.5, .5, 1])

        with col111:

            # projeta o html no streamlit
            HtmlFile = open("ona.html", 'r', encoding='utf-8')
            source_code = HtmlFile.read()
            components.html(source_code, height=650)

        # legenda
        with col1123:
            if not all_options_comm:
                if selected_atr_cg == 'Atributos dos Colaboradores':
                    color_pairs = set((d['color'], d[selected_property])
                                        for n, d in G.nodes(data=True))

                    df_legend = pd.DataFrame(color_pairs, columns=[
                                                'color', selected_property])

                    # capitaliza as categorias para todas serem ordenadas alfabeticamente
                    df_legend.iloc[:, 1] = df_legend.iloc[:,
                                                            1].str.capitalize()
                    

                    # Exclude color 'lightgray' when label is not 'Unknown'
                    df_legend = df_legend[~((df_legend['color'] == 'lightgray'))]

                    # Ordena alfabeticamente as categories
                    df_legend = df_legend.sort_values(by=selected_property)

                    patches = [mpatches.Patch(
                        color=row['color'], label=row[selected_property]) for idx, row in df_legend.iterrows()]
                    fig, ax = plt.subplots()
                    legend = plt.legend(
                        handles=patches, loc='upper left', prop={"size": 25})

                    # esconde os eixos do grafico
                    plt.axis('off')

                    # Display gravura streamlit
                    st.pyplot(fig)

                elif selected_atr_cg == 'Clientes e Guildas':
                    color_pairs = set(
                        (d['color'], d[str.lower(selected_option)]) for n, d in G.nodes(data=True))

                    df_legend = pd.DataFrame(color_pairs, columns=[
                                                'color', selected_option])

                    # capitaliza as categorias para todas serem ordenadas alfabeticamente
                    df_legend.iloc[:, 1] = df_legend.iloc[:,
                                                            1].str.capitalize()

                    # exclui cor '#lightgray' quando label é não 'Unknown'
                    df_legend = df_legend[~((df_legend['color'] == 'lightgray') )]

                    # Ordena alfabeticamente as categories
                    df_legend = df_legend.sort_values(by=selected_option)

                    patches = [mpatches.Patch(
                        color=row['color'], label=row[selected_option]) for idx, row in df_legend.iterrows()]
                    fig, ax = plt.subplots()
                    legend = plt.legend(
                        handles=patches, loc='upper left', prop={"size": 25})

                    # esconde os eixos do grafico
                    plt.axis('off')

                    # Display gravura streamlit
                    st.pyplot(fig)
        
        with col222:

            col4, col5 = st.columns([.5 ,3.5])
            with col5:
                st.title(f"Medidas descritivas")
                st.write("Laços:", str(num_ties))
                st.write("Nós:", str(num_nodes))
                st.write("Grau médio:", str(round(average_degree, 2)))
                st.write("Densidade (%):", str(round(density, 2)*100)[:4])
                st.write("Diâmetro:", str(diameter))                
            

    ##############################################################################################################################
    # tab redes guilda
    ##############################################################################################################################
    with tab2:
        # cria o driver python
        driver = GraphDatabase.driver(
            NEO4J_URL, auth=(NEO4J_USER, NEO4J_PASSWORD))

        def perform_eua3(session, dimensao, nome, rodada, email):

            main_query = f"""
                MATCH (n)-[r:PERTENCE_{rodada}]->(guilda:Guilda {{nome:$nome}})
                WHERE n.email IS NOT NULL AND n.email <> "" AND n.email IN $email
                WITH collect(n) AS GuildaNodes, guilda
                UNWIND GuildaNodes as n1

                OPTIONAL MATCH (n1)-[r2 {{rodada: $rodada, dimensao: $dimensao}}]->(n2)
                WHERE (n2)-[:PERTENCE_{rodada}]->(guilda) OR n2 IS NULL

                RETURN DISTINCT n1, r2, n2, id(n1) AS n1_id, id(n2) AS n2_id
            """
            def read_tx(tx):
                result = tx.run(main_query, email=email, nome=nome, rodada=rodada, dimensao=dimensao)
                records = list(result)
                summary = result.consume()
                return records, summary

            records, summary = session.execute_read(read_tx)
            
            # Extrair node ids do nós da rede
            node_ids = node_ids = [record['n1_id'] for record in records] + [record['n2_id'] for record in records]
            unique_node_ids = list(set(node_ids))

            update_query = f"""
                MATCH (c:Colaborador)-[:ATUA_COMO_{rodada}]->(p:Papel)
                WHERE id(c) IN $node_ids
                SET c.papel = p.nome
            """
            session.execute_write(lambda tx: tx.run(update_query, node_ids=unique_node_ids))
            return records, summary
        
        # título
        title = st.title(f"Rede EuA3:")
        st.markdown('<style>h1{font-size: 24px;}</style>',
                    unsafe_allow_html=True)

        with driver.session() as session:

            # Cria a caixa lateral das medidas de centralidade
            col1, col2, col3, col4 = st.columns(4)
            col1b, col2b, col3b, col4b = st.columns(4)  

            with col3:    
                query_rodada = f"MATCH (n)-[r]->(n2) WHERE r.rodada IS NOT NULL RETURN DISTINCT r.rodada AS r_rod"
                rod_g = [record["r_rod"]for record in session.run(query_rodada)]

                def to_date(s):    
                    months = ["JAN", "FEV", "MAR", "ABR", "MAI", "JUN", "JUL", "AGO", "SET", "OUT", "NOV", "DEZ"]
                    month, year = s[:3], s[3:]
                    return datetime(int(year), months.index(month) + 1, 1)
                rod_g.sort(key=to_date, reverse=True)
                
                rod = st.selectbox("Selecione a rodada", rod_g, key = 'Guilda_d')
                
            with col4:
                query_dimensao = f"MATCH (n)-[r]->(n2) WHERE r.rodada = '{rod}' RETURN DISTINCT r.dimensao AS r_dim"
                dim_g = [record["r_dim"]for record in session.run(query_dimensao)]
                dim = st.selectbox("Selecione a dimensão", sorted(dim_g), key='Guilda_c')

            with col1b:
                query_cg_3 = f"MATCH ()-[r: PERTENCE_{rod}]->(g:Guilda) RETURN DISTINCT g.nome AS g_nome"
                rede_list_Guilda = [record["g_nome"]for record in session.run(query_cg_3)]
                nome = st.selectbox("Selecione a Guilda:", sorted(rede_list_Guilda), key='Guilda_b')

            with col1:
                
                e_list = [
                    (record["c1_email"], record["c2_email"])
                    for record in session.run(
                        f"""
                        MATCH (n)-[r:PERTENCE_{rod}]->(:Guilda {{nome:$nome}})
                        WITH collect(n) AS GuildaNodes
                        UNWIND GuildaNodes as c1
                        UNWIND GuildaNodes as c2
                        OPTIONAL MATCH (c1)-[b1{{rodada: $rodada, dimensao: $dimensao}}]->(c2) 
                        RETURN DISTINCT c1.email AS c1_email, c2.email AS c2_email
                        """,

                        dimensao=dim,
                        rodada=rod,
                        nome=nome
                    )]          

                email_list = {name for tuple_ in e_list for name in tuple_ if name is not None}

                source = st.multiselect(
                    "Selecione um ou mais EuA3:", sorted(email_list), key='Guilda_e')
                source_ind = source

            with col2:   
                st.write('\n')
                st.write('\n')   
                all_options_eua3 = st.checkbox("Selecione todos os EuA3", key = 'checkbox_4')
                if all_options_eua3:
                    source = list(email_list) 

            records, summary = perform_eua3(session, dimensao=dim, nome=nome, rodada=rod, email=source)

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

            records_ind, summary = perform_eua3(session, dimensao=dim, nome=nome, rodada=rod, email=source)

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

            # selecionar Guilda para colorir como caracteristicas o nós
            col51, col52, col54, col55 = st.columns(4)

            query_cg = None
            selected_option = None
            selected_property = None

            with col2b:
                selected_property = st.selectbox('Atributos dos Colaboradores:', [
                                                'Selecione uma característica', 'Papel', 'Gênero', 'Tempo de casa'], key = 'propriedade_c')
                if selected_property == 'Papel':
                    selected_property = 'papel'
                elif selected_property == 'Gênero':
                    selected_property = 'genero'
                elif selected_property == 'Tempo de casa':
                    selected_property = 'tempoDeCasa'
                else:
                    selected_property = None 

            G = nx.DiGraph()

            for record in records:
                source_id = record["n1"].element_id
                G.add_node(source_id, label=record["n1"]["email"], genero=record["n1"]["genero"], papel=record["n1"]
                        ["papel"], tempoDeCasa=record["n1"]["tempoDeCasa"], idade=record["n1"]["idade"])

                if record["n2"] is not None:  # Checar se n2 não é none
                    target_id = record["n2"].element_id
                    G.add_node(target_id, label=record["n2"]["email"], genero=record["n2"]["genero"], papel=record["n2"]
                            ["papel"], tempoDeCasa=record["n2"]["tempoDeCasa"], idade=record["n2"]["idade"])
                    
                    if record["r2"] is not None:  # Checar se r2 não é none
                        G.add_edge(source_id, target_id, label=record["r2"])

            # define o tipo de visualização do networkx
            pos = nx.fruchterman_reingold_layout(G)

            # exclui edge label '= None' que vem do neo4j para não ser visualizado no networkx para melhorar a vizualização geral do grafo
            nx.set_edge_attributes(G, None, "label")
            nx.draw_networkx_nodes(G, pos)
            nx.draw_networkx_edges(G, pos)
            nx.draw_networkx_labels(G, pos)

            # Define a caixa com as medidas de centralidade
            centrality_measures = {
                'InDegree': nx.in_degree_centrality(G),
                'OutDegree': nx.out_degree_centrality(G),
                'Betweenness': nx.betweenness_centrality(G),
                'Closeness': nx.closeness_centrality(G),
                'Page Rank': nx.pagerank(G)
            }

            # Cria a caixa lateral das medidas de centralidade
            col11, col12, col01, col02 = st.columns(4)

            with col11:
                centrality = st.selectbox(
                    'Selecione uma medida de centralização:', list(centrality_measures.keys()), key = 'Guilda_select')

            # visualizacao
            # Calcula as medidas descritiva

            # número de laços
            num_ties = G.number_of_edges()

            # número de nós
            num_nodes = G.number_of_nodes()

            # densidade
            density = nx.density(G)
        
            # calcula o diametro do maior componente
            def effective_diameter(Gu):
                subG = Gu.subgraph([node for node, degree in Gu.degree() if degree > 0])
                if not nx.is_connected(subG):
                    diameters = [nx.diameter(subG.subgraph(c)) for c in nx.connected_components(subG)]
                    return max(diameters)
                else:
                    return nx.diameter(subG)
            if all(degree == 0 for _, degree in G.degree()):
                diameter = 0
            else:
                Gu = nx.to_undirected(G)
                diameter = effective_diameter(Gu) 

            # degree médio
            if G.number_of_nodes() > 0:
                average_degree = num_ties/num_nodes
            else:
                average_degree = 0


            # título
            title.title(
                f"Rede EuA3: Dimensão {dim} em {rod} - Medida de Centralidade: {centrality}")
            st.markdown('<style>h1{font-size: 24px;}</style>',
                        unsafe_allow_html=True)

            st.write('\n')

            # Pega as medidas de centralização do networkx
            nx.set_node_attributes(G, 10, 'size')
            nc = nx.get_node_attributes(G, 'size')

            ns = centrality_measures[centrality]
            ns = {node: ns[node]*50 + 10 for node in G.nodes()}
            nx.set_node_attributes(G, ns, 'size')
            nc = nx.get_node_attributes(G, 'size')

            # colorir os nós mais centrais
            sorted_nodes = sorted(nc, key=nc.get, reverse=True)
            # Cria um botão para selecionar o numéro de nós mais centrais
            num_highlight = None

            with col12:
                num_highlight = st.selectbox("Escolha a quantidade de EuA3 mais centrais", [
                    i for i in range(0, len(sorted_nodes) + 1)], key = 'mais_centrais_c')

            if selected_property is None:
                # Defini as cores do nós da rede e pela centralidade
                if len(source_ind) == 0:
                    node_color_dict = {}
                    for node in G.nodes():
                        # rede apos a escolha de medidas de centralidade
                        if node in sorted_nodes[:num_highlight]:
                            node_color_dict[node] = '#dd4b39'  # cor mais centrais
                        else:
                            node_color_dict[node] = '#1779e1'  # cor menos centrais
                    nx.set_node_attributes(G, node_color_dict, 'color')

                if len(source_ind) != 0:
                    node_color_dict = {}
                    for node in G.nodes():
                        # rede apos a escolha de medidas de centralidade
                        if node in sorted_nodes[:num_highlight]:
                            node_color_dict[node] = '#dd4b39'  # cor mais centrais
                        else:
                            node_color_dict[node] = '#1779e1'  # cor menos centrais
                    nx.set_node_attributes(G, node_color_dict, 'color')

                    for node in G.nodes():
                        # checar se os labels não estão no 'email_unicos_ind' list
                        if G.nodes[node]['label'] not in email_unicos_ind:
                            G.nodes[node]['color'] = 'lightgray'


            elif selected_property != 'Selecione uma característica':
                # cor propriedades
                # cor genero
                node_color_dict = {}
                if selected_property == 'genero':
                    for node in G.nodes():
                        if G.nodes[node]['genero'] == 'f':
                            node_color_dict[node] = '#00CC00'
                        elif G.nodes[node]['genero'] == 'm':
                            node_color_dict[node] = '#d3d020e8'
                        elif G.nodes[node]['genero'] == 'feminino':
                            node_color_dict[node] = '#00CC00'
                        elif G.nodes[node]['genero'] == 'masculino':
                            node_color_dict[node] = '#d3d020e8'
                        elif G.nodes[node]['genero'] == 'Unknown':
                            node_color_dict[node] = '#000000'
                        else:
                            pass
                nx.set_node_attributes(G, node_color_dict, 'color')

                node_color_dict = {}
                if all_options_eua3:
                    if len(source_ind) != 0:
                        if selected_property == 'genero':
                            for node in G.nodes():
                                if G.nodes[node]['genero'] == 'f':
                                    node_color_dict[node] = '#00CC00'
                                elif G.nodes[node]['genero'] == 'm':
                                    node_color_dict[node] = '#d3d020e8'
                                elif G.nodes[node]['genero'] == 'feminino':
                                    node_color_dict[node] = '#00CC00'
                                elif G.nodes[node]['genero'] == 'masculino':
                                    node_color_dict[node] = '#d3d020e8'
                                elif G.nodes[node]['genero'] == 'Unknown':
                                    node_color_dict[node] = '#000000'
                                else:
                                    pass
                        nx.set_node_attributes(G, node_color_dict, 'color')
                        for node in G.nodes():
                            # checar se os labels não estão no 'email_unicos_ind' list
                            if G.nodes[node]['label'] not in email_unicos_ind:
                                G.nodes[node]['color'] = 'lightgray'

                # cor tempo de casa
                node_color_dict = {}
                if selected_property == 'tempoDeCasa':
                    for node in G.nodes():
                        if G.nodes[node]['tempoDeCasa'] == 'De 1 a 6 meses':
                            node_color_dict[node] = '#ca98ec'
                        elif G.nodes[node]['tempoDeCasa'] == 'De 6 meses a 1 ano':
                            node_color_dict[node] = '#ca9800'
                        elif G.nodes[node]['tempoDeCasa'] == 'Recém ingressante':
                            node_color_dict[node] = '#097c20'
                        elif G.nodes[node]['tempoDeCasa'] == 'Mais que um ano':
                            node_color_dict[node] = '#ca0000'
                        elif G.nodes[node]['tempoDeCasa'] == 'Unknown':
                            node_color_dict[node] = '#000000'
                        else:
                            pass
                nx.set_node_attributes(G, node_color_dict, 'color')

                node_color_dict = {}
                if all_options_eua3:
                    if len(source_ind) != 0:
                        if selected_property == 'tempoDeCasa':
                            for node in G.nodes():
                                if G.nodes[node]['tempoDeCasa'] == 'De 1 a 6 meses':
                                    node_color_dict[node] = '#ca98ec'
                                elif G.nodes[node]['tempoDeCasa'] == 'De 6 meses a 1 ano':
                                    node_color_dict[node] = '#ca9800'
                                elif G.nodes[node]['tempoDeCasa'] == 'Recém ingressante':
                                    node_color_dict[node] = '#097c20'
                                elif G.nodes[node]['tempoDeCasa'] == 'Mais que um ano':
                                    node_color_dict[node] = '#ca0000'
                                elif G.nodes[node]['tempoDeCasa'] == 'Unknown':
                                    node_color_dict[node] = '#000000'
                                else:
                                    pass
                        nx.set_node_attributes(G, node_color_dict, 'color')
                        for node in G.nodes():
                            # checar se os labels não estão no 'email_unicos_ind' list
                            if G.nodes[node]['label'] not in email_unicos_ind:
                                G.nodes[node]['color'] = 'lightgray'

                # cor papel
                color_mapping = {}
                if selected_property == 'papel':
                    unique_values = list(set(node['papel']
                                        for node in G.nodes().values()))
                    num_unique_values = len(unique_values)

                    # 50 cores para funçoes
                    distinct_colors = [
                        (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255,
                                                                                255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
                        (128, 0, 128), (0, 128, 128), (128, 128, 128), (255, 128,
                                                                        0), (255, 0, 128), (0, 255, 128), (128, 0, 255), (0, 128, 255),
                        (255, 128, 128), (128, 255, 128), (128, 128, 255), (255, 255,
                                                                            128), (255, 128, 255), (128, 255, 255), (192, 0, 0), (0, 192, 0),
                        (0, 0, 192), (192, 192, 0), (192, 0, 192), (0, 192, 192), (192,
                                                                                192, 192), (255, 192, 0), (255, 0, 192), (0, 255, 192),
                        (192, 0, 255), (0, 192, 255), (255, 192, 192), (192, 255, 192), (192,
                                                                                        192, 255), (255, 255, 192), (255, 192, 255), (192, 255, 255),
                        (64, 0, 0), (0, 64, 0), (0, 0, 64), (64, 64, 0), (64,
                                                                        0, 64), (0, 64, 64), (64, 64, 64), (128, 64, 0),
                        (128, 0, 64), (0, 128, 64), (64, 0, 128), (0, 64, 128), (128,
                                                                                64, 64), (64, 128, 64), (64, 64, 128), (128, 128, 64),
                        (128, 64, 128), (64, 128, 128), (0, 0,
                                                        0), (128, 128, 128), (255, 255, 255)
                    ]

                    # Normaliza o rgb 0 to 1
                    distinct_colors_norm = [(r / 255, g / 255, b / 255)
                                            for r, g, b in distinct_colors]

                    # Cria dicionario de cores
                    color_mapping = {value: colors.rgb2hex(color) for color, value in zip(
                        distinct_colors_norm, unique_values)}

                    for node in G.nodes():
                        papel = G.nodes[node]['papel']
                        if papel == 'Unknown':
                            # cinza para desconhecidos
                            G.nodes[node]['color'] = '#000000'
                        else:
                            color = color_mapping.get(papel)
                            if color is not None:
                                G.nodes[node]['color'] = color

                    if all_options_eua3:
                        if len(source_ind) != 0:
                            for node in G.nodes():
                                papel = G.nodes[node]['papel']
                                if papel == 'Unknown':
                                    # cinza para desconhecidos
                                    G.nodes[node]['color'] = '#000000'
                                else:
                                    color = color_mapping.get(papel)
                                    if color is not None:
                                        G.nodes[node]['color'] = color

                                for node in G.nodes():
                                    # checar se os labels não estão no 'email_unicos_ind' list
                                    if G.nodes[node]['label'] not in email_unicos_ind:
                                        G.nodes[node]['color'] = 'lightgray'
            # 
            if len(source) > 0:
                with col01:
                    st.write('\n')
                    st.write('\n')
                    all_options_comm = st.checkbox("Detectar comunidades", key = 'checkbox_c')

                with col02:
                    resolution = st.slider("Escolha o nível de modularidade:", min_value=0.5,
                                        max_value=1.5, step=0.1, value=1.0, disabled=not all_options_comm, key = 'slider_b')

                    if all_options_comm:
                        num_edges = G.number_of_edges()
                        if num_edges == 0:
                            color = plt.cm.hsv(0)
                            hex_color = matplotlib.colors.rgb2hex(color)
                            
                            # Assign this color to all nodes in the graph
                            nx.set_node_attributes(G, {node: hex_color for node in G.nodes()}, 'color')
                        else:
                            # Detecta comunidades
                            communities = nx.community.greedy_modularity_communities(
                                G, resolution=resolution)

                            # Gera cores distintas para as comunidades
                            num_communities = len(communities)
                            color_palette = plt.cm.hsv(
                                np.linspace(0, 1, num_communities))

                            # Cria dicionario para mapear as cores
                            community2color = {i: matplotlib.colors.rgb2hex(
                                color_palette[i]) for i in range(num_communities)}

                            # Adiciona cores para as comunidades
                            node2color = {}
                            for i, community in enumerate(communities):
                                for node in community:
                                    node2color[node] = community2color[i]

                            # atribui cores aos nós
                            nx.set_node_attributes(G, node2color, 'color')

            # pyvis no streamlit
            net = Network(directed=True, notebook=True,
                        cdn_resources='in_line', neighborhood_highlight=True)
            # net.repulsion()  # necessário para estabilizar a visualização
            # net.show_buttons(filter_= ['nodes'])
            net.set_options("""var options = {
                                "edges": {
                                    "color": {
                                    "inherit": true
                                    },
                                    "smooth": false
                                },

                                "layout": {
                                        
                                    "set_separation": 200

                                },

                                "physics": {
                                    "hierarchicalRepulsion": {
                                    "centralGravity": 0.9,
                                    "springConstant": 0.05,
                                    "nodeDistance": 200
                                    },
                                    "minVelocity": 0.75,
                                    "solver": "hierarchicalRepulsion"
                                }
                                }""")
            net.from_nx(G)

            # adiciona nós que vieram do networkx ao pyvis já com as pŕopriedades criadas e armazenadas no objeto networkx
            for node in G.nodes():
                size = G.nodes[node]['size']
                net.add_node(node, size=size)

            # criar o arquivo htlml utilizado para visualização no pyvix
            html = net.generate_html()
            with open("ona.html", mode='w', encoding='utf-8') as fp:
                fp.write(html)
                
            col111, col1123, col222 = st.columns([2.5, .5, 1])

            with col111:

                # projeta o html no streamlit
                HtmlFile = open("ona.html", 'r', encoding='utf-8')
                source_code = HtmlFile.read()
                components.html(source_code, height=650)

            # legenda
            with col1123:
                if selected_property is not None:
                    color_pairs = set((d['color'], d[selected_property])
                                    for n, d in G.nodes(data=True))

                    df_legend = pd.DataFrame(color_pairs, columns=[
                                            'color', selected_property])

                    # capitaliza as categorias para todas serem ordenadas alfabeticamente
                    df_legend.iloc[:, 1] = df_legend.iloc[:,
                                                        1].str.capitalize()

                    # Exclude color 'lightgray' when label is not 'Unknown'
                    df_legend = df_legend[~((df_legend['color'] == 'lightgray'))]

                    # Ordena alfabeticamente as categories
                    df_legend = df_legend.sort_values(by=selected_property)

                    patches = [mpatches.Patch(
                        color=row['color'], label=row[selected_property]) for idx, row in df_legend.iterrows()]
                    fig, ax = plt.subplots()
                    legend = plt.legend(
                        handles=patches, loc='upper left', prop={"size": 25})

                    # esconde os eixos do grafico
                    plt.axis('off')

                    # Display gravura streamlit
                    st.pyplot(fig)

            with col222:

                col4, col5 = st.columns([.5 ,3.5])
                with col5:
                    st.title(f"Medidas descritivas")
                    st.write("Laços:", str(num_ties))
                    st.write("Nós:", str(num_nodes))
                    st.write("Grau médio:", str(round(average_degree, 2)))
                    st.write("Densidade (%):", str(round(density, 2)*100)[:4])
                    st.write("Diâmetro:", str(diameter)) 
                    
    ##############################################################################################################################
    # tab redes cliente
    ##############################################################################################################################
    with tab3:
        # cria o driver python
        driver = GraphDatabase.driver(
            NEO4J_URL, auth=(NEO4J_USER, NEO4J_PASSWORD))

        def perform_eua3(session, dimensao, nome, rodada, email):

            main_query = f"""
                MATCH (n)-[r:ALOCADO_{rodada}]->(cliente:Cliente {{nome:$nome}})
                WHERE n.email IS NOT NULL AND n.email <> "" AND n.email IN $email
                WITH collect(n) AS ClienteNodes, cliente
                UNWIND ClienteNodes as n1

                OPTIONAL MATCH (n1)-[r2 {{rodada: $rodada, dimensao: $dimensao}}]->(n2)
                WHERE (n2)-[:ALOCADO_{rodada}]->(cliente) OR n2 IS NULL

                RETURN DISTINCT n1, r2, n2, id(n1) AS n1_id, id(n2) AS n2_id
            """
            def read_tx(tx):
                result = tx.run(main_query, email=email, nome=nome, rodada=rodada, dimensao=dimensao)
                records = list(result)
                summary = result.consume()
                return records, summary

            records, summary = session.execute_read(read_tx)
            
            # Extrair node ids do nós da rede
            node_ids = node_ids = [record['n1_id'] for record in records] + [record['n2_id'] for record in records]
            unique_node_ids = list(set(node_ids))

            update_query = f"""
                MATCH (c:Colaborador)-[:ATUA_COMO_{rodada}]->(p:Papel)
                WHERE id(c) IN $node_ids
                SET c.papel = p.nome
            """
            session.execute_write(lambda tx: tx.run(update_query, node_ids=unique_node_ids))
            return records, summary
        
        # título
        title = st.title(f"Rede EuA3:")
        st.markdown('<style>h1{font-size: 24px;}</style>',
                    unsafe_allow_html=True)

        with driver.session() as session:

            # Cria a caixa lateral das medidas de centralidade
            col1, col2, col3, col4 = st.columns(4)
            col1b, col2b, col3b, col4b = st.columns(4)  

            with col3:    
                query_rodada = f"MATCH (n)-[r]->(n2) WHERE r.rodada IS NOT NULL RETURN DISTINCT r.rodada AS r_rod"
                rod_g = [record["r_rod"]
                            for record in session.run(query_rodada)]

                def to_date(s):    
                    months = ["JAN", "FEV", "MAR", "ABR", "MAI", "JUN", "JUL", "AGO", "SET", "OUT", "NOV", "DEZ"]
                    month, year = s[:3], s[3:]
                    return datetime(int(year), months.index(month) + 1, 1)
                rod_g.sort(key=to_date, reverse=True)
                
                rod = st.selectbox("Selecione a rodada", rod_g, key = 'Cliente_d')
                
            with col4:
                query_dimensao = f"MATCH (n)-[r]->(n2) WHERE r.rodada = '{rod}' RETURN DISTINCT r.dimensao AS r_dim"
                dim_g = [record["r_dim"]
                            for record in session.run(query_dimensao)]
                dim = st.selectbox(
                    "Selecione a dimensão", sorted(dim_g), key='Cliente_c')

            with col1b:
                query_cg_3 = f"MATCH ()-[r: ALOCADO_{rod}]->(g:Cliente) RETURN DISTINCT g.nome AS g_nome"
                rede_list_Cliente = [record["g_nome"]
                                    for record in session.run(query_cg_3)]
                nome = st.selectbox(
                    "Selecione o cliente:", sorted(rede_list_Cliente), key='Cliente_b')

            with col1:
            
                e_list = [
                    (record["c1_email"], record["c2_email"])
                    for record in session.run(
                        f"""
                        MATCH (n)-[r:ALOCADO_{rod}]->(:Cliente {{nome:$nome}})
                        WITH collect(n) AS ClienteNodes
                        UNWIND ClienteNodes as c1
                        UNWIND ClienteNodes as c2
                        OPTIONAL MATCH (c1)-[b1{{rodada: $rodada, dimensao: $dimensao}}]->(c2) 
                        RETURN DISTINCT c1.email AS c1_email, c2.email AS c2_email
                        """,

                        dimensao=dim,
                        rodada=rod,
                        nome=nome
                    )]          

                email_list = {name for tuple_ in e_list for name in tuple_ if name is not None}

                source = st.multiselect(
                    "Selecione um ou mais EuA3:", sorted(email_list), key='Cliente_e')
                source_ind = source

            with col2:   
                st.write('\n')
                st.write('\n')   
                all_options_eua3 = st.checkbox("Selecione todos os EuA3", key = 'checkbox_3')
                if all_options_eua3:
                    source = list(email_list) 

            records, summary = perform_eua3(session, dimensao=dim, nome=nome, rodada=rod, email=source)

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

            records_ind, summary = perform_eua3(session, dimensao=dim, nome=nome, rodada=rod, email=source)

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

            # selecionar Cliente e cliente para colorir como caracteristicas o nós
            col51, col52, col54, col55 = st.columns(4)

            query_cg = None
            selected_option = None
            selected_property = None

            with col2b:
                selected_property = st.selectbox('Atributos dos Colaboradores:', [
                                                'Selecione uma característica', 'Papel', 'Gênero', 'Tempo de casa'], key = 'propriedade_b')
                if selected_property == 'Papel':
                    selected_property = 'papel'
                elif selected_property == 'Gênero':
                    selected_property = 'genero'
                elif selected_property == 'Tempo de casa':
                    selected_property = 'tempoDeCasa'
                else:
                    selected_property = None 

            G = nx.DiGraph()

            # for record in records:
            #     source_id = record["n1"].element_id
            #     target_id = record["n2"].element_id
            #     G.add_node(source_id, label=record["n1"]["email"], genero=record["n1"]["genero"], papel=record["n1"]
            #             ["papel"], tempoDeCasa=record["n1"]["tempoDeCasa"], idade=record["n1"]["idade"])
            #     G.add_node(target_id, label=record["n2"]["email"], genero=record["n2"]["genero"], papel=record["n2"]
            #             ["papel"], tempoDeCasa=record["n2"]["tempoDeCasa"], idade=record["n2"]["idade"])
            #     G.add_edge(source_id, target_id, label=record["r2"])

            for record in records:
                source_id = record["n1"].element_id
                G.add_node(source_id, label=record["n1"]["email"], genero=record["n1"]["genero"], papel=record["n1"]
                        ["papel"], tempoDeCasa=record["n1"]["tempoDeCasa"], idade=record["n1"]["idade"])

                if record["n2"] is not None:  # Checar se n2 não é none
                    target_id = record["n2"].element_id
                    G.add_node(target_id, label=record["n2"]["email"], genero=record["n2"]["genero"], papel=record["n2"]
                            ["papel"], tempoDeCasa=record["n2"]["tempoDeCasa"], idade=record["n2"]["idade"])
                    
                    if record["r2"] is not None:  # Checar se r2 não é none
                        G.add_edge(source_id, target_id, label=record["r2"])

            # define o tipo de visualização do networkx
            pos = nx.fruchterman_reingold_layout(G)

            # exclui edge label '= None' que vem do neo4j para não ser visualizado no networkx para melhorar a vizualização geral do grafo
            nx.set_edge_attributes(G, None, "label")
            nx.draw_networkx_nodes(G, pos)
            nx.draw_networkx_edges(G, pos)
            nx.draw_networkx_labels(G, pos)

            # Define a caixa com as medidas de centralidade
            centrality_measures = {
                'InDegree': nx.in_degree_centrality(G),
                'OutDegree': nx.out_degree_centrality(G),
                'Betweenness': nx.betweenness_centrality(G),
                'Closeness': nx.closeness_centrality(G),
                'Page Rank': nx.pagerank(G)
            }

            # Cria a caixa lateral das medidas de centralidade
            col11, col12, col01, col02 = st.columns(4)

            with col11:
                centrality = st.selectbox(
                    'Selecione uma medida de centralização:', list(centrality_measures.keys()), key = 'Cliente_select')

            # visualizacao
            # Calcula as medidas descritiva

            # número de laços
            num_ties = G.number_of_edges()

            # número de nós
            num_nodes = G.number_of_nodes()

            # densidade
            density = nx.density(G)
        
            # calcula o diametro do maior componente
            def effective_diameter(Gu):
                subG = Gu.subgraph([node for node, degree in Gu.degree() if degree > 0])
                if not nx.is_connected(subG):
                    diameters = [nx.diameter(subG.subgraph(c)) for c in nx.connected_components(subG)]
                    return max(diameters)
                else:
                    return nx.diameter(subG)
            if all(degree == 0 for _, degree in G.degree()):
                diameter = 0
            else:
                Gu = nx.to_undirected(G)
                diameter = effective_diameter(Gu) 

            # degree médio
            if G.number_of_nodes() > 0:
                average_degree = num_ties/num_nodes
            else:
                average_degree = 0

            # título
            title.title(
                f"Rede EuA3: Dimensão {dim} em {rod} - Medida de Centralidade: {centrality}")
            st.markdown('<style>h1{font-size: 24px;}</style>',
                        unsafe_allow_html=True)

            st.write('\n')

            # Pega as medidas de centralização do networkx
            nx.set_node_attributes(G, 10, 'size')
            nc = nx.get_node_attributes(G, 'size')

            ns = centrality_measures[centrality]
            ns = {node: ns[node]*50 + 10 for node in G.nodes()}
            nx.set_node_attributes(G, ns, 'size')
            nc = nx.get_node_attributes(G, 'size')

            # colorir os nós mais centrais
            sorted_nodes = sorted(nc, key=nc.get, reverse=True)
            # Cria um botão para selecionar o numéro de nós mais centrais
            num_highlight = None

            with col12:
                num_highlight = st.selectbox("Escolha a quantidade de EuA3 mais centrais", [
                    i for i in range(0, len(sorted_nodes) + 1)], key = 'mais_centrais c')

            if selected_property is None:
                # Defini as cores do nós da rede e pela centralidade
                if len(source_ind) == 0:
                    node_color_dict = {}
                    for node in G.nodes():
                        # rede apos a escolha de medidas de centralidade
                        if node in sorted_nodes[:num_highlight]:
                            node_color_dict[node] = '#dd4b39'  # cor mais centrais
                        else:
                            node_color_dict[node] = '#1779e1'  # cor menos centrais
                    nx.set_node_attributes(G, node_color_dict, 'color')

                if len(source_ind) != 0:
                    node_color_dict = {}
                    for node in G.nodes():
                        # rede apos a escolha de medidas de centralidade
                        if node in sorted_nodes[:num_highlight]:
                            node_color_dict[node] = '#dd4b39'  # cor mais centrais
                        else:
                            node_color_dict[node] = '#1779e1'  # cor menos centrais
                    nx.set_node_attributes(G, node_color_dict, 'color')

                    for node in G.nodes():
                        # checar se os labels não estão no 'email_unicos_ind' list
                        if G.nodes[node]['label'] not in email_unicos_ind:
                            G.nodes[node]['color'] = 'lightgray'


            elif selected_property != 'Selecione uma característica':
                # cor propriedades
                # cor genero
                node_color_dict = {}
                if selected_property == 'genero':
                    for node in G.nodes():
                        if G.nodes[node]['genero'] == 'f':
                            node_color_dict[node] = '#00CC00'
                        elif G.nodes[node]['genero'] == 'm':
                            node_color_dict[node] = '#d3d020e8'
                        elif G.nodes[node]['genero'] == 'feminino':
                            node_color_dict[node] = '#00CC00'
                        elif G.nodes[node]['genero'] == 'masculino':
                            node_color_dict[node] = '#d3d020e8'
                        elif G.nodes[node]['genero'] == 'Unknown':
                            node_color_dict[node] = '#000000'
                        else:
                            pass
                nx.set_node_attributes(G, node_color_dict, 'color')

                node_color_dict = {}
                if all_options_eua3:
                    if len(source_ind) != 0:
                        if selected_property == 'genero':
                            for node in G.nodes():
                                if G.nodes[node]['genero'] == 'f':
                                    node_color_dict[node] = '#00CC00'
                                elif G.nodes[node]['genero'] == 'm':
                                    node_color_dict[node] = '#d3d020e8'
                                elif G.nodes[node]['genero'] == 'feminino':
                                    node_color_dict[node] = '#00CC00'
                                elif G.nodes[node]['genero'] == 'masculino':
                                    node_color_dict[node] = '#d3d020e8'
                                elif G.nodes[node]['genero'] == 'Unknown':
                                    node_color_dict[node] = '#000000'
                                else:
                                    pass
                        nx.set_node_attributes(G, node_color_dict, 'color')
                        for node in G.nodes():
                            # checar se os labels não estão no 'email_unicos_ind' list
                            if G.nodes[node]['label'] not in email_unicos_ind:
                                G.nodes[node]['color'] = 'lightgray'

                # cor tempo de casa
                node_color_dict = {}
                if selected_property == 'tempoDeCasa':
                    for node in G.nodes():
                        if G.nodes[node]['tempoDeCasa'] == 'De 1 a 6 meses':
                            node_color_dict[node] = '#ca98ec'
                        elif G.nodes[node]['tempoDeCasa'] == 'De 6 meses a 1 ano':
                            node_color_dict[node] = '#ca9800'
                        elif G.nodes[node]['tempoDeCasa'] == 'Recém ingressante':
                            node_color_dict[node] = '#097c20'
                        elif G.nodes[node]['tempoDeCasa'] == 'Mais que um ano':
                            node_color_dict[node] = '#ca0000'
                        elif G.nodes[node]['tempoDeCasa'] == 'Unknown':
                            node_color_dict[node] = '#000000'
                        else:
                            pass
                nx.set_node_attributes(G, node_color_dict, 'color')

                node_color_dict = {}
                if all_options_eua3:
                    if len(source_ind) != 0:
                        if selected_property == 'tempoDeCasa':
                            for node in G.nodes():
                                if G.nodes[node]['tempoDeCasa'] == 'De 1 a 6 meses':
                                    node_color_dict[node] = '#ca98ec'
                                elif G.nodes[node]['tempoDeCasa'] == 'De 6 meses a 1 ano':
                                    node_color_dict[node] = '#ca9800'
                                elif G.nodes[node]['tempoDeCasa'] == 'Recém ingressante':
                                    node_color_dict[node] = '#097c20'
                                elif G.nodes[node]['tempoDeCasa'] == 'Mais que um ano':
                                    node_color_dict[node] = '#ca0000'
                                elif G.nodes[node]['tempoDeCasa'] == 'Unknown':
                                    node_color_dict[node] = '#000000'
                                else:
                                    pass
                        nx.set_node_attributes(G, node_color_dict, 'color')
                        for node in G.nodes():
                            # checar se os labels não estão no 'email_unicos_ind' list
                            if G.nodes[node]['label'] not in email_unicos_ind:
                                G.nodes[node]['color'] = 'lightgray'

                # cor papel
                color_mapping = {}
                if selected_property == 'papel':
                    unique_values = list(set(node['papel']
                                        for node in G.nodes().values()))
                    num_unique_values = len(unique_values)

                    # 50 cores para funçoes
                    distinct_colors = [
                        (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255,
                                                                                255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
                        (128, 0, 128), (0, 128, 128), (128, 128, 128), (255, 128,
                                                                        0), (255, 0, 128), (0, 255, 128), (128, 0, 255), (0, 128, 255),
                        (255, 128, 128), (128, 255, 128), (128, 128, 255), (255, 255,
                                                                            128), (255, 128, 255), (128, 255, 255), (192, 0, 0), (0, 192, 0),
                        (0, 0, 192), (192, 192, 0), (192, 0, 192), (0, 192, 192), (192,
                                                                                192, 192), (255, 192, 0), (255, 0, 192), (0, 255, 192),
                        (192, 0, 255), (0, 192, 255), (255, 192, 192), (192, 255, 192), (192,
                                                                                        192, 255), (255, 255, 192), (255, 192, 255), (192, 255, 255),
                        (64, 0, 0), (0, 64, 0), (0, 0, 64), (64, 64, 0), (64,
                                                                        0, 64), (0, 64, 64), (64, 64, 64), (128, 64, 0),
                        (128, 0, 64), (0, 128, 64), (64, 0, 128), (0, 64, 128), (128,
                                                                                64, 64), (64, 128, 64), (64, 64, 128), (128, 128, 64),
                        (128, 64, 128), (64, 128, 128), (0, 0,
                                                        0), (128, 128, 128), (255, 255, 255)
                    ]

                    # Normaliza o rgb 0 to 1
                    distinct_colors_norm = [(r / 255, g / 255, b / 255)
                                            for r, g, b in distinct_colors]

                    # Cria dicionario de cores
                    color_mapping = {value: colors.rgb2hex(color) for color, value in zip(
                        distinct_colors_norm, unique_values)}

                    for node in G.nodes():
                        papel = G.nodes[node]['papel']
                        if papel == 'Unknown':
                            # cinza para desconhecidos
                            G.nodes[node]['color'] = '#000000'
                        else:
                            color = color_mapping.get(papel)
                            if color is not None:
                                G.nodes[node]['color'] = color

                    if all_options_eua3:
                        if len(source_ind) != 0:
                            for node in G.nodes():
                                papel = G.nodes[node]['papel']
                                if papel == 'Unknown':
                                    # cinza para desconhecidos
                                    G.nodes[node]['color'] = '#000000'
                                else:
                                    color = color_mapping.get(papel)
                                    if color is not None:
                                        G.nodes[node]['color'] = color

                                for node in G.nodes():
                                    # checar se os labels não estão no 'email_unicos_ind' list
                                    if G.nodes[node]['label'] not in email_unicos_ind:
                                        G.nodes[node]['color'] = 'lightgray'
            # 
            if len(source) > 0:
                with col01:
                    st.write('\n')
                    st.write('\n')
                    all_options_comm = st.checkbox("Detectar comunidades", key = 'checkbox_d')

                with col02:
                    resolution = st.slider("Escolha o nível de modularidade:", min_value=0.5,
                                        max_value=1.5, step=0.1, value=1.0, disabled=not all_options_comm, key = 'slider_c')

                    if all_options_comm:
                        num_edges = G.number_of_edges()
                        if num_edges == 0:
                            color = plt.cm.hsv(0)
                            hex_color = matplotlib.colors.rgb2hex(color)
                            
                            # Assign this color to all nodes in the graph
                            nx.set_node_attributes(G, {node: hex_color for node in G.nodes()}, 'color')
                        else:
                            # Detecta comunidades
                            communities = nx.community.greedy_modularity_communities(
                                G, resolution=resolution)

                            # Gera cores distintas para as comunidades
                            num_communities = len(communities)
                            color_palette = plt.cm.hsv(
                                np.linspace(0, 1, num_communities))

                            # Cria dicionario para mapear as cores
                            community2color = {i: matplotlib.colors.rgb2hex(
                                color_palette[i]) for i in range(num_communities)}

                            # Adiciona cores para as comunidades
                            node2color = {}
                            for i, community in enumerate(communities):
                                for node in community:
                                    node2color[node] = community2color[i]

                            # atribui cores aos nós
                            nx.set_node_attributes(G, node2color, 'color')

            # pyvis no streamlit
            net = Network(directed=True, notebook=True,
                        cdn_resources='in_line', neighborhood_highlight=True)
            # net.repulsion()  # necessário para estabilizar a visualização
            # net.show_buttons(filter_= ['nodes'])
            net.set_options("""var options = {
                                "edges": {
                                    "color": {
                                    "inherit": true
                                    },
                                    "smooth": false
                                },

                                "layout": {
                                        
                                    "set_separation": 200

                                },

                                "physics": {
                                    "hierarchicalRepulsion": {
                                    "centralGravity": 0.9,
                                    "springConstant": 0.05,
                                    "nodeDistance": 200
                                    },
                                    "minVelocity": 0.75,
                                    "solver": "hierarchicalRepulsion"
                                }
                                }""")
            net.from_nx(G)

            # adiciona nós que vieram do networkx ao pyvis já com as pŕopriedades criadas e armazenadas no objeto networkx
            for node in G.nodes():
                size = G.nodes[node]['size']
                net.add_node(node, size=size)

            # criar o arquivo htlml utilizado para visualização no pyvix
            html = net.generate_html()
            with open("ona.html", mode='w', encoding='utf-8') as fp:
                fp.write(html)
                
            col111, col1123, col222 = st.columns([2.5, .5, 1])

            with col111:

                # projeta o html no streamlit
                HtmlFile = open("ona.html", 'r', encoding='utf-8')
                source_code = HtmlFile.read()
                components.html(source_code, height=650)

            # legenda
            with col1123:
                if selected_property is not None:
                    color_pairs = set((d['color'], d[selected_property])
                                    for n, d in G.nodes(data=True))

                    df_legend = pd.DataFrame(color_pairs, columns=[
                                            'color', selected_property])

                    # capitaliza as categorias para todas serem ordenadas alfabeticamente
                    df_legend.iloc[:, 1] = df_legend.iloc[:,
                                                        1].str.capitalize()

                    # Exclude color 'lightgray' when label is not 'Unknown'
                    df_legend = df_legend[~((df_legend['color'] == 'lightgray'))]

                    # Ordena alfabeticamente as categories
                    df_legend = df_legend.sort_values(by=selected_property)

                    patches = [mpatches.Patch(
                        color=row['color'], label=row[selected_property]) for idx, row in df_legend.iterrows()]
                    fig, ax = plt.subplots()
                    legend = plt.legend(
                        handles=patches, loc='upper left', prop={"size": 25})

                    # esconde os eixos do grafico
                    plt.axis('off')

                    # Display gravura streamlit
                    st.pyplot(fig)

            with col222:

                col4, col5 = st.columns([.5 ,3.5])
                with col5:
                    st.title(f"Medidas descritivas")
                    st.write("Laços:", str(num_ties))
                    st.write("Nós:", str(num_nodes))
                    st.write("Grau médio:", str(round(average_degree, 2)))
                    st.write("Densidade (%):", str(round(density, 2)*100)[:4])
                    st.write("Diâmetro:", str(diameter)) 


