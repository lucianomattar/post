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
            ["Redes Papéis", "Papéis de destaque", "Indivíduos de destaque"])  # cria as tabs
        
    with tab1:

        # cria o driver python
        driver = GraphDatabase.driver(
            NEO4J_URL, auth=(NEO4J_USER, NEO4J_PASSWORD))

        def perform_eua3(session, dimensao, nome, rodada, email):

            main_query = f"""
                MATCH (n)-[r:ATUA_COMO_{rodada}]->(papel:Papel {{nome:$nome}})
                WHERE n.email IS NOT NULL AND n.email <> "" AND n.email IN $email
                WITH collect(n) AS PapelNodes, papel
                UNWIND PapelNodes as n1

                OPTIONAL MATCH (n1)-[r2 {{rodada: $rodada, dimensao: $dimensao}}]->(n2)
                WHERE (n2)-[:ATUA_COMO_{rodada}]->(papel) OR n2 IS NULL

                RETURN DISTINCT n1, r2, n2, id(n1) AS n1_id, id(n2) AS n2_id
            """
            def read_tx(tx):
                result = tx.run(main_query, email=email, nome=nome, rodada=rodada, dimensao=dimensao)
                records = list(result)
                summary = result.consume()
                return records, summary

            records, summary = session.execute_read(read_tx)
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
                rod = st.selectbox("Selecione a rodada", rod_g, key = 'papel_a')
                
            with col4:
                query_dimensao = f"MATCH (n)-[r]->(n2) WHERE r.rodada = '{rod}' RETURN DISTINCT r.dimensao AS r_dim"
                dim_g = [record["r_dim"]for record in session.run(query_dimensao)]
                dim = st.selectbox("Selecione a dimensão", sorted(dim_g), key='papel_b')

            with col1b:
                query_cg_3 = f"MATCH ()-[r: ATUA_COMO_{rod}]->(g:Papel) RETURN DISTINCT g.nome AS g_nome"
                rede_list_Cliente = [record["g_nome"]for record in session.run(query_cg_3)]
                nome = st.selectbox("Selecione o papel:", sorted(rede_list_Cliente, key=lambda s: s.lower()), key='papel_c')

            with col1:
            
                e_list = [
                    (record["c1_email"], record["c2_email"])
                    for record in session.run(
                        f"""
                        MATCH (n)-[r:ATUA_COMO_{rod}]->(:Papel {{nome:$nome}})
                        WITH collect(n) AS PapelNodes
                        UNWIND PapelNodes as c1
                        UNWIND PapelNodes as c2
                        OPTIONAL MATCH (c1)-[b1{{rodada: $rodada, dimensao: $dimensao}}]->(c2) 
                        RETURN DISTINCT c1.email AS c1_email, c2.email AS c2_email
                        """,
                        dimensao=dim,
                        rodada=rod,
                        nome=nome
                    )]          

                email_list = {name for tuple_ in e_list for name in tuple_ if name is not None}

                source = st.multiselect(
                    "Selecione um ou mais EuA3:", sorted(email_list), key='papel_e')
                source_ind = source

            with col2:   
                st.write('\n')
                st.write('\n')   
                all_options_eua3 = st.checkbox("Selecione todos os EuA3", key = 'checkbox_papel_3')
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
                                                'Selecione uma característica', 'Gênero', 'Tempo de casa'], key = 'papel_propriedade_b')
                if selected_property == 'Gênero':
                    selected_property = 'genero'
                elif selected_property == 'Tempo de casa':
                    selected_property = 'tempoDeCasa'
                else:
                    selected_property = None 

            G = nx.DiGraph()

            for record in records:
                source_id = record["n1"].element_id
                G.add_node(source_id, label=record["n1"]["email"], genero=record["n1"]["genero"], tempoDeCasa=record["n1"]["tempoDeCasa"], idade=record["n1"]["idade"])

                if record["n2"] is not None:  # Checar se n2 não é none
                    target_id = record["n2"].element_id
                    G.add_node(target_id, label=record["n2"]["email"], genero=record["n2"]["genero"], tempoDeCasa=record["n2"]["tempoDeCasa"], idade=record["n2"]["idade"])
                    
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
                    'Selecione uma medida de centralização:', list(centrality_measures.keys()), key = 'papel_select')

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
                    i for i in range(0, len(sorted_nodes) + 1)], key = 'mais_centrais papel c')

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
            
            # 
            if len(source) > 0:
                with col01:
                    st.write('\n')
                    st.write('\n')
                    all_options_comm = st.checkbox("Detectar comunidades", key = 'checkbox_e')

                with col02:
                    resolution = st.slider("Escolha o nível de modularidade:", min_value=0.5,
                                        max_value=1.5, step=0.1, value=1.0, disabled=not all_options_comm, key = 'slider_d')

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
  
##########################################
# papeis de destaque
##########################################
    with tab2:

        col1, col2 = st.columns([1, 3])

        st.markdown("<br>"*2, unsafe_allow_html=True)

        col3, col4 = st.columns([1, 3])

        with col1:
            # cria o driver python
            driver = GraphDatabase.driver(
            	NEO4J_URL, auth=(NEO4J_USER, NEO4J_PASSWORD))

            def perform_eua3(session, dimensao, rodada):
                main_query = f"""
                    MATCH (n)-[r:ATUA_COMO_{rodada}]->(papel)
                    WHERE n.email IS NOT NULL
                    WITH collect(n) AS PapelNodes, papel
                    UNWIND PapelNodes as n1

                    OPTIONAL MATCH (n1)-[r2 {{rodada: $rodada, dimensao: $dimensao}}]->(n2)
                    WHERE (n2)-[:ATUA_COMO_{rodada}]->(papel) OR n2 IS NULL

                    RETURN DISTINCT n1, r2, n2, id(n1) AS n1_id, id(n2) AS n2_id, papel
                """
                def read_tx(tx):
                    result = tx.run(main_query, rodada=rodada, dimensao=dimensao)
                    records = list(result)
                    summary = result.consume()
                    return records, summary

                records, summary = session.execute_read(read_tx)
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
                    rod = st.selectbox("Selecione a rodada:",rodada1_list, key='selectbox_rod_papel')

                    dimension_list = [record["b1_dimensao"] for record in session.run(
                        f"MATCH ()-[b1:BUSCOU]->() RETURN DISTINCT b1.dimensao AS b1_dimensao")]
                    dim = st.selectbox("Selecione a dimensão:", sorted(dimension_list), key='selectbox_dim_papel')
                     

                # executa a query apos construida todas as variaveis de entrada
                records, summary = perform_eua3(session, dimensao=dim, rodada=rod)

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

                records, summary = perform_eua3(session, dimensao=dim, rodada=rod)

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

                query_cg_2 = f"MATCH (n1:Colaborador)-[:ATUA_COMO_{rod}]->(papel) RETURN DISTINCT papel.nome AS papel_nome"
                rede_list_papel = [record["papel_nome"]
                                    for record in session.run(query_cg_2)]
                query = f"MATCH (e)-[:ATUA_COMO_{rod}]->(papel) RETURN DISTINCT e.email AS email, papel.nome AS papel_nome"
                papel = [(record["email"], record['papel_nome'])
                        for record in session.run(query)]
                # cria lista para usar para criar propriedades de cliente
                modified_list_c = []
                for sublist in papel:
                    modified_dict = {
                        "email": sublist[0],
                        "cliente_guilda": sublist[1]
                    }
                    modified_list_c.append(modified_dict)

                G = nx.DiGraph()

                # pega todas as caracteristicas laços e nodos de 'cada vez'
                # adiciona nodos e laços a partir dos resultados do query da rede neo4j
                for record in records:
                    source_id = record["n1"].element_id
                    G.add_node(source_id, label=record["n1"]["email"])

                    if record["n2"] is not None:  # Checar se n2 não é none
                        target_id = record["n2"].element_id
                        G.add_node(target_id, label=record["n2"]["email"])
                        
                        if record["r2"] is not None:  # Checar se r2 não é none
                            G.add_edge(source_id, target_id, label=record["r2"])

                graph_dict = {}
                original_G3 = G.copy()

                for papel in rede_list_papel:
                    G3 = original_G3.copy() 
                    for i in range(len(modified_list_c)):
                        if modified_list_c[i]['cliente_guilda'] == papel:
                            for node in G3.nodes():
                                if G3.nodes[node]['label'] == modified_list_c[i]['email']:
                                    G3.nodes[node]['color'] = '#11d63f'
                                    G3.nodes[node]['papel'] = papel

                    nodes_to_remove = [node for node, data in G3.nodes(
                        data=True) if data.get('color') != '#11d63f']
                    G3.remove_nodes_from(nodes_to_remove)
                    graph_dict[papel] = G3

                centrality_dict_c = {}
                node_count_dict = {}

                with col1:
                    centrality = st.selectbox(
                        'Selecione uma medida de centralização:', [
                            'Degree', 'Betweenness', 'Closeness', 'Page rank',
                            'Densidade'
                        ], key = 'centrality = bcpapel')
                    if centrality == 'Degree':
                        for papel, graph in graph_dict.items():
                            measure_dict = nx.in_degree_centrality(graph)
                            centrality_dict_c[papel] = pd.Series(measure_dict).mean()
                            node_count_dict[papel] = graph.number_of_nodes()
                    elif centrality == 'Betweenness':
                        for papel, graph in graph_dict.items():
                            measure_dict = nx.betweenness_centrality(graph)
                            centrality_dict_c[papel] = pd.Series(measure_dict).mean()
                            node_count_dict[papel] = graph.number_of_nodes()
                    elif centrality == 'Betweenness':
                        for papel, graph in graph_dict.items():
                            measure_dict = nx.betweenness_centrality(graph)
                            centrality_dict_c[papel] = pd.Series(measure_dict).mean()
                            node_count_dict[papel] = graph.number_of_nodes()
                    elif centrality == 'Closeness':
                        for papel, graph in graph_dict.items():
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
                            centrality_dict_c[papel] = pd.Series(closeness_centrality).mean()
                            node_count_dict[papel] = graph.number_of_nodes()
                    elif centrality == 'Page rank':
                        for papel, graph in graph_dict.items():
                            measure_dict = nx.pagerank(graph)
                            centrality_dict_c[papel] = pd.Series(measure_dict).mean()
                            node_count_dict[papel] = graph.number_of_nodes()
                    elif centrality == 'Densidade':
                        for papel, graph in graph_dict.items():
                            measure_dict = nx.density(graph)
                            centrality_dict_c[papel] = pd.Series(measure_dict).mean()
                            node_count_dict[papel] = graph.number_of_nodes()

                    node_count_series = pd.Series(node_count_dict)

                    df = pd.DataFrame(centrality_dict_c.items(), columns=['papel', 'Measure'])
                    df = df.loc[df['papel'] != 'Unknown']
                    df = df.sort_values('Measure', ascending=True)

                    df['Num_nodes'] = df['papel'].map(node_count_series)
                    df = df.loc[~((df['Num_nodes'] == 1) & (df['Measure'] == 1))]# exclui o caso de redes com 1 (total de medida) e 1 nós apenas.

                    fig, ax = plt.subplots()

                    with col2:
                        chart = alt.Chart(df.reset_index()).mark_bar().encode(
                            x=alt.X('Measure:Q', title='Centrality'),
                            y=alt.Y('papel:N', sort='-x', title=' '),
                            tooltip=[alt.Tooltip('papel:N'), alt.Tooltip('Measure:Q'), alt.Tooltip('Num_nodes:Q')]
                        ).properties(
                            title='Papéis de destaque'
                        ).configure_axisY(
                            labelLimit=200  # Increase this value to allow for wider labels on the y-axis
                        ).interactive()

                        st.altair_chart(chart, use_container_width=True)

######################################
# individuos de destaque
######################################

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
                    MATCH (n)-[r:ATUA_COMO_{rodada}]->(papel:Papel {{nome: $nome}})
                    WHERE n.email IS NOT NULL
                    WITH collect(n) AS PapelNodes, papel
                    UNWIND PapelNodes as n1

                    OPTIONAL MATCH (n1)-[r2 {{rodada: $rodada, dimensao: $dimensao}}]->(n2)
                    WHERE (n2)-[:ATUA_COMO_{rodada}]->(papel) OR n2 IS NULL

                    RETURN DISTINCT n1, r2, n2, id(n1) AS n1_id, id(n2) AS n2_id, papel
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

                    query_cg_3 = f"MATCH ()-[: ATUA_COMO_{rod}]->(p:Papel) RETURN DISTINCT p.nome AS p_nome"
                    rede_list_papel = [record["p_nome"]for record in session.run(query_cg_3) if record["p_nome"] != "Unknown"]
                    nome = st.selectbox("Selecione o papel:", sorted(rede_list_papel))

                    records, summary = get_eua3(session, nome=nome, dimensao=dim, rodada=rod)

                    G = nx.DiGraph()

                    for record in records:
                        source_id = record["n1"].element_id
                        G.add_node(source_id, label=record["n1"]["email"])

                        if record["n2"] is not None:  # Checar se n2 não é none
                            target_id = record["n2"].element_id
                            G.add_node(target_id, label=record["n2"]["email"])
                            
                            if record["r2"] is not None:  # Checar se r2 não é none
                                G.add_edge(source_id, target_id, label=record["r2"])

                    centrality = st.selectbox(
                        'Selecione uma medida de centralidade:',
                        ['InDegree', 'OutDegree', 'Betweenness', 'Closeness', 'Page rank'], key='centrality abcd papel')

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
                        )

                        st.altair_chart(chart, use_container_width=True)

   
