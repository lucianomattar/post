import sys
import importlib
import streamlit as st
from auth import *
import base64

########################################## layout
st.set_page_config(layout="wide")

#retira botao deploy streamlit e footer
st.markdown("""
    <style>
        MainMenu {visibility: hidden;}
        .stDeployButton {display:none;}
        footer {visibility: hidden;}
        stDecoration {display:none;}
    </style>
""", unsafe_allow_html=True)

#barra lateral
st.markdown("""
    <style>
      section[data-testid="stSidebar"] {
        top: 5%;  # Adjust this value to increase or decrease whitespace at the top
        height: 80% !important;
        z-index: 1000;
      }
    </style>""", unsafe_allow_html=True)

#logo e faixa a3data
with open("logo_id.png", "rb") as f:
    data = base64.b64encode(f.read()).decode("utf-8")
    st.markdown(
        f"""
        <div style="position: fixed; top: 46px; left: 0px; width: 100%; height: 60px; background-color: #f1f1f1; 
            border-bottom: 1px solid #ccc; border-top: 1px solid #ccc; z-index: 10000;">
            <div style="position: absolute; top: 15px; left: 25px; z-index: 1001;">
                <img src="data:image/png;base64,{data}" width="132.73" height="25">
            </div>
        </div>
        """, 
        unsafe_allow_html=True,
    )
            
##########################################

if __name__ == '__main__':
    
    if not st.session_state.get('authenticated'):        
        st.markdown("""
            <style>
                .shadow-box {
                    border: 1px solid #ddd;
                    padding: 20px;
                    border-radius: 5px;
                    box-shadow: 2px 2px 12px rgba(0, 0, 0, 0.1);
                    background-color: #fff;
                    width: 80%; /* Adjust the width as per requirement */
                    max-width: 300px; /* Maximum width */
                    margin: 20px auto; /* Auto margins for horizontal centering */
                }
        
                /* Media query for smartphones */
                @media (max-width: 768px) {
                    .shadow-box {
                        width: 90%; /* Adjust the width for smaller devices */
                    }
                }
            </style>
        """, unsafe_allow_html=True)


        st.markdown("""
            <div class="shadow-box">
                <h1 style='text-align: center; color: #3c6bbb;font-weight: normal;'>ONA A3data</h1>
                {}
            """.format(get_login_str()), unsafe_allow_html=True)
        display_user()

    else:
        option = st.sidebar.radio(
            'Selecione:',
            ('O que é a ONA', 'Redes Sociais', 'Análises de grupo', 
            'Análises de indivíduos', 'Análises cruzadas', 'Análises de papéis', 'Informações')
        )
        
        user_email = st.session_state.get('email', 'Unknown User')
        st.sidebar.markdown(f'Usuário: **{user_email}**')

        # Separate logout button
        if st.sidebar.button('Logout'):
            st.session_state['authenticated'] = False
            st.rerun()
       
        if option == 'O que é a ONA':
            oqueeona = importlib.import_module("1O que é a ONA")
            oqueeona.run()
        elif option == 'Redes Sociais':
            redessociais = importlib.import_module("2Redes sociais")
            redessociais.run()
        elif option == 'Análises de grupo':
            analisesdegrupo = importlib.import_module("3Análises de grupo")
            analisesdegrupo.run()
        elif option == 'Análises de indivíduos':
            analisesdeindividuos = importlib.import_module("4Análises de indivíduos")
            analisesdeindividuos.run()
        elif option == 'Análises cruzadas':
            analisescruzadas = importlib.import_module("5Análises cruzadas")
            analisescruzadas.run()
        elif option == 'Análises de papéis':
            analisepapeis = importlib.import_module("6Análises de papéis")
            analisepapeis.run()
        elif option == 'Informações':
            informacoes = importlib.import_module("7Informações")
            informacoes.run()
