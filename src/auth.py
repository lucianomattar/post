import os 
import streamlit as st
import asyncio
from httpx_oauth.clients.google import GoogleOAuth2
from dotenv import load_dotenv

load_dotenv('.env')

CLIENT_ID = os.environ['CLIENT_ID']
CLIENT_SECRET = os.environ['CLIENT_SECRET']
REDIRECT_URI = os.environ['REDIRECT_URI']
client = GoogleOAuth2(CLIENT_ID, CLIENT_SECRET)

async def get_authorization_url(client, redirect_uri):
    authorization_url = await client.get_authorization_url(redirect_uri, scope=["profile", "email"])
    authorization_url += "&prompt=select_account"
    return authorization_url

async def get_access_token(client, redirect_uri, code):
    token = await client.get_access_token(code, redirect_uri)
    return token

async def get_email(client, token):
    user_id, user_email = await client.get_id_email(token)
    return user_id, user_email

def get_login_str():
    authorization_url = asyncio.run(get_authorization_url(client, REDIRECT_URI))
    return f'''
    <div style="display: flex; justify-content: center; align-items: center; height: 100px;">
        <a style="display: inline-block; padding: 10px; color: white; background-color: #286090; 
        text-align: center; border-radius: 4px; margin-bottom: 20px; font-size: 25px;text-decoration: none;
        " target="_self" href="{authorization_url}">
            Please log in
        </a>
    </div>
    '''

def display_user():
    try:
        code = st.experimental_get_query_params()['code']
        token = asyncio.run(get_access_token(client, REDIRECT_URI, code))
        user_id, user_email = asyncio.run(get_email(client, token['access_token']))

        with open("emails.txt", "r") as file:
            allowed_emails = set(line.strip() for line in file)
        
        if user_email not in allowed_emails:
            st.markdown('<div style="text-align: center;">O usuário/email não tem permissão.</div>', unsafe_allow_html=True)
            st.session_state['authenticated'] = False
        else:
            st.session_state['authenticated'] = True
            st.session_state['email'] = user_email
            st.rerun()

    except KeyError as e:
        st.session_state['authenticated'] = False
    except Exception as e:
        pass
