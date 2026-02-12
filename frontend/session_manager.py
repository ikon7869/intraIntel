import streamlit as st
import uuid

def init_session_state():
    """
    Initializes necessary variables in Streamlit's session state.
    This ensures data persists across reruns of the Streamlit app.
    """

    if "messages" not in st.session_state:
        st.session_state.messages = []
    

    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

        st.session_state.messages.append({"role": "assistant", "content": "Hello! How can I help you today?"})


    if "web_search_enabled" not in st.session_state:
        st.session_state.web_search_enabled = True