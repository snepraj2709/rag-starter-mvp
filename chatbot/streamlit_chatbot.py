import streamlit as st
import time
from statefull_bot import get_response

st.set_page_config(page_title="RAG Chatbot for Finance Analysts")
st.title("A RAG Chatbot for Finance Analysts")
st.caption("Note: this MVP uses my OPENAI API key and may error if the quota/limit is reached.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        { "role": "assistant", "question": "Ask me anything about the SEBI Annual report 2024-25"}
    ]

# Render message history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["question"])

# User Input
if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({ "role": "human","question": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call get_response and pass chat history
    try:
        raw_response = get_response(prompt, chat_history=st.session_state.messages)
    except Exception as e:
        err = f"Error while getting response: {e}"
        with st.chat_message("assistant"):
            st.markdown(err)
        st.session_state.messages.append({"question": err})
    else:
        if isinstance(raw_response, dict):
            response_text = (
                raw_response.get("result")
                or raw_response.get("answer")
                or str(raw_response)
            )
        else:
            response_text = str(raw_response)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            displayed = ""
            for word in response_text.split():
                displayed += word + " "
                placeholder.markdown(displayed + "▌")
                time.sleep(0.02)  # simulate streaming
            placeholder.markdown(displayed)

        # Add assistant response to chat history
        st.session_state.messages.append({ "role": "assistant", "question": displayed})
