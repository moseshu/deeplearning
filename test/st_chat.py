import streamlit as st
import random
import time
from streamlit_chat import message as chat_msg

with st.sidebar:
    st.title("😊 ☁️Moses LLM Chat App")
    st.markdown(
        """ 
        ## About
        This App is an LLM-powered chatbot build using
        - [Streamlit](https://streamlit.io)
        - [Langchain](https://python.langchain.com/)
        """
    )
    st.markdown(
        """
        ### Prams
    """)
    temprature = st.slider("Temprature", min_value=0.1, value=0.7, max_value=1.0, step=0.1)
    top_k = st.slider("Top_K", min_value=1, value=40, max_value=100, step=1)
    top_p = st.slider("Top_P", min_value=0.1, value=0.9, max_value=1.0, step=0.01)
    prenet = st.slider("leng pre", min_value=1.0, value=1.1, max_value=2.0, step=0.1)
    do_sample = st.checkbox('do_sample', value=True)
    st.write("Model with ❤️ by Moses")
    # st.button("Clear History")
    reset_button_key = "reset_button"
    reset_button = st.button("Reset Chat", key=reset_button_key)
    if reset_button:
        st.session_state.messages = []
        # st.session_state.chat_history = None
st.title("ChatBot:ship:")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

def predict(prompt):
    assistant_response = random.choice(
        [
            "你好hello what",
            "我需要 你，的帮助",
            "吃饭了吗",
        ]
    )
    # Simulate stream of response with milliseconds delay
    assistant_response = f"{assistant_response}"
    for i in assistant_response.split():
        yield i
# Display chat messages from history on app rerun

for i in range(len(st.session_state.messages)):
    message = st.session_state.messages[i]
    if message['role'] == "user":
        chat_msg(message["content"], is_user=True, key=f"{i}_user")
    elif message['role'] == 'assistant':
        chat_msg(message['content'], allow_html=True, key=f"{i}")


def myform():

    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_area("You:", key='input', height=100)
        submit_button = st.form_submit_button(label='Send')

    if submit_button and user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        chat_msg(user_input, is_user=True, key="user")
        resp = predict(user_input)
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            for chunk in resp:
                full_response += chunk + " "
                time.sleep(0.1)
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})

def chat():
    if prompt := st.chat_input("What is up?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message in chat message container
        chat_msg(prompt, is_user=True, key="user")
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            resp = predict(prompt)
            for chunk in resp:
                full_response += chunk + " "
                time.sleep(1)
                # Add a blinking cursor to simulate typing
                message_placeholder.markdown(full_response + "▌")

            message_placeholder.markdown(full_response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})


def main():
    # Accept user input
    # myform()
    chat()


if __name__ == '__main__':
    main()

