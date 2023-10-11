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
            "Hello there! How can I assist you today?",
            "Hi, human! Is there anything I can help you with?",
            "Do you need help?",
        ]
    )
    # Simulate stream of response with milliseconds delay
    assistant_response = f"{assistant_response}\ntmp:{temprature}\ntop_p{top_p}\ndo_sample{do_sample}"
    for i in assistant_response.split():
        yield i
# Display chat messages from history on app rerun
for i in range(len(st.session_state.messages)):
    message = st.session_state.messages[i]
    # with st.chat_message(message["role"]):
        # st.markdown(message["content"])
    if message['role'] == "user":
        chat_msg(message["content"], is_user=True, key=f"{i}_user")

    elif message['role'] == 'assistant':
        chat_msg(message['content'], allow_html=True, key=f"{i}")

# Accept user input
def main():
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
                time.sleep(0.1)
                # Add a blinking cursor to simulate typing
                message_placeholder.markdown(full_response + "▌")

            message_placeholder.markdown(full_response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})


if __name__ == '__main__':
    main()

