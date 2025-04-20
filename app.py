import streamlit as st
import requests
import config

st.set_page_config(page_title="Conversational Q&A Bot", layout="wide")
st.title("Conversational Q&A Bot")

# Session state setup
if "pdf_uploaded" not in st.session_state:
    st.session_state["pdf_uploaded"] = False
if "active_mode" not in st.session_state:
    st.session_state["active_mode"] = None  # None | "sentiment" | "chatbot"
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "sentiment_result" not in st.session_state:
    st.session_state["sentiment_result"] = None

# Sidebar
with st.sidebar:
    st.markdown("### Upload a PDF")
    file_object = st.file_uploader("Upload PDF", label_visibility="collapsed")

    if file_object is not None and not st.session_state["pdf_uploaded"]:
        with st.spinner("Uploading PDF to backend..."):
            files = {"file": (file_object.name, file_object.read())}
            response = requests.post(f"{config.BACKEND_URL}/process_pdf/", files=files)
            if response.status_code == 200:
                st.success("PDF uploaded successfully!")
                st.session_state["pdf_uploaded"] = True
            else:
                st.error("Error uploading PDF. Please try again.")

    # Show mode buttons only after successful upload and when no mode is active
    if st.session_state["pdf_uploaded"] and st.session_state["active_mode"] is None:
        st.markdown("### Choose Action")
        if st.button("Sentiment"):
            st.session_state["active_mode"] = "sentiment"
        if st.button("Chatbot"):
            st.session_state["active_mode"] = "chatbot"

# Main content logic
if not st.session_state["pdf_uploaded"]:
    st.markdown("#### 👋 Welcome to the PDF Bot!")
    st.markdown("""
        Upload a PDF using the sidebar to get started.

        While you're here, did you know?

        > **Fun Fact:** The first PDF was created by Adobe in 1993!

        Or maybe go grab a coffee ☕ while we wait?
    """)
elif st.session_state["active_mode"] == "sentiment":
    st.subheader("Sentiment Analysis")
    files = {"file": (file_object.name, file_object.read())}
    sentiment_response = requests.post(f"{config.BACKEND_URL}/sentiment/", files=files)
    st.session_state["sentiment_result"] = sentiment_response.text

    if st.session_state["sentiment_result"]:
        st.write("Sentiment Result:", st.session_state["sentiment_result"])
        if st.button("❌ Close Sentiment"):
            st.session_state["active_mode"] = None
            st.session_state["sentiment_result"] = None
            st.rerun()

elif st.session_state["active_mode"] == "chatbot":
    st.subheader("Chat with the PDF")

    # Cancel button for chatbot
    if st.button("❌ Close Chatbot"):
        st.session_state["active_mode"] = None
        st.session_state["chat_history"] = []
        st.rerun()

    def send_message(message: str, history: list):
        payload = {"message": message, "history": history}
        response = requests.post(f"{config.BACKEND_URL}/chat/", json=payload)
        response_data = response.json()
        return response_data.get('response', "Error: No 'response' key found in backend data.")

    # Display chat history
    for message in st.session_state["chat_history"]:
        with st.chat_message("user"):
            st.markdown(message[0])
        with st.chat_message("assistant"):
            st.markdown(message[1])

    user_input = st.chat_input("Ask your question:")

    if user_input:
        st.session_state["chat_history"].append([user_input])
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.spinner("Thinking..."):
            assistant_response = send_message(user_input, st.session_state["chat_history"][:-1])

        with st.chat_message("assistant"):
            st.markdown(assistant_response)

        st.session_state["chat_history"][-1].append(assistant_response)
        st.rerun()