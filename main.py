# streamlit run main.py
import os
import streamlit as st
from dotenv import load_dotenv
from vertexai.preview.generative_models import GenerativeModel, Tool
import vertexai
# Load environment variables
load_dotenv()

# Configure Streamlit page settings
st.set_page_config(
    page_title="Chat with Gemini-Pro!",
    page_icon=":brain:",  # Favicon emoji
    layout="centered",  # Page layout option
)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PROJECT_ID = os.getenv("PROJECT_ID")

# Set up Google Gemini-Pro AI model
vertexai.init(project=PROJECT_ID, location="us-central1")
from vertexai.preview.generative_models import grounding as preview_grounding

DATA_STORE_PROJECT_ID = PROJECT_ID  # @param {type:"string"}
DATA_STORE_REGION = "global"  # @param {type:"string"}
DATA_STORE_ID = os.getenv("DATA_STORE_ID")

t = Tool.from_retrieval(
    preview_grounding.Retrieval(
        preview_grounding.VertexAISearch(  # loading 빠르다
            datastore=DATA_STORE_ID,
            project=DATA_STORE_PROJECT_ID,
            location=DATA_STORE_REGION,
            )
        )
    )
    # return rag.VertexRagStore( # loading 느린듯
    #     rag_resources=[
    #         rag.RagResource(
    #             rag_corpus=rag_corpus_name,  # Currently only 1 corpus is allowed.
    #                     # Optional: supply IDs from `rag.list_files()`.
    #                     # rag_file_ids=["rag-file-1", "rag-file-2", ...],
    #                 )
    #             ],
    #             similarity_top_k=3,  # Optional
    #             vector_distance_threshold=0.5,  # Optional
    #         )

system_instruction=[
    "You are a helpful support agent. Your task is to answer about Brad Yoo."
]
# System instructions https://cloud.google.com/vertex-ai/generative-ai/docs/reference/python/latest#system-instructions
# model = GenerativeModel(model_name='gemini-pro', 
# model = GenerativeModel(model_name='textembedding-gecko',
model = GenerativeModel(model_name='gemini-1.0-pro',
# model = GenerativeModel(model_name='gemini-1.5-flash',
                        tools=[t],
                        system_instruction=system_instruction
                        )

# Function to translate roles between Gemini-Pro and Streamlit terminology
def translate_role_for_streamlit(user_role):
    if user_role == "model":
        return "assistant"
    else:
        return user_role


# Initialize chat session in Streamlit if not already present
if "chat_session" not in st.session_state:
    st.session_state.chat_session = model.start_chat(history=[])


# Display the chatbot's title on the page
st.title("🤖 Brad Yoo's chatbot")
st.caption("Start by asking 'Where was Brad Yoo born?'. This is a Brad's Chatbot based on https://storage.googleapis.com/cloudrun-bradtalk/20240318_brad_eng.pdf")
st.chat_message("assistant").markdown("Hello there! I am Brad Yoo. How may I help?")

# Display the chat history
for message in st.session_state.chat_session.history:
    with st.chat_message(translate_role_for_streamlit(message.role)):
        st.markdown(message.parts[0].text)

# Input field for user's message
user_prompt = st.chat_input("Talk to chatbot...")

if user_prompt:
    # Add user's message to chat and display it
    st.chat_message("user").markdown(user_prompt)

    # Send user's message to Gemini-Pro and get the response
    gemini_response = st.session_state.chat_session.send_message(user_prompt)

    # Display Gemini-Pro's response
    with st.chat_message("assistant"):
        st.markdown(gemini_response.text)