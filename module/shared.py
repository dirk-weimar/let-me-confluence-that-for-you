from openai import OpenAI
import streamlit as st

# ------------- Shared variables ------------- #
openai_client = OpenAI(api_key = st.secrets['OPENAI_API_KEY'])
tokenizer_encoding_name = 'cl100k_base'
embedding_model = 'text-embedding-ada-002'


# ------------------ Config ------------------ #
max_tokens_response         = 400   # adjust for longer or shorter answers from the chatbot
tokens_system_message       = 150   # adjust if you change system message
tokens_context_message      = 15    # adjust if you change context message
tokens_meta_infos           = 135   # for example 'role: system' in chat messages

max_tokens_completion_model = 4096  # adjust if you change completion model

max_num_tokens = max_tokens_completion_model  \
  - max_tokens_response \
  - tokens_system_message \
  - tokens_context_message \
  - tokens_meta_infos


# ------------- Shared functions ------------- #
def get_file_name_for_space(file_name: str, space: str) -> str:
    return file_name + '_' + space + '.csv'

def create_embeddings(text: str, model: str) -> list:

    response = openai_client.embeddings.create(
        model="text-embedding-ada-002",  # oder ein anderes Modell
        input=text
    )
    return response.data[0].embedding
