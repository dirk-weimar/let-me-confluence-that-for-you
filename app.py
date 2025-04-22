import openai
import tiktoken
import pandas as pd
import numpy as np
from typing import List
import streamlit as st

# ------------- Module variables ------------- #
import module.lang as lang;
from module.shared import \
  openai_client, \
  embedding_model, \
  max_tokens_response

# ------------- Module functions ------------- #
from module.shared import \
  get_file_name_for_space, \
  create_embeddings
from module.collect_data import write_csv
from module.collect_data import get_confluence_spaces

# ------------------ Config ------------------ #
completion_model    = 'gpt-3.5-turbo'
file_name           = 'data/pages_data'
user_language       = 'english' # german
user_avatar         = '👨‍💻️'
assistant_avatar    = '🦉'

# Config messages
# With gpt-3.5-turbo, we can put 4096 tokens into a message
# A message with activated memory typically looks like this:
# ------------------------------------------------------------------------------
# (1) {"role": "system", "content": system message}  150  tokens_system_message
# (2) {"role": "assistant", "content": context}     1720  max_tokens_per_context
# (3) {"role": "user", "content": query}             100
# (4) {"role": "assistant", "content": response}     300  max_tokens_response
# (5) {"role": "assistant", "content": context}     1720  max_tokens_per_context
# (6) {"role": "user", "content": query}             100
# ------------------------------------------------------------------------------
# ...

# Change to commented values to activate memory of previous question & answer
max_tokens_per_context = 3390   # 1720
max_messages_to_keep = 3        # 6

# Set language
if user_language  == 'german':
    lang_data = lang.german
else:
    lang_data = lang.english


# ------------------ Helper ------------------ #
def get_num_tokens_from_string(string: str, encoding_name: str) -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def vector_similarity(x: List[float], y:List[float]) -> float:
    return np.dot(np.array(x), np.array(y))

def parse_numbers(s: str) -> List[float]:
  return [float(x) for x in s.strip('[]').split(',')]

def get_avatar(role):
    avatars = {
        'user': user_avatar,
        'assistant': assistant_avatar,
        'system': '⚙️'
    }
    return avatars[role]


# ----------------- Functions ---------------- #
def initialize_memory() -> List[tuple]:
    memory = []
    memory.append({"role": "system", "content": lang_data['system_message']})
    return (memory)

def read_csv(confluence_spaces: List[str], file_name: str) -> pd.DataFrame:
    csv_files = []
    data_frames = []

    for space in confluence_spaces:
        file_name_space = get_file_name_for_space(file_name, space)
        csv_files.append(file_name_space)

    for file in csv_files:
        df = pd.read_csv(file, dtype = {'embeddings': object})
        data_frames.append(df)

    data_frame = pd.concat(data_frames, axis = 0)
    data_frame['embeddings'] = data_frame['embeddings'].apply(lambda x: parse_numbers(x))

    return data_frame

def ask(query: str, memory: List[tuple], pages_df: pd.DataFrame):

    # Sort pages by similarity of the embeddings with the query
    pages_df_sorted = sort_documents(query, pages_df)

    # Get context for query
    context = get_context(query, pages_df_sorted)

    # Construct the prompt
    memory = contruct_prompt(query, memory, context)

    # Call api to get answer
    response = openai_client.chat.completions.create(
        model = completion_model,
        messages = memory,
        max_tokens = max_tokens_response,
        temperature = 0,
        stream = True
    )

    return response

def sort_documents(query: str, pages_df: pd.DataFrame) -> pd.DataFrame:

    query_embedding = create_embeddings(query, model = embedding_model)
    pages_df['similarity'] = pages_df['embeddings'].apply(lambda x: vector_similarity(x, query_embedding))
    pages_df.sort_values(by = 'similarity', inplace = True, ascending = False)
    pages_df.reset_index(drop = True, inplace = True)

    return pages_df

def get_context(query: str, pages_df: pd.DataFrame) -> pd.DataFrame:

    chosen_pages_links = []
    chosen_pages_content = []
    sum_tokens_chosen = 0


    for i in range(len(pages_df)):

        page = pages_df.loc[i]

        # Add context until we run out of space.
        if sum_tokens_chosen + page.num_tokens > max_tokens_per_context:
            break

        chosen_pages_content.append(page.page_content)
        chosen_pages_links.append(page.space + ': [' + page.title +']' + '(' + page.link + ')')

        sum_tokens_chosen += page.num_tokens + 4  # separator

    # Print info message
    st.sidebar.write('### ' + lang_data['used_confluence_pages'])
    st.sidebar.markdown("* " + "\n* ".join(chosen_pages_links))

    context = lang_data['context_message'] + '\n\n' . join(chosen_pages_content)

    return context

def contruct_prompt(query: str, memory: List[tuple], context: str) -> List[tuple]:

    memory.append({"role": "assistant", "content": context})
    memory.append({"role": "user", "content": query})

    st.session_state.messages_history.append({"role": "user", "content": query})

    # Keep [max_messages_to_keep] messages in memory to stay within max_tokens
    # Always keep first system message (i.e. delete messages 2, 3 and 4)
    if len(memory) > max_messages_to_keep:
        del memory[1:4]

    return (memory)


# --------------------- App --------------------- #
st.title('🦉 Let Me Confluence That For You!')

# Custom CSS - make avatars a bit bigger
custom_css = """
<style>
.stChatMessage div:first-child {
    border: none;
    background-color: inherit;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html = True)


# ------------------- Sidebar ------------------- #
st.sidebar.write('### ' + lang_data["configuration"])
confluence_spaces = get_confluence_spaces()
selected_confluence_names = st.sidebar.multiselect(lang_data["confluence_spaces"], options=[name for key, name in confluence_spaces])


# Create a dictionary to map readable names to IDs
confluence_dict = {name: key for key, name in confluence_spaces}

# Initialize selected confluence keys
if "selected_confluence_keys" not in st.session_state:
    st.session_state.selected_confluence_keys = []

# Retrieve the IDs from the selected readable names
st.session_state.selected_confluence_keys = [confluence_dict[name] for name in selected_confluence_names]


# --------------------- Main -------------------- #
if not st.session_state.selected_confluence_keys:
    st.info(lang_data["please_choose_confluence_spaces"])

else:
    try:
        pages_df = read_csv(st.session_state.selected_confluence_keys, file_name)

        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = initialize_memory()
        if "messages_history" not in st.session_state:
            st.session_state.messages_history = initialize_memory()

        # Display chat messages from history on app rerun
        for message in st.session_state.messages_history:
            with st.chat_message(message["role"], avatar = get_avatar(message["role"])):
                if not message['content'].startswith(lang_data['context_message']):
                    st.markdown(message["content"])

        if prompt := st.chat_input(lang_data["ask_a_question"]):

            # Print current user promt
            message = st.chat_message("user", avatar = user_avatar)
            message.write(prompt)

            # Assistant Promt
            with st.chat_message("assistant", avatar = assistant_avatar):
                message_placeholder = st.empty()
                full_response = ""

                # Get answer from AI
                response = ask(prompt, st.session_state.messages, pages_df)

                # Print answer to the user
                for chunk in response:

                    delta = chunk.choices[0].delta
                    if delta.content:
                        chunk_content = chunk.choices[0].delta.content
                        full_response += chunk_content
                        message_placeholder.markdown(full_response + "▌")

                message_placeholder.markdown(full_response)

                st.session_state.messages.append({"role": "assistant", "content": full_response})
                st.session_state.messages_history.append({"role": "assistant", "content": full_response})


    except FileNotFoundError:
        st.info(lang_data["load_data_first"])

        if st.button(lang_data["load_confluence_data"]):
            write_csv(st.session_state.selected_confluence_keys, file_name)
            st.rerun()
