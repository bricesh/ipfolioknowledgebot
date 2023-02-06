import streamlit as st
import pandas as pd
import openai
import numpy as np
import tiktoken

MAX_SECTION_LEN = 3000
SEPARATOR = "\n* "
ENCODING = "cl100k_base"  # encoding for text-embedding-ada-002
COMPLETIONS_MODEL = "text-davinci-003"
EMBEDDING_MODEL = "text-embedding-ada-002"
COMPLETIONS_API_PARAMS = {
    # We use temperature of 0.0 because it gives the most predictable, factual answer.
    "temperature": 0.0,
    "max_tokens": 500,
    "model": COMPLETIONS_MODEL,
}

openai.api_key = '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'

def search_documentation(df, user_query, n=3):
    user_query_embedding = openai.Embedding.create(input=user_query, model=EMBEDDING_MODEL)['data'][0]['embedding']
    df["similarity"] = df['ada_embedding'].apply(lambda x: np.dot(x, user_query_embedding))
    
    results = df.sort_values("similarity", ascending=False).head(n)
    return results
    
def construct_prompt(question: str, df: pd.DataFrame) -> str:
    """
    Fetch relevant 
    """
    most_relevant_document_sections = search_documentation(ipfolio, question)
    
    chosen_sections = []
    chosen_sections_len = 0
    chosen_sections_indexes = []
    
    for section_index, row in most_relevant_document_sections.iterrows():
        # Add contexts until we run out of space.
        document_section = (row['document'])
        chosen_sections_len += len(encoding.encode(document_section)) + separator_len

        if chosen_sections_len > MAX_SECTION_LEN:
            break
        
        chosen_sections.append(SEPARATOR + document_section.replace("\n", " "))
        chosen_sections_indexes.append(str(section_index))
    
    # Useful diagnostic information
    #print(f"Selected {len(chosen_sections)} document sections:")
    #print("\n".join(chosen_sections_indexes))
    
    header = """Answer the qurey as truthfully as possible using the provided context, and if the answer is not contained within the text below, say "I don't know."\n\nContext:\n"""
    
    return header + "".join(chosen_sections) + "\n\n Q: " + question + "\n A:", chosen_sections_indexes

def answer_query_with_context(query: str, df: pd.DataFrame, show_prompt: bool = False) -> str:
    prompt, chosen_sections_indexes = construct_prompt(query, df)
    
    if show_prompt:
        print(prompt)
    
    response = openai.Completion.create(
                prompt=prompt,
                **COMPLETIONS_API_PARAMS
    )
    
    return response["choices"][0]["text"].strip(" \n"), chosen_sections_indexes

user_guide = pd.read_csv('ipfolio_user_guide_with_embeddings.csv')
user_guide["ada_embedding"] = user_guide['ada_embedding'].apply(eval).apply(np.array)

quip_doco = pd.read_csv('quip_doco_with_embeddings.csv')
quip_doco["ada_embedding"] = quip_doco['ada_embedding'].apply(eval).apply(np.array)

ipfolio = pd.concat([user_guide, quip_doco]).reset_index(drop=True)

encoding = tiktoken.get_encoding(ENCODING)
separator_len = len(encoding.encode(SEPARATOR))

st.title("IPfolio Knowledge Bot")

if 'history' not in st.session_state:
    st.session_state.history = ''

query = st.text_input("What would you like to know?")

if query:
    answer, chosen_sections_indexes = answer_query_with_context(query, ipfolio)
    history_entry = '<br><div style="font-weight: bold;color:#7A7777">Q: ' + query + '</div><br>A: ' + answer + '<br>'
    if answer != 'I don\'t know.':
        for i in chosen_sections_indexes:
            history_entry = history_entry + ipfolio.iloc[int(i),0] + '<br>'
    st.session_state.history = history_entry + st.session_state.history
    print(chosen_sections_indexes)
else:
    st.warning("Please enter a query.")
     
st.markdown(st.session_state.history, unsafe_allow_html=True)

