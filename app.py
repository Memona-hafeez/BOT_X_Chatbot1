import streamlit as st
import argparse
import time
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama

from loader import get_embedding_function

CHROMA_PATH = "nomicDB"

# context - all chunks from db that best match the query
# question - actual question we want to ask

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def query_rag(query_text: str):
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB and give top k most relevant chunks
    results = db.similarity_search_with_score(query_text, k=30)

    # Combine the top results with the original question to generate the prompt
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print("prompt", prompt)

    model = Ollama(model="mistral")
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    return response_text, sources


def main():
    st.title("PolitiBot: Ask Me Anything!")

    # Initialize conversation history
    if "conversation" not in st.session_state:
        st.session_state.conversation = []

    # User input for the question
    user_question = st.text_input("Enter your question here:")

    if st.button("Send"):
        if user_question:
            response, sources = query_rag(user_question)
            st.session_state.conversation.append({"question": user_question, "response": response, "sources": sources})

    # Display conversation history
    for idx, convo in enumerate(st.session_state.conversation):
        st.markdown(f'<p style="font-family:Helvetica; font-weight:bold">Question {idx + 1}: {convo["question"]}</p>', unsafe_allow_html=True)
        st.markdown(f'<p style="font-family:Helvetica;">{convo["response"]}</p>', unsafe_allow_html=True)
        st.write(f"Sources: {convo['sources']}")



         

if __name__ == "__main__":
    main()