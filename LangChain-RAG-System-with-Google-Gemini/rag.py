import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import NLTKTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Read the API key
with open("gemini_api_key.txt") as f:
    key = f.read().strip()

import nltk
nltk.download("punkt")

# Setup chat model
chat_model = ChatGoogleGenerativeAI(google_api_key=key, model="gemini-1.5-pro-latest")

# Load the doc
loader = PyPDFLoader("LeaveNoContextBehind.pdf")
pages = loader.load_and_split()

# Split the document into chunks
text_splitter = NLTKTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = text_splitter.split_documents(pages)

# Creating Chunks Embedding
embedding_model = GoogleGenerativeAIEmbeddings(google_api_key=key, model="models/embedding-001")

# Embed each chunk and load it into the vector store
db = Chroma.from_documents(chunks, embedding_model, persist_directory="./chroma_db_")

# Persist the database on drive
db.persist()

# Setting a Connection with the ChromaDB
db_connection = Chroma(persist_directory="./chroma_db_", embedding_function=embedding_model)

# Converting CHROMA db_connection to Retriever Object
retriever = db_connection.as_retriever(search_kwargs={"k": 5})

# Setup chat template
chat_template = ChatPromptTemplate.from_messages([
    # System Message Prompt Template
    SystemMessage(content="""You are a Helpful AI Bot.
                  Given a context and question from user,
                  you should answer based on the given context."""),

    # Human Message Prompt Template
    HumanMessagePromptTemplate.from_template("""Answer the question based on the given context.
    Context: {context}
    Question: {question}
    Answer: """)
])

# Set up output parser 
output_parser = StrOutputParser()

# rag chain function
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | chat_template
    | chat_model
    | output_parser
)

# Streamlit UI
with st.sidebar:
    st.title(":blue[A Retrieval Augmented System on the 'Leave No Context Behind' Paper]")
st.title(":blue[💬Document Chatbot]")
query = st.text_area("Enter your query:", placeholder="Enter your query here...", height=100)

if st.button("Submit Your Query"):
    if query:
        # Ensuring query is passed in correct format
        context = retriever.retrieve(query)
        formatted_context = format_docs(context)
        
        # Invoke rag_chain with the correct inputs
        response = rag_chain.invoke({
            "context": formatted_context,
            "question": query
        })
        
        st.write(response)
    else:
        st.warning("Please enter a question.")
