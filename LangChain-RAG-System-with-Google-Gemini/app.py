import sys
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


def load_api_key():
    with open("gemini_api_key.txt") as f:
        return f.read().strip()


def load_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=100
    )
    return text_splitter.split_documents(pages)


def create_embeddings_and_vector_store(chunks, key):
    embedding_model = GoogleGenerativeAIEmbeddings(google_api_key=key, model="models/embedding-001")
    db = Chroma.from_documents(chunks, embedding_model, persist_directory="./chroma_db_")
    db.persist()
    return Chroma(persist_directory="./chroma_db_", embedding_function=embedding_model)


def setup_prompt():
    chat_template = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are a Helpful AI Bot. Given a context and question from user, you should answer based on the given context."""),
        HumanMessagePromptTemplate.from_template("""Answer the question based on the given context.
Context: {context}
Question: {question}
Answer: """)
    ])
    return chat_template

def ask_question(question, retriever, chat_template, chat_model, output_parser):
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | chat_template
        | chat_model
        | output_parser
    )
    return rag_chain.invoke(question)

# Fonction pour formater les documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def main():
    key = load_api_key()

    chunks = load_pdf("LeaveNoContextBehind.pdf")

    db_connection = create_embeddings_and_vector_store(chunks, key)

    chat_model = ChatGoogleGenerativeAI(google_api_key=key, model="gemini-1.5-pro-latest")


    chat_template = setup_prompt()
    output_parser = StrOutputParser()

    retriever = db_connection.as_retriever(search_kwargs={"k": 5})

    print("Bienvenue dans le système de question-réponse RAG!")
    print("Posez vos questions sur le document.")

    while True:
        question = input("Entrez votre question (ou 'exit' pour quitter): ")
        if question.lower() == "exit":
            print("Merci d'avoir utilisé le système de question-réponse.")
            break

        response = ask_question(question, retriever, chat_template, chat_model, output_parser)
        print(f"Réponse: {response}")

if __name__ == "__main__":
    main()
