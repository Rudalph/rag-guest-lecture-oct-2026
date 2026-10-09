# Load the document and extract the content
from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader("MediSense_Research_Paper.pdf")
data = loader.load()


# Split the content into chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=250)
docs = text_splitter.split_documents(data)


# Load the Embedding Model
from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# Create the instance of Chroma DB (Vector Database)
# Convert the docs in to embeddings
# Store the embeddings into vector database
from langchain_chroma import Chroma
vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)


# Create a retriever
retriever = vectorstore.as_retriever(search_type="similarity")


# Create the instance of LLMs 
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
load_dotenv()
llm_api_key = os.getenv("GROQ_API_KEY")
llm=ChatGroq(groq_api_key=llm_api_key, model_name="openai/gpt-oss-120b", temperature=0.5)


# Take query from user
user_query = input("Enter your question: ")


# Create a system prompt
# Instruction to LLM
# Combine System prompt and User query
from langchain_core.prompts import ChatPromptTemplate
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise.\n\n{context}"
)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])


# Chain - One chain: responsible of doing one task
# Chains - Multiple chains work together - O/P of one chain I/P of another chain
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
if user_query:
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    response = rag_chain.invoke({"input": user_query})
    print(response["answer"])