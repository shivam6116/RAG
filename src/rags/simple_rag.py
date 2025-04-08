from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import os

 
loader = TextLoader("knowledge_base.txt")  # Could be PDF, CSV, etc.
documents = loader.load()


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
splits = text_splitter.split_documents(documents)


vectorstore = FAISS.from_documents(
    documents=splits,
    embedding=OpenAIEmbeddings()
)
retriever = vectorstore.as_retriever()

# 4. Define prompt template
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# 5. Initialize LLM (using gpt-3.5-turbo)
model = ChatOpenAI(model="gpt-3.5-turbo")

# 6. Create RAG chain
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

# 7. Query the system
response = rag_chain.invoke("What are the key features of your product?")
print(response)