import os
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# Load environment variables (e.g., OPENAI_API_KEY)
load_dotenv()

def create_rag_agent():
    print("Loading data source...")
    # Data Source: NASA's Perseverance Rover Wikipedia page
    # Using a different and highly detailed source to enable complex and multi-step questions
    url = "https://en.wikipedia.org/wiki/Perseverance_(rover)"
    loader = WebBaseLoader(
        web_paths=(url,),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                "div",
                class_="mw-content-container"
            )
        )
    )
    docs = loader.load()

    print("Splitting documents...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    print(f"Created {len(splits)} document chunks.")

    print("Embedding and creating vector store...")
    # FAISS is used as an easy in-memory vector store
    vectorstore = FAISS.from_documents(splits, OpenAIEmbeddings())
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # Create the retriever tool for the Agent to use
    retriever_tool = create_retriever_tool(
        retriever,
        "perseverance_rover_search",
        "Search for information about the Mars Perseverance rover. You must use this tool to answer any questions about its mission, instruments, landing, or history."
    )

    tools = [retriever_tool]

    print("Initializing LLM and Agent...")
    # Initialize the LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # System prompt for the agent
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an advanced AI assistant tasked with answering questions about the Mars Perseverance rover.
You have access to a retriever tool with information from its Wikipedia page.
For complex, multi-step questions, you should break down the questions and use the tool multiple times if necessary to find all the pieces of information before synthesizing your final answer.
Always cite the facts you found."""),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    # Create the agent that uses OpenAI's tool calling feature
    agent = create_tool_calling_agent(llm, tools, prompt)
    
    # AgentExecutor runs the agent loop (Thought -> Action -> Observation -> ... -> Final Answer)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    return agent_executor

def main():
    # Check if API key is set
    if not os.environ.get("OPENAI_API_KEY"):
        print("WARNING: OPENAI_API_KEY environment variable not set. Please set it before running.")
        print("Example: set OPENAI_API_KEY=sk-...")
        return

    agent_executor = create_rag_agent()

    print("\n" + "="*50)
    print("RAG Agent Ready!")
    print("="*50 + "\n")

    # Multi-step question examples that require looking up multiple facts.
    questions = [
        "What is the name of the helicopter attached to Perseverance, and on what date did the rover land on Mars?",
        "What are the names of the seven primary scientific instruments on the rover? Briefly describe what MEDA and MOXIE do."
    ]

    for q in questions:
        print(f"\nUser Question: {q}")
        response = agent_executor.invoke({"input": q})
        print(f"\nFinal Answer: {response['output']}\n")
        print("-" * 50)

if __name__ == "__main__":
    main()
