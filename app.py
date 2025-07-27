import os
from agno.agent import Agent
from agno.knowledge.pdf_url import PDFUrlKnowledgeBase
from agno.models.openai import OpenAIChat
from agno.vectordb.pgvector import PgVector
from dotenv import load_dotenv

load_dotenv()
db_url = os.getenv("DB_URL", "postgresql+psycopg://ai:ai@localhost:5532/ai")
openai_api_key = os.getenv("OPENAI_API_KEY")

# Load multiple Pakistani legal PDFs
pdf_urls = [
    "https://www.pakistani.org/pakistan/legislation/1860/actXLVof1860.pdf",         # PPC
    "https://www.pakistani.org/pakistan/constitution.pdf",                           # Constitution
    "https://www.pakistani.org/pakistan/legislation/1872/actIof1872.pdf",            # Evidence Act
    "https://www.pakistani.org/pakistan/legislation/1898/actVof1898.pdf",            # CrPC
    "https://www.pakistani.org/pakistan/legislation/1908/actVof1908.pdf",            # Civil Procedure
]

knowledge_base = PDFUrlKnowledgeBase(
    urls=pdf_urls,
    vector_db=PgVector(table_name="pakistan_law_docs", db_url=db_url),
)

# Load documents into vector DB
try:
    knowledge_base.load(recreate=False)
except Exception as e:
    print("Error loading documents:", e)

# Create Legal Advisor Chatbot
legal_agent = Agent(
    name="PakistaniLegalBot",
    knowledge=knowledge_base,
    search_knowledge=True,
    model=OpenAIChat(id="gpt-4o", api_key=openai_api_key),
    markdown=True,
    instructions=[
        "Answer questions strictly based on Pakistani law documents.",
        "Cite the exact article, section, or law when possible.",
        "Clarify that this is general legal information, not professional legal advice.",
        "Always recommend consulting a Pakistani lawyer for official legal matters.",
    ],
)

# Ask a test question
legal_agent.print_response("What are the rights of a citizen under Article 19 of the Constitution?", stream=True)
