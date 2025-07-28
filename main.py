import os
import sys
from typing import List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

try:
    from langchain.document_loaders import PyPDFDirectoryLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_pinecone import PineconeVectorStore
    from pinecone import Pinecone
    from langchain.embeddings import FakeEmbeddings  # Replaces sentence-transformers
    from langchain.chains.question_answering import load_qa_chain
    from langchain_google_genai import GoogleGenerativeAI
except ImportError as e:
    print(f"❌ Missing library: {e}")
    sys.exit(1)


class RAGSystem:
    def __init__(self, documents_dir: str = "documents/", index_name: str = "langchainvector"):
        self.documents_dir = documents_dir
        self.index_name = index_name
        self.vectorstore = None
        self.chain = None
        self.embeddings = None

        self._setup_embeddings()
        self._setup_pinecone()
        self._setup_llm()

    def _setup_embeddings(self):
        try:
            print("🔧 Setting up lightweight (fake) embeddings for Railway...")
            self.embeddings = FakeEmbeddings(size=1536)
            print("✅ Lightweight embeddings initialized")
        except Exception as e:
            print(f"❌ Error setting up embeddings: {e}")
            raise

    def _setup_pinecone(self):
        try:
            api_key = os.getenv("PINECONE_API_KEY")
            if not api_key:
                raise ValueError("PINECONE_API_KEY not found")

            print("🔧 Connecting to Pinecone...")
            pc = Pinecone(api_key=api_key)

            try:
                _ = pc.describe_index(self.index_name)
                print(f"✅ Index '{self.index_name}' exists")
            except Exception as e:
                print(f"❌ Index '{self.index_name}' not found: {e}")
                raise

        except Exception as e:
            print(f"❌ Error setting up Pinecone: {e}")
            raise

    def _setup_llm(self):
        try:
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY not set")

            print("🔧 Setting up LLM (Gemini)...")
            llm = GoogleGenerativeAI(
                model="gemini-1.5-flash",
                api_key=api_key,
                temperature=0.5
            )
            self.chain = load_qa_chain(llm, chain_type="stuff")
            print("✅ LLM initialized")
        except Exception as e:
            print(f"❌ Error setting up LLM: {e}")
            raise

    def read_documents(self) -> List:
        try:
            if not os.path.exists(self.documents_dir):
                raise FileNotFoundError(f"Directory '{self.documents_dir}' not found")

            loader = PyPDFDirectoryLoader(self.documents_dir)
            documents = loader.load()
            print(f"✅ Loaded {len(documents)} document(s)")
            return documents
        except Exception as e:
            print(f"❌ Error loading documents: {e}")
            raise

    def chunk_documents(self, documents: List, chunk_size: int = 500, chunk_overlap: int = 30) -> List:
        try:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            chunks = splitter.split_documents(documents)
            print(f"✅ Created {len(chunks)} chunks")
            return chunks
        except Exception as e:
            print(f"❌ Error chunking documents: {e}")
            raise

    def create_vectorstore(self, documents: List) -> None:
        try:
            self.vectorstore = PineconeVectorStore.from_documents(
                documents=documents,
                embedding=self.embeddings,
                index_name=self.index_name
            )
            print("✅ Vector store created")
        except Exception as e:
            print(f"❌ Error creating vector store: {e}")
            raise

    def retrieve_query(self, query: str, k: int = 2) -> List:
        if not self.vectorstore:
            raise ValueError("Vector store not initialized")

        try:
            return self.vectorstore.similarity_search(query, k=k)
        except Exception as e:
            print(f"❌ Error retrieving query: {e}")
            raise

    def get_answer(self, query: str, k: int = 2) -> str:
        try:
            print(f"🔍 Searching for: {query}")
            docs = self.retrieve_query(query, k)
            if not docs:
                return "No relevant documents found."
            return self.chain.run(input_documents=docs, question=query)
        except Exception as e:
            return f"❌ Error answering: {str(e)}"

    def setup_documents(self) -> None:
        try:
            docs = self.read_documents()
            chunks = self.chunk_documents(docs)
            self.create_vectorstore(chunks)
            print("🎉 Document setup complete")
        except Exception as e:
            print(f"❌ Document setup failed: {e}")
            raise


def main():
    try:
        print("🚀 Starting RAG System...")
        rag = RAGSystem()
        rag.setup_documents()

        print("\n💬 Ask questions (type 'exit' to quit)")
        while True:
            query = input("❓ Your question: ").strip()
            if query.lower() in ['exit', 'quit']:
                break
            if not query:
                continue
            print(f"🤖 {rag.get_answer(query)}")
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
