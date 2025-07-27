import os
import sys
from typing import List, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import required libraries
try:
    from langchain.document_loaders import PyPDFDirectoryLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_pinecone import PineconeVectorStore
    from pinecone import Pinecone
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.chains.question_answering import load_qa_chain
    from langchain_google_genai import GoogleGenerativeAI
except ImportError as e:
    print(f"❌ Missing required library: {e}")
    print("Please install required packages: pip install langchain langchain-pinecone langchain-google-genai pinecone-client python-dotenv sentence-transformers")
    sys.exit(1)


class RAGSystem:
    def __init__(self, documents_dir: str = "documents/", index_name: str = "langchainvector"):
        """
        Initialize the RAG (Retrieval-Augmented Generation) system.
        
        Args:
            documents_dir: Directory containing PDF documents
            index_name: Name of the Pinecone index
        """
        self.documents_dir = documents_dir
        self.index_name = index_name
        self.vectorstore = None
        self.chain = None
        self.embeddings = None
        
        # Initialize components
        self._setup_embeddings()
        self._setup_pinecone()
        self._setup_llm()
    
    def _setup_embeddings(self):
        """Initialize HuggingFace embeddings."""
        try:
            print("🔧 Setting up embeddings...")
            self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            print("✅ Embeddings initialized successfully!")
        except Exception as e:
            print(f"❌ Error setting up embeddings: {e}")
            raise
    
    def _setup_pinecone(self):
        """Initialize Pinecone connection and check index."""
        try:
            api_key = os.getenv("PINECONE_API_KEY")
            if not api_key:
                raise ValueError("PINECONE_API_KEY not found in environment variables")
            
            print("🔧 Connecting to Pinecone...")
            pc = Pinecone(api_key=api_key)
            
            # Check if index exists
            try:
                index_info = pc.describe_index(self.index_name)
                print(f"✅ Index '{self.index_name}' exists and is ready")
            except Exception as e:
                print(f"❌ Index '{self.index_name}' not found: {e}")
                print("Please create the index in Pinecone dashboard or use a different index name")
                raise
                
        except Exception as e:
            print(f"❌ Error setting up Pinecone: {e}")
            raise
    
    def _setup_llm(self):
        """Initialize Google Generative AI LLM."""
        try:
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                # Fallback to hardcoded key (not recommended for production)
                api_key = "AIzaSyC00wXTsPIM44G_wcnN8BVt45QEFCDWMcs"
                print("⚠️  Using fallback API key. Consider setting GOOGLE_API_KEY environment variable.")
            
            print("🔧 Setting up LLM...")
            llm = GoogleGenerativeAI(
                model="gemini-1.5-flash",
                api_key=api_key,
                temperature=0.5
            )
            self.chain = load_qa_chain(llm, chain_type="stuff")
            print("✅ LLM initialized successfully!")
            
        except Exception as e:
            print(f"❌ Error setting up LLM: {e}")
            raise
    
    def read_documents(self) -> List:
        """Load documents from the specified directory."""
        try:
            if not os.path.exists(self.documents_dir):
                raise FileNotFoundError(f"Documents directory '{self.documents_dir}' not found")
            
            print(f"📚 Loading documents from '{self.documents_dir}'...")
            file_loader = PyPDFDirectoryLoader(self.documents_dir)
            documents = file_loader.load()
            print(f"✅ Loaded {len(documents)} documents")
            return documents
            
        except Exception as e:
            print(f"❌ Error loading documents: {e}")
            raise
    
    def chunk_documents(self, documents: List, chunk_size: int = 500, chunk_overlap: int = 30) -> List:
        """Split documents into chunks for processing."""
        try:
            print(f"✂️  Chunking documents (size: {chunk_size}, overlap: {chunk_overlap})...")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, 
                chunk_overlap=chunk_overlap
            )
            chunks = text_splitter.split_documents(documents)
            print(f"✅ Created {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            print(f"❌ Error chunking documents: {e}")
            raise
    
    def create_vectorstore(self, documents: List) -> None:
        """Create and populate the vector store."""
        try:
            print("🔧 Creating vector store...")
            self.vectorstore = PineconeVectorStore.from_documents(
                documents=documents,
                embedding=self.embeddings,
                index_name=self.index_name
            )
            print("✅ Vector store created successfully!")
            
        except Exception as e:
            print(f"❌ Error creating vector store: {e}")
            raise
    
    def retrieve_query(self, query: str, k: int = 2) -> List:
        """Retrieve relevant documents for a query."""
        if not self.vectorstore:
            raise ValueError("Vector store not initialized. Call create_vectorstore() first.")
        
        try:
            matching_results = self.vectorstore.similarity_search(query, k=k)
            return matching_results
        except Exception as e:
            print(f"❌ Error retrieving query: {e}")
            raise
    
    def get_answer(self, query: str, k: int = 2) -> str:
        """Get answer for a query using RAG."""
        try:
            print(f"🔍 Searching for: '{query}'")
            doc_search = self.retrieve_query(query, k=k)
            
            if not doc_search:
                return "No relevant documents found for your query."
            
            print(f"📄 Found {len(doc_search)} relevant documents")
            response = self.chain.run(input_documents=doc_search, question=query)
            return response
            
        except Exception as e:
            print(f"❌ Error getting answer: {e}")
            return f"Error processing your query: {str(e)}"
    
    def setup_documents(self) -> None:
        """Complete setup process: load, chunk, and create vector store."""
        try:
            # Load documents
            documents = self.read_documents()
            
            # Chunk documents
            chunks = self.chunk_documents(documents)
            
            # Create vector store
            self.create_vectorstore(chunks)
            
            print("🎉 Document setup completed successfully!")
            
        except Exception as e:
            print(f"❌ Error in document setup: {e}")
            raise


def main():
    """Main function to run the RAG system."""
    try:
        print("🚀 Starting RAG System...")
        
        # Initialize RAG system
        rag = RAGSystem()
        
        # Setup documents (only needed once)
        rag.setup_documents()
        
        # Interactive query loop
        print("\n" + "="*50)
        print("💬 RAG System Ready! Ask questions (type 'quit' to exit)")
        print("="*50)
        
        while True:
            query = input("\n❓ Your question: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not query:
                print("⚠️  Please enter a question.")
                continue
            
            # Get answer
            answer = rag.get_answer(query)
            print(f"\n🤖 Answer: {answer}")
            
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()