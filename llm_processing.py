import os
import chromadb

from sentence_transformers import SentenceTransformer
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.prompts import PromptTemplate
import streamlit as st


class RAGProcessor:
    def __init__(self, chroma_path="./chroma", collection_name="docs", model_name="llama3.2"):
        """
        Initialize the RAG processor with ChromaDB and LLM
        
        Args:
            chroma_path: Path to ChromaDB storage
            collection_name: Name of the document collection
            model_name: Ollama model name (e.g., 'llama3.2', 'mistral', 'codellama')
        """
        self.chroma_path = chroma_path
        self.collection_name = collection_name
        self.model_name = model_name
        
        # Initialize embedding model (same as used for document processing)
        self.embedding_model = SentenceTransformerEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        
        # Initialize LLM
        self.llm = OllamaLLM(model=model_name, temperature=0.1)
        
        # Setup vector store
        self.vectorstore = None
        self.retriever = None
        self.invoke = None
        
        self._setup_vectorstore()
        self._setup_invoke()
    
    def _setup_vectorstore(self):
        """Setup ChromaDB vector store"""
        try:
            # Connect to existing ChromaDB
            client = chromadb.PersistentClient(path=self.chroma_path)
            collection = client.get_collection(name=self.collection_name)
            
            # Create Langchain-compatible vector store
            self.vectorstore = Chroma(
                client=client,
                collection_name=self.collection_name,
                embedding_function=self.embedding_model
            )
            
            # Setup retriever
            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}  # Return top 5 most similar chunks
            )
            
            print(f"✅ Vector store loaded successfully with collection: {self.collection_name}")
            
        except Exception as e:
            print(f"❌ Error setting up vector store: {str(e)}")
            print("Make sure you've run load_docs.py first to create the vector database.")
            
    def _setup_invoke(self):
        """Setup the QA chain with custom prompt"""
        if not self.retriever:
            print("❌ Cannot setup QA chain: retriever not initialized")
            return
            
        # Custom prompt template
        prompt_template = """
        Tu es un assistant intelligent qui répond aux questions en utilisant uniquement les informations fournies dans le contexte.
        
        Contexte des documents:
        {context}
        
        Question: {question}
        
        Instructions:
        - Réponds uniquement en utilisant les informations du contexte fourni
        - Si l'information n'est pas dans le contexte, dis "Je ne trouve pas cette information dans les documents fournis"
        - Sois précis et concis
        - Cite les sources quand c'est possible
        - Réponds en français
        
        Réponse:
        """
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create QA chain
        self.invoke = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )
        
        print("✅ QA chain setup complete")
    
    def ask_question(self, question: str):
        """
        Ask a question and get an answer from the RAG system
        
        Args:
            question: The question to ask
            
        Returns:
            dict: Contains 'answer' and 'source_documents'
        """
        if not self.invoke:
            return {
                "answer": "❌ Le système RAG n'est pas initialisé correctement. Vérifiez la base de données vectorielle.",
                "source_documents": []
            }
        
        try:
            # Get answer from QA chain
            result = self.invoke({"query": question})
            
            return {
                "answer": result["result"],
                "source_documents": result["source_documents"]
            }
            
        except Exception as e:
            return {
                "answer": f"❌ Erreur lors du traitement de la question: {str(e)}",
                "source_documents": []
            }
    
    def search_similar_documents(self, query: str, k: int = 5):
        """
        Search for similar documents without LLM processing
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            list: List of similar document chunks
        """
        if not self.vectorstore:
            return []
        
        try:
            results = self.vectorstore.similarity_search(query, k=k)
            return results
        except Exception as e:
            print(f"❌ Error in similarity search: {str(e)}")
            return []


def create_streamlit_app():
    """Create a Streamlit web interface for the RAG system"""
    st.title("🤖 RAG System - Assistant Intelligent")
    st.write("Posez vos questions sur les documents chargés dans la base de connaissances")
    
    # Initialize RAG processor
    if 'rag_processor' not in st.session_state:
        with st.spinner("Initialisation du système RAG..."):
            st.session_state.rag_processor = RAGProcessor()
    
    # Chat interface
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show sources if available
            if message["role"] == "assistant" and "sources" in message:
                if message["sources"]:
                    with st.expander("📄 Sources"):
                        for i, source in enumerate(message["sources"]):
                            st.write(f"**Source {i+1}:**")
                            st.write(source.page_content[:200] + "...")
                            if hasattr(source, 'metadata'):
                                st.write(f"*Metadata: {source.metadata}*")
    
    # Input for new question
    if question := st.chat_input("Posez votre question ici..."):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": question})
        
        with st.chat_message("user"):
            st.markdown(question)
        
        # Get answer from RAG system
        with st.chat_message("assistant"):
            with st.spinner("Recherche dans la base de connaissances..."):
                result = st.session_state.rag_processor.ask_question(question)
                
            st.markdown(result["answer"])
            
            # Show sources
            if result["source_documents"]:
                with st.expander("📄 Sources"):
                    for i, source in enumerate(result["source_documents"]):
                        st.write(f"**Source {i+1}:**")
                        st.write(source.page_content[:200] + "...")
                        if hasattr(source, 'metadata'):
                            st.write(f"*Metadata: {source.metadata}*")
        
        # Add assistant message to chat
        st.session_state.messages.append({
            "role": "assistant", 
            "content": result["answer"],
            "sources": result["source_documents"]
        })
    
    # Sidebar with system info
    with st.sidebar:
        st.title("⚙️ Système RAG")
        st.write("**Modèle:** Ollama")
        st.write("**Embeddings:** all-MiniLM-L6-v2")
        st.write("**Base vectorielle:** ChromaDB")
        
        if st.button("🔄 Réinitialiser la conversation"):
            st.session_state.messages = []
            st.rerun()


# Command line interface
def main():
    """Main function for command line usage"""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "streamlit":
        # Run Streamlit app
        create_streamlit_app()
    else:
        # Interactive command line mode
        print("🤖 RAG System - Mode Interactif")
        print("Tapez 'quit' pour quitter\n")
        
        rag = RAGProcessor()
        
        while True:
            question = input("\n❓ Votre question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Au revoir!")
                break
                
            if not question:
                continue
            
            print("\n🔍 Recherche en cours...")
            result = rag.ask_question(question)
            
            print(f"\n🤖 Réponse:\n{result['answer']}")
            
            if result['source_documents']:
                print(f"\n📄 Sources trouvées ({len(result['source_documents'])}):")
                for i, doc in enumerate(result['source_documents'], 1):
                    print(f"\nSource {i}:")
                    print(doc.page_content[:150] + "...")


if __name__ == "__main__":
    main()