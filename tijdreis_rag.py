# tijdreis_rag.py
import os
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool

# We initialiseren het embedding-model centraal voor beide tools
embeddings_model = OllamaEmbeddings(model="nomic-embed-text")

@tool
def vraag_tijdreis_kennisbank(tijdperk: str, zoekvraag: str) -> str:
    """
    Doorzoekt de historische en toekomstige database van het tijdreis-dorp.
    Gebruik deze tool als je specifieke informatie nodig hebt over de cultuur, 
    regels, kleding, omgangsvormen of objecten van de 'prehistorie' of 'toekomst'.
    
    Args:
        tijdperk (str): Moet exact 'prehistorie' of 'toekomst' zijn.
        zoekvraag (str): De specifieke vraag of situatie waar je context bij zoekt.
    Returns:
        str: De relevante context uit de documenten.
    """
    bestandsnaam = f"werelden/{tijdperk.lower()}.txt"
    if not os.path.exists(bestandsnaam):
        return f"Fout: Het tijdperk '{tijdperk}' is onbekend of het bestand '{bestandsnaam}' ontbreekt."
        
    try:
        loader = TextLoader(bestandsnaam)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=30)
        chunks = text_splitter.split_documents(documents)
        
        vectorstore = FAISS.from_documents(chunks, embeddings_model)
        relevante_docs = vectorstore.similarity_search(zoekvraag, k=2)
        
        if not relevante_docs:
            return f"Er is in de database van de {tijdperk} niets gevonden over '{zoekvraag}'."
            
        context = "\n---\n".join([doc.page_content for doc in relevante_docs])
        return f"Relevante informatie over de {tijdperk} voor de vraag '{zoekvraag}':\n\n{context}"
        
    except Exception as e:
        return f"Er ging iets mis bij het doorzoeken van de lokale RAG: {str(e)}"


@tool
def vraag_karakter_biografie(naam: str, zoekvraag: str) -> str:
    """
    NIEUW: Doorzoekt de persoonlijke achtergrond, biografie en specifieke karaktertrekken van de bewoners.
    Gebruik deze tool ALTIJD als je wilt weten hoe een specifieke Sim (zoals Emma, Lars, Fatima, Daan of Sofia)
    in elkaar steekt, wat hun achtergrond is, of hoe zij reageren op situaties.
    
    Args:
        naam (str): De naam van de bewoner (bijv. 'Lars', 'Emma', 'Fatima', 'Daan', 'Sofia').
        zoekvraag (str): De specifieke vraag over het gedrag of de eigenschappen van dit karakter.
    Returns:
        str: De relevante biografie-details.
    """
    bestandsnaam = f"karakters/{naam.lower()}.txt"
    if not os.path.exists(bestandsnaam):
        return f"Fout: Er is geen karakter-bestand gevonden voor '{naam}'."
        
    try:
        loader = TextLoader(bestandsnaam)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=20)
        chunks = text_splitter.split_documents(documents)
        
        vectorstore = FAISS.from_documents(chunks, embeddings_model)
        relevante_docs = vectorstore.similarity_search(zoekvraag, k=1)
        
        if not relevante_docs:
            return f"Geen specifieke karakterdetails gevonden voor {naam} over '{zoekvraag}'."
            
        return f"Persoonlijke achtergrond van {naam}:\n\n{relevante_docs[0].page_content}"
        
    except Exception as e:
        return f"Er ging iets mis bij het doorzoeken van de karakter-RAG: {str(e)}"