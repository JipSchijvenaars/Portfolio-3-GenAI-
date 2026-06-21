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
    Simpele RAG-versie zonder FAISS, zodat het stabiel werkt in het notebook.
    """
    bestandsnaam = f"werelden/{tijdperk.lower()}.txt"

    if not os.path.exists(bestandsnaam):
        return f"Fout: Het tijdperk '{tijdperk}' is onbekend of het bestand '{bestandsnaam}' ontbreekt."

    try:
        with open(bestandsnaam, "r", encoding="utf-8") as f:
            tekst = f.read()

        # Simpele pre-retrieval: zoekvraag opschonen naar zoekwoorden
        stopwoorden = {"de", "het", "een", "en", "of", "bij", "in", "op", "te", "is", "heeft", "zoekt"}
        zoekwoorden = [
            woord.lower().strip(".,!?")
            for woord in zoekvraag.split()
            if woord.lower().strip(".,!?") not in stopwoorden
        ]

        # Simpele chunking per zin
        zinnen = [zin.strip() for zin in tekst.replace("\n", " ").split(".") if zin.strip()]

        # Retrieval: score zinnen op overlap met zoekwoorden
        gescoorde_zinnen = []
        for zin in zinnen:
            score = sum(1 for woord in zoekwoorden if woord in zin.lower())
            if score > 0:
                gescoorde_zinnen.append((score, zin))

        gescoorde_zinnen.sort(reverse=True, key=lambda x: x[0])

        # Post-retrieval: neem maximaal 2 relevantste zinnen
        relevante_zinnen = [zin for score, zin in gescoorde_zinnen[:2]]

        if not relevante_zinnen:
            relevante_zinnen = zinnen[:2]

        context = ". ".join(relevante_zinnen)

        return (
            f"Relevante informatie over de {tijdperk} voor de vraag '{zoekvraag}':\n\n"
            f"{context}."
        )

    except Exception as e:
        return f"Er ging iets mis bij het doorzoeken van de lokale RAG: {str(e)}"


@tool
def vraag_karakter_biografie(naam: str, zoekvraag: str) -> str:
    """
    Doorzoekt de persoonlijke achtergrond, biografie en specifieke karaktertrekken van de bewoners.
    Simpele RAG-versie zonder FAISS.
    """
    bestandsnaam = f"karakters/{naam.lower()}.txt"

    if not os.path.exists(bestandsnaam):
        return f"Fout: Er is geen karakter-bestand gevonden voor '{naam}'."

    try:
        with open(bestandsnaam, "r", encoding="utf-8") as f:
            tekst = f.read()

        stopwoorden = {"de", "het", "een", "en", "of", "bij", "in", "op", "te", "is", "heeft", "zoekt"}
        zoekwoorden = [
            woord.lower().strip(".,!?")
            for woord in zoekvraag.split()
            if woord.lower().strip(".,!?") not in stopwoorden
        ]

        zinnen = [zin.strip() for zin in tekst.replace("\n", " ").split(".") if zin.strip()]

        gescoorde_zinnen = []
        for zin in zinnen:
            score = sum(1 for woord in zoekwoorden if woord in zin.lower())
            if score > 0:
                gescoorde_zinnen.append((score, zin))

        gescoorde_zinnen.sort(reverse=True, key=lambda x: x[0])

        relevante_zinnen = [zin for score, zin in gescoorde_zinnen[:2]]

        if not relevante_zinnen:
            relevante_zinnen = zinnen[:2]

        context = ". ".join(relevante_zinnen)

        return f"Persoonlijke achtergrond van {naam}:\n\n{context}."

    except Exception as e:
        return f"Er ging iets mis bij het doorzoeken van de karakter-RAG: {str(e)}"