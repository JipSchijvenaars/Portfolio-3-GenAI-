# Lokale RAG & Karakter Toolkit (Ollama)

Vanaf deze branch draait het RAG-systeem volledig lokaal, gratis en onbeperkt via Ollama. Dit lost de `429 Resource Exhausted` quota-fouten van Google definitief op. Daarnaast is de RAG-architectuur nu opgesplitst in een Wereld-kennisbank én een persoonlijke Karakter-biografie.

---

## Installatie & Setup (Eenmalig)

Om dit script te kunnen draaien, moet je computer de lokale AI-modellen kunnen hosten:

1. Download Ollama: Ga naar https://ollama.com/ en installeer de bijbehorende versie. Zorg dat het lama-icoontje rechtsonderin je taakbalk actief is.
2. Download de Modellen: Open een normale terminal (PowerShell) en run de volgende twee commando's:
```bash
   ollama pull nomic-embed-text
   ollama pull llama3.2
```
3. Python package installeren: Zorg dat je Anaconda-omgeving actief is in de VS Code terminal en run:
```bash
   pip install langchain-ollama
```

## Hoe te gebruiken & Modulair hergebruik

De RAG-functionaliteit is volledig losgetrokken van de GUI en staat in tijdreis_rag.py. Hierin zitten twee officiële LangChain Tools die je direct kunt importeren in andere notebooks of kunt meegeven aan een AI Agent:
```python
from tijdreis_rag import vraag_tijdreis_kennisbank, vraag_karakter_biografie

# Voorbeeld 1: Handmatige RAG-check
cultuur = vraag_tijdreis_kennisbank.invoke({"tijdperk": "prehistorie", "zoekvraag": "eten"})
paspoort = vraag_karakter_biografie.invoke({"naam": "Lars", "zoekvraag": "wifi"})

# Voorbeeld 2: Direct koppelen aan een Agent
mijn_tools = [vraag_tijdreis_kennisbank, vraag_karakter_biografie]
```

## De karakters map

In de map karakters/ vind je voor alle 5 de bewoners een eigen .txt bestand. De Sims baseren hun keuzes en emoties live op de teksten in deze bestanden via de RAG-loop.
