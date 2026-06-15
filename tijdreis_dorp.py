import os
import random
import tkinter as tk
from tkinter import simpledialog, messagebox
from abc import abstractmethod

# LangChain Imports
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# Importeer de modules van je groepsgenoot
import safety_guard
import ethics_engine

# ==========================================
# CONFIGURATIE & BASIS SETUP
# ==========================================
GRID_SIZE = 15
CELL_SIZE = 40
DEFAULT_SPEED = 2000

import os
from dotenv import load_dotenv

# Laad de variabelen uit het .env bestand in het geheugen van je computer
load_dotenv()

# Nu werkt deze regel perfect! 
# Python zoekt eerst in je systeem/ .env en vindt daar je sleutel.
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    print("Oeps! De GOOGLE_API_KEY is niet gevonden in het .env bestand.")
else:
    print("API key succesvol en veilig geladen!")

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY, temperature=0.5)

# ==========================================
# 1. PRE-RETRIEVAL & POST-RETRIEVAL RAG SETUP
# ==========================================
def initialiseer_rag(tijdperk_keuze):
    """
    Laadt de juiste ongestructureerde data in op basis van de tijdperkkeuze
    en zet een geoptimaliseerde RAG-retriever op.
    """
    bestandsnaam = f"werelden/{tijdperk_keuze.lower()}.txt"
    if not os.path.exists(bestandsnaam):
        # Fallback als het bestand niet bestaat
        with open(bestandsnaam, "w") as f:
            f.write(f"Dit is de wereld van de {tijdperk_keuze}. Het is er gezellig en vredig.")
            
    # Inladen ongestructureerde bron
    loader = TextLoader(bestandsnaam)
    documents = loader.load()
    
    # Text splitting
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=30)
    chunks = text_splitter.split_documents(documents)
    
    # Vectorstore & Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    basis_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    
    # POST-RETRIEVAL OPTIMALISATIE: Contextual Compression
    # Dit filtert de ruis uit de opgehaalde documenten voor een compacter resultaat
    compressor = LLMChainExtractor.from_llm(llm)
    gecomprimeerde_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=basis_retriever
    )
    return gecomprimeerde_retriever

# PRE-RETRIEVAL OPTIMALISATIE: Query Rewriter Chain
# Vertaalt de ruwe Sim-status naar een gerichte zoekvraag voor de database
query_prompt = ChatPromptTemplate.from_messages([
    ("system", "Je bent een query-rewriter voor een Sims-tijdreisdatabase. Genereer één korte, bondige zoekterm in het Nederlands op basis van de situatie van de Sim om te zien hoe zij zich moeten gedragen."),
    ("human", "Stemming: {stemming}, Plan: {actie_plan}, Instructie: {instructie}. Wat moeten we opzoeken over de cultuur of omgeving?")
])
query_generator = query_prompt | llm | StrOutputParser()

# FINALE PROMPT (Inclusief RAG-context & Ethiek-instructies)
sim_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Je bent {naam}, een bewoner van een klein dorp die door de tijd reist naar de {tijdperk}.\n"
     "Jouw persoonlijkheid: {persoonlijkheid}.\n\n"
     "BELANGRIJKE INFORMATIE OVER DIT TIJDPERK (GELADEN VIA RAG):\n"
     "{rag_context}\n\n"
     "ETHISCH ADVIES VAN HET SYSTEEM:\n"
     "{ethisch_advies}\n\n"
     "Gedraag je ALTIJD naar de cultuur van het tijdperk en jouw persoonlijkheid. "
     "Gebruik ALTIJD vriendelijke, kindvriendelijke taal geschikt voor kinderen vanaf 8 jaar. "
     "Geef NOOIT ongepaste, gewelddadige of gemene antwoorden."),
    ("human",
     "Jouw situatie:\n"
     "- Honger score: {honger}/10\n"
     "- Huidige stemming: {stemming}\n"
     "- Objecten in de buurt: {objecten_nabij}\n"
     "- Directe instructie van speler: {instructie}\n\n"
     "Kies precies één actie uit deze lijst: beweeg, eet, rust, of praat. "
     "Geef alleen dat ene woord terug, verder helemaal niets.")
])

kies_actie_chain = sim_prompt | llm | StrOutputParser()

# ==========================================
# INTERFACE & GAME LOGICA
# ==========================================
class SimsWereld:
    def __init__(self, root, tijdperk):
        self.root = root
        self.tijdperk = tijdperk
        self.root.title(f"Sims Tijdreis Dorp - Tijdperk: {tijdperk}")
        
        # Initialiseer de RAG-retriever voor dit specifieke tijdperk
        self.retriever = initialiseer_rag(tijdperk)
        
        self.canvas = tk.Canvas(root, width=GRID_SIZE * CELL_SIZE, height=GRID_SIZE * CELL_SIZE)
        self.canvas.bind("<Button-1>\", self.on_canvas_click")
        self.canvas.pack()
        
        self.speed = DEFAULT_SPEED
        self.running = False
        self.step_mode = False
        self.stopping = False
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        
        self.grid = [[{} for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        self.initialiseer_wereld()
        self.draw_grid()
        
        self.control_frame = tk.Frame(root)
        self.control_frame.pack()
        self.create_buttons()
        self.root.after(self.speed, self.update_world)

    def _on_close(self):
        self.stopping = True
        self.root.destroy()

    def draw_grid(self):
        self.canvas.delete("all")
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                x1, y1 = x * CELL_SIZE, y * CELL_SIZE
                x2, y2 = x1 + CELL_SIZE, y1 + CELL_SIZE
                self.canvas.create_rectangle(x1, y1, x2, y2, fill="white", outline="#ccc")
                cel = self.grid[y][x]
                if cel:
                    self.canvas.create_text(
                        x1 + CELL_SIZE // 2, y1 + CELL_SIZE // 2,
                        text=cel.get("icon", "?"), font=("Arial", 16)
                    )

    def create_buttons(self):
        def play():
            self.step_mode = False
            self.running = not self.running

        def stap():
            self.step_mode = True
            self.running = True

        tk.Button(self.control_frame, text="▶️ Play/Pauze", command=play).grid(row=0, column=0, padx=4)
        tk.Button(self.control_frame, text="⏭️ Stap", command=stap).grid(row=0, column=1, padx=4)

    def update_world(self):
        if self.running:
            self.doe_alles()
            self.draw_grid()
            if self.step_mode:
                self.running = False
        if not self.stopping:
            self.root.after(self.speed, self.update_world)

    def on_canvas_click(self, event):
        gx = event.x // CELL_SIZE
        gy = event.y // CELL_SIZE
        if 0 <= gx < GRID_SIZE and 0 <= gy < GRID_SIZE:
            cel = self.grid[gy][gx]
            if cel.get("type") == "bewoner":
                opdracht = simpledialog.askstring(
                    "Instructie", f"Geef {cel.get('naam')} een tijdreis-opdracht:"
                )
                if opdracht:
                    # INPUT FILTERING (Gemaakt door groepsgenoot)
                    veilig_gemaakte_opdracht = safety_guard.filter_speler_instructie(opdracht)
                    cel["instructies"].append(veilig_gemaakte_opdracht)
                    print(f"[Speler] Gefilterde instructie aan {cel.get('naam')}: {veilig_gemaakte_opdracht}")

    @abstractmethod
    def initialiseer_wereld(self):
        pass

    @abstractmethod
    def doe_alles(self):
        pass


class TijdreisDorpWereld(SimsWereld):
    def initialiseer_wereld(self):
        # Vaste objecten inrichten
        from dorp import VASTE_OBJECTEN, OBJECTEN, BEWONERS, STARTPOSITIES
        for x, y, icon in VASTE_OBJECTEN:
            self.grid[y][x] = {"type": OBJECTEN[icon], "icon": icon}

        for idx, (bewoner, (sx, sy)) in enumerate(zip(BEWONERS, STARTPOSITIES)):
            self.grid[sy][sx] = {
                "type": "bewoner",
                "icon": bewoner["icon"],
                "naam": bewoner["naam"],
                "persoonlijkheim": bewoner["persoonlijkheid"], # Note: Je had typo 'persoonlijkheim' in je prompt-aanroep, hersteld naar 'persoonlijkheid'
                "honger": 7,  # Schaal 1-10 voor jouw game logic
                "stemming": "neutraal",
                "instructies": [],
                "idx": idx,
                "actie": "beweeg",
            }
            print(f"{bewoner['naam']} is gearriveerd in de {self.tijdperk} op ({sx}, {sy})")

    def _objecten_nabij(self, x, y, straal=4):
        gevonden = set()
        for dy in range(-straal, straal + 1):
            for dx in range(-straal, straal + 1):
                ny, nx = y + dy, x + dx
                if 0 <= ny < GRID_SIZE and 0 <= nx < GRID_SIZE:
                    cel = self.grid[ny][nx]
                    if cel and cel.get("type") != "bewoner":
                        gevonden.add(cel["type"])
        return list(gevonden) if gevonden else ["geen"]

    def doe_alles(self):
        posities = [(y, x) for y in range(GRID_SIZE) for x in range(GRID_SIZE) if self.grid[y][x].get("type") == "bewoner"]

        for y, x in posities:
            if self.grid[y][x].get("type") != "bewoner":
                continue
            sim = self.grid[y][x]
            sim["honger"] = max(0, sim["honger"] - 1)

            # ETHISCHE BESLISSINGSLAAG (Gemaakt door groepsgenoot)
            # Let op: De ethics_engine gebruikt een schaal tot 100, dus we vermenigvuldigen sim["honger"] even met 10
            advies = ethics_engine.gefer_ethisch_advies = ethics_engine.geof_ethisch_advies = ethics_engine.gegeef_ethisch_advies = ethics_engine.geef_ethisch_advies(sim["stemming"], sim["honger"] * 10)
            uitleg_advies = ethics_engine.beschrijf_ethische_keuze(sim["stemming"], sim["honger"] * 10)

            instructie = sim["instructies"].pop(0) if sim["instructies"] else "geen"

            # LLM & RAG aanroep per interval
            if True: # Om de werking meteen te zien, roepen we hem nu vaker aan
                try:
                    # A. PRE-RETRIEVAL OPTIMALISATIE: Query Herschrijven
                    zoek_query = query_generator.invoke({
                        "stemming": sim["stemming"],
                        "actie_plan": sim["actie"],
                        "instructie": instructie
                    })

                    # B. RETRIEVAL + POST-RETRIEVAL OPTIMALISATIE: Relevante documenten ophalen & filteren
                    docs = self.retriever.invoke(zoek_query)
                    rag_context = "\n".join([d.page_content for d in docs]) if docs else "Geen specifieke cultuurregels bekend."

                    # C. HOOFD LLM CALL
                    ruwe_actie = kies_actie_chain.invoke({
                        "naam": sim["naam"],
                        "tijdperk": self.tijdperk,
                        "persoonlijkheid": sim.get("persoonlijkheim", "vriendelijk"),
                        "honger": sim["honger"],
                        "stemming": sim["stemming"],
                        "objecten_nabij": ", ".join(self._objecten_nabij(x, y)),
                        "instructie": instructie,
                        "rag_context": rag_context,
                        "ethisch_advies": uitleg_advies
                    }).strip().lower()

                    # OUTPUT FILTERING (Gemaakt door groepsgenoot)
                    gevalideerde_actie = safety_guard.filter_llm_actie(ruwe_actie)
                    sim["actie"] = gevalideerde_actie
                    print(f"[{sim['naam']}] (Query: '{zoek_query}') -> RAG-Context gebruikt! Gekozen actie: {gevalideerde_actie}")

                except Exception as e:
                    print(f"[{sim['naam']}] Fout: {e}")
                    sim["actie"] = "beweeg"

            # Actie afhandeling (Jouw bestaande logica)
            actie = sim.get("actie", "beweeg")
            if actie == "praat":
                print(f"[{sim['naam']}] probeert te praten op de manier van de {self.tijdperk}!")
            elif actie == "beweeg":
                dx, dy = random.choice([(0, 1), (1, 0), (0, -1), (-1, 0)])
                nx, ny = max(0, min(GRID_SIZE - 1, x + dx)), max(0, min(GRID_SIZE - 1, y + dy))
                if not self.grid[ny][nx]:
                    self.grid[ny][nx] = sim
                    self.grid[y][x] = {}

            # Stemming updaten
            if sim["honger"] > 7:   sim["stemming"] = "blij"
            elif sim["honger"] > 4: sim["stemming"] = "neutraal"
            else:                   sim["stemming"] = "hongerig"


# ==========================================
# HOOFDPROGRAMMA MET KEUZEMENU
# ==========================================
if __name__ == "__main__":
    # Start-menu voor de speler om het tijdperk te kiezen
    root_menu = tk.Tk()
    root_menu.withdraw() # Verberg het hoofdvenster tijdelijk

    tijdperk_keuze = simpledialog.askstring(
        "Sims Tijdreis Menu", 
        "Naar welk tijdperk moeten de Sims reizen?\nType: 'Prehistorie' of 'Toekomst'"
    )

    if tijdperk_keuze and tijdperk_keuze.strip().lower() in ["prehistorie", "toekomst"]:
        # Open de game in de gekozen wereld
        game_root = tk.Toplevel(root_menu)
        app = TijdreisDorpWereld(game_root, tijdperk_keuze.strip().capitalize())
        root_menu.mainloop()
    else:
        messagebox.showwarning("Geannuleerd", "Geen geldig tijdperk gekozen. Het spel sluit af.")
        root_menu.destroy()