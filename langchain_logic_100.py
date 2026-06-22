"""
LangChain-beslislaag voor de Sims/tijdreis-game.

Exporteert:
- kies_actie_voor_sim_dict()    → hoofdbeslissing per Sim
- kies_reactie_voor_nora()      → LLM-reactie van Nora op een Sim
- registreer_resultaat()        → sla actieresultaat op in memory
- sla_gesprek_op()              → sla dialoog op in koppelgeheugen
- geef_gesprek_context()        → haal eerdere gesprekken op voor prompt
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, TypedDict

from pydantic import BaseModel, Field

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

try:
    from tijdreis_rag import vraag_tijdreis_kennisbank, vraag_karakter_biografie
except Exception:
    vraag_tijdreis_kennisbank = None
    vraag_karakter_biografie = None

try:
    from safety_guard import filter_llm_actie, filter_speler_instructie
except Exception:
    filter_llm_actie = None
    filter_speler_instructie = None

try:
    from ethics_engine import geef_ethisch_advies
except Exception:
    geef_ethisch_advies = None


# ---------------------------------------------------------------------------
# Types & Pydantic-modellen
# ---------------------------------------------------------------------------

Actie = Literal["beweeg", "eet", "rust", "praat"]
TOEGESTANE_ACTIES: List[str] = ["beweeg", "eet", "rust", "praat"]
Emotie = Literal[
    "blij", "rustig", "nieuwsgierig", "verdrietig",
    "bezorgd", "moe", "hongerig", "neutraal", "eenzaam"
]


class SimInput(TypedDict):
    naam: str
    persoonlijkheid: str
    honger: int
    honger_label: str       # ← menselijk leesbaar label voor de LLM
    stemming: str
    objecten_nabij: str
    instructie: str
    tijdperk: str
    herinneringen: str
    laatste_nieuws: str
    buur_naam: str
    buur_dialoog: str
    gesprek_context: str


class SimBeslissing(BaseModel):
    """Gestructureerde JSON-output van de LLM voor de gekozen actie."""
    actie: Actie = Field(description="Exact een van: beweeg, eet, rust, praat")
    reden: str = Field(description="Korte kindvriendelijke reden voor deze actie")
    emotie: Emotie = Field(description="Emotie van de Sim")
    dialoog: str = Field(description="Wat de Sim zegt, max 20 woorden")
    gebruikt_rag: bool = Field(description="Of de beslissing RAG-context gebruikte")


class NoraReactie(BaseModel):
    """Output van Nora na een gesprek met een Sim."""
    reactie: str = Field(description="Nora's reactie, max 25 woorden, kindvriendelijk")
    emotie: Emotie = Field(description="Nora's emotie na dit gesprek")


# ---------------------------------------------------------------------------
# Honger → menselijk label
# ---------------------------------------------------------------------------

def honger_label(honger: int) -> str:
    """
    Zet het getal (0=vol, 100=uitgehongerd) om naar een duidelijk
    Nederlands label zodat de LLM de schaal niet verkeerd interpreteert.
    """
    if honger <= 10:
        return "helemaal vol – heeft geen honger"
    elif honger <= 30:
        return "licht verzadigd – nauwelijks honger"
    elif honger <= 50:
        return "een beetje honger, maar niet urgent"
    elif honger <= 70:
        return "behoorlijk hongerig"
    elif honger <= 85:
        return "erg hongerig – eten is verstandig"
    else:
        return "uitgehongerd – moet nu eten"


# ---------------------------------------------------------------------------
# Memory – individueel per Sim
# ---------------------------------------------------------------------------

sim_memories: Dict[str, InMemoryChatMessageHistory] = {}


def geef_memory_context(naam: str) -> str:
    """Laatste 6 berichten uit de individuele buffer als leesbare string."""
    if naam not in sim_memories:
        sim_memories[naam] = InMemoryChatMessageHistory()
    recente = sim_memories[naam].messages[-6:]
    if not recente:
        return "Nog geen eerdere herinneringen."
    regels = []
    for msg in recente:
        if isinstance(msg, HumanMessage):
            regels.append(f"Situatie: {msg.content}")
        elif isinstance(msg, AIMessage):
            regels.append(f"Actie: {msg.content}")
    return "\n".join(regels)


def voeg_memory_toe(naam: str, input_text: str, output_text: str) -> None:
    """Slaat een uitgevoerde actie op in de individuele buffer van de Sim."""
    if naam not in sim_memories:
        sim_memories[naam] = InMemoryChatMessageHistory()
    sim_memories[naam].add_user_message(input_text)
    sim_memories[naam].add_ai_message(output_text)


# ---------------------------------------------------------------------------
# Memory – per koppel (gesprekken tussen twee karakters)
# ---------------------------------------------------------------------------

gesprekken_geheugen: Dict[str, List[Dict]] = {}


def _koppel_sleutel(naam1: str, naam2: str) -> str:
    return "-".join(sorted([naam1, naam2]))


def geef_gesprek_context(naam1: str, naam2: str, max_regels: int = 6) -> str:
    """
    Haalt de laatste gesprekswisselingen op tussen twee karakters.
    Wordt als {gesprek_context} meegegeven aan de LLM-prompt.
    """
    sleutel = _koppel_sleutel(naam1, naam2)
    geschiedenis = gesprekken_geheugen.get(sleutel, [])
    if not geschiedenis:
        return "Nog geen eerdere gesprekken tussen hen."
    fragmenten = geschiedenis[-max_regels:]
    return "\n".join([f'{g["spreker"]}: "{g["dialoog"]}"' for g in fragmenten])


def sla_gesprek_op(naam1: str, naam2: str, spreker: str, dialoog: str) -> None:
    """
    Bewaart een gesproken zin in het koppelgeheugen én in de individuele
    buffers van beide deelnemers.
    """
    sleutel = _koppel_sleutel(naam1, naam2)
    if sleutel not in gesprekken_geheugen:
        gesprekken_geheugen[sleutel] = []
    gesprekken_geheugen[sleutel].append({"spreker": spreker, "dialoog": dialoog})

    for naam in {naam1, naam2}:
        ander = naam2 if naam == naam1 else naam1
        voeg_memory_toe(naam, f"Gesprek met {ander}", f"{spreker}: {dialoog}")


# ---------------------------------------------------------------------------
# LangChain Tools
# ---------------------------------------------------------------------------

@tool
def valideer_kindvriendelijke_actie(actie: str) -> str:
    """
    Controleert of een gekozen actie veilig, geldig en kindvriendelijk is.
    Als de actie onbekend of onveilig is, valt de engine terug op 'rust'.
    """
    actie = actie.strip().lower()
    if filter_llm_actie is not None:
        try:
            return filter_llm_actie(actie)
        except Exception:
            pass
    return actie if actie in TOEGESTANE_ACTIES else "rust"


@tool
def geef_welzijnsadvies(stemming: str, honger: int) -> str:
    """
    Geeft ethisch welzijnsadvies wanneer een Sim hongerig, moe of verdrietig is.
    honger: 0 = helemaal vol, 100 = uitgehongerd.
    """
    if geef_ethisch_advies is not None:
        try:
            advies = geef_ethisch_advies(stemming, honger)
            return advies or "Geen bijzonder welzijnsadvies."
        except Exception:
            pass
    if honger >= 80:
        return "De Sim is uitgehongerd (honger >= 80). Eten is nu verstandig."
    if honger >= 60:
        return "De Sim begint honger te krijgen. Eten is straks nodig."
    if stemming in {"verdrietig", "bang", "gestrest", "eenzaam"}:
        return "De Sim heeft steun nodig. Praten of rusten is verstandig."
    return "Geen bijzonder welzijnsadvies."


# ---------------------------------------------------------------------------
# LLM & parsers
# ---------------------------------------------------------------------------

llm = ChatOllama(model="llama3.2", temperature=0.2)
parser = PydanticOutputParser(pydantic_object=SimBeslissing)
nora_parser = PydanticOutputParser(pydantic_object=NoraReactie)


# ---------------------------------------------------------------------------
# RAG-keten
# ---------------------------------------------------------------------------

context_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Je bent een contextplanner voor een kindvriendelijke Sims-tijdreisgame. "
        "Bepaal welke informatie nodig is om een goede beslissing te nemen.",
    ),
    (
        "human",
        "Sim: {naam}\nPersoonlijkheid: {persoonlijkheid}\nTijdperk: {tijdperk}\n"
        "Stemming: {stemming}\nHonger: {honger_label}\n"
        "Objecten nabij: {objecten_nabij}\nInstructie: {instructie}\n\n"
        "Geef een korte zoekvraag voor RAG. Alleen de zoekvraag, geen uitleg.",
    ),
])

rag_query_chain = context_prompt | llm | RunnableLambda(lambda msg: getattr(msg, "content", str(msg)).strip())


def _veilige_instructie(instructie: str) -> str:
    if filter_speler_instructie is not None:
        try:
            return filter_speler_instructie(instructie)
        except Exception:
            return "geen"
    return instructie if instructie else "geen"


def _haal_rag_context(input_data: SimInput) -> Dict[str, str]:
    zoekvraag = rag_query_chain.invoke(input_data)
    wereld_context = "Geen wereldcontext beschikbaar."
    karakter_context = "Geen karaktercontext beschikbaar."

    if vraag_tijdreis_kennisbank is not None:
        wereld_context = vraag_tijdreis_kennisbank.invoke(
            {"tijdperk": input_data["tijdperk"], "zoekvraag": zoekvraag}
        )
    if vraag_karakter_biografie is not None:
        karakter_context = vraag_karakter_biografie.invoke(
            {"naam": input_data["naam"], "zoekvraag": zoekvraag}
        )

    return {
        "rag_zoekvraag": zoekvraag,
        "wereld_context": wereld_context,
        "karakter_context": karakter_context,
    }


def _lege_rag_context(input_data: SimInput) -> Dict[str, str]:
    return {
        "rag_zoekvraag": "RAG uitgeschakeld.",
        "wereld_context": "Geen.",
        "karakter_context": "Geen.",
    }


rag_context_chain = RunnableLambda(_haal_rag_context)


# ---------------------------------------------------------------------------
# Hoofd-prompt  (gebruikt honger_label – nooit het ruwe getal)
# ---------------------------------------------------------------------------

beslis_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Je bent het beslisbrein van een kindvriendelijke Sims-tijdreisgame. "
        "Kies één actie voor de Sim. Toegestane acties: beweeg, eet, rust, praat.\n\n"
        "HONGER-SCHAAL (lees dit goed!):\n"
        "  'helemaal vol'          → de Sim heeft GEEN honger → kies NOOIT 'eet'\n"
        "  'licht verzadigd'       → nauwelijks honger → kies liever 'beweeg' of 'praat'\n"
        "  'een beetje honger'     → niet urgent → kies 'beweeg', 'praat' of 'rust'\n"
        "  'behoorlijk hongerig'   → eten mag, maar is niet verplicht\n"
        "  'erg hongerig'          → 'eet' is de beste keuze\n"
        "  'uitgehongerd'          → MOET nu eten → kies altijd 'eet'\n\n"
        "Voorbeelden:\n"
        "  honger='helemaal vol'   → {{\"actie\":\"beweeg\", \"dialoog\":\"Wat een mooie dag!\", \"emotie\":\"blij\", \"gebruikt_rag\":true}}\n"
        "  honger='licht verzadigd'→ {{\"actie\":\"praat\", \"dialoog\":\"Zullen we samen spelen?\", \"emotie\":\"blij\", \"gebruikt_rag\":true}}\n"
        "  honger='erg hongerig'   → {{\"actie\":\"eet\",   \"dialoog\":\"Ik heb honger!\", \"emotie\":\"hongerig\", \"gebruikt_rag\":true}}\n"
        "  honger='uitgehongerd'   → {{\"actie\":\"eet\",   \"dialoog\":\"Ik moet nu eten!\", \"emotie\":\"hongerig\", \"gebruikt_rag\":true}}\n\n"
        "Als je praat met Nora: reageer op het laatste nieuws.\n"
        "Als je praat met een andere Sim: reageer op wat die Sim zojuist zei.\n"
        "Eerdere gesprekken met deze persoon:\n{gesprek_context}\n\n"
        "{format_instructions}"
    ),
    (
        "human",
        "Sim: {naam}\n"
        "Persoonlijkheid: {persoonlijkheid}\n"
        "Tijdperk: {tijdperk}\n"
        "Honger: {honger_label}\n"
        "Stemming: {stemming}\n"
        "Objecten nabij: {objecten_nabij}\n"
        "Spelerinstructie: {instructie}\n\n"
        "Herinneringen:\n{herinneringen}\n\n"
        "RAG-zoekvraag: {rag_zoekvraag}\n"
        "Wereldcontext: {wereld_context}\n"
        "Karaktercontext: {karakter_context}\n"
        "Welzijnsadvies: {welzijnsadvies}\n\n"
        "Laatste nieuws van Nora: {laatste_nieuws}\n"
        "Buurman/buurvrouw: {buur_naam}\n"
        "Wat {buur_naam} zojuist zei: {buur_dialoog}\n\n"
        "Kies nu de beste actie als geldige JSON."
    ),
])


def _maak_prompt_input(data: Dict) -> Dict:
    basis = data["basis"]
    rag = data["rag"]
    return {
        **basis,
        **rag,
        "format_instructions": parser.get_format_instructions(),
        "welzijnsadvies": geef_welzijnsadvies.invoke(
            {"stemming": basis["stemming"], "honger": basis["honger"]}
        ),
    }


# ---------------------------------------------------------------------------
# LCEL orkestratie
# ---------------------------------------------------------------------------

def _maak_beslis_chain(gebruik_rag: bool):
    gekozen_rag = rag_context_chain if gebruik_rag else RunnableLambda(_lege_rag_context)
    return (
        RunnableParallel(basis=RunnablePassthrough(), rag=gekozen_rag)
        | RunnableLambda(_maak_prompt_input)
        | beslis_prompt
        | llm
        | parser
    )


class SimDecisionAgent:
    def invoke(self, input_data: SimInput, gebruik_rag: bool = False) -> SimBeslissing:
        chain = _maak_beslis_chain(gebruik_rag=gebruik_rag)
        return chain.invoke(input_data)


sim_decision_agent = SimDecisionAgent()


# ---------------------------------------------------------------------------
# Nora-keten
# ---------------------------------------------------------------------------

nora_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Je bent Nora, een vriendelijke nieuwslezer in een kindvriendelijke Sims-tijdreisgame. "
        "Je hebt zojuist nieuws gelezen en reageert kort en kindvriendelijk op wat een Sim tegen je zegt. "
        "Max 25 woorden. Verwijs naar het nieuws als dat relevant is. "
        "Gebruik eerdere gesprekken als context.\n"
        "Eerdere gesprekken:\n{gesprek_context}\n\n"
        "{format_instructions}"
    ),
    (
        "human",
        "Nora's stemming: {nora_stemming}\n"
        "Nora's persoonlijkheid: {nora_persoonlijkheid}\n"
        "Laatste nieuws dat Nora las: {laatste_nieuws}\n"
        "Sim {sim_naam} zegt tegen Nora: '{sim_dialoog}'\n\n"
        "Wat zegt Nora terug? Geef geldige JSON."
    ),
])

nora_chain = nora_prompt | llm | nora_parser

NORA_FALLBACKS = [
    "Interessant! Vertel eens meer.",
    "Dat klinkt spannend, wat bedoel je precies?",
    "Hmm, daar moet ik even over nadenken.",
    "Goed om te weten! Ik heb ook wat nieuws gehoord.",
    "Wat fijn dat je dat deelt!",
]

_nora_fallback_index = 0


def kies_reactie_voor_nora(
    sim_naam: str,
    sim_dialoog: str,
    nora: Dict,
    gesprek_context: str = "Nog geen eerdere gesprekken.",
) -> NoraReactie:
    """
    Laat Nora via de LLM reageren op wat een Sim tegen haar zei.
    Gebruikt het nora-dict voor stemming, persoonlijkheid en laatste nieuws.
    """
    global _nora_fallback_index
    try:
        return nora_chain.invoke({
            "sim_naam": sim_naam,
            "sim_dialoog": sim_dialoog,
            "nora_stemming": nora.get("stemming", "neutraal"),
            "nora_persoonlijkheid": nora.get("persoonlijkheid", "rustig"),
            "laatste_nieuws": nora.get("laatste_nieuws") or "Nog geen nieuws beschikbaar.",
            "gesprek_context": gesprek_context,
            "format_instructions": nora_parser.get_format_instructions(),
        })
    except Exception as e:
        print(f"⚠️ Nora LLM fout: {e}")
        zin = NORA_FALLBACKS[_nora_fallback_index % len(NORA_FALLBACKS)]
        _nora_fallback_index += 1
        return NoraReactie(reactie=zin, emotie="neutraal")


# ---------------------------------------------------------------------------
# Fallback & exportfuncties
# ---------------------------------------------------------------------------

def _fallback_beslissing(sim: Dict, objecten_nabij: List[str]) -> SimBeslissing:
    """Deterministische fallback als de LLM/RAG tijdelijk faalt."""
    honger = sim.get("honger", 0)
    stemming = sim.get("stemming", "neutraal")

    if honger >= 80:
        actie: Actie = "eet"
        dialoog = "Ik heb honger, ik ga eten."
    elif stemming in {"verdrietig", "bang", "gestrest", "eenzaam"}:
        actie = "praat"
        dialoog = "Ik voel me niet zo goed, kan ik even praten?"
    elif stemming == "moe":
        actie = "rust"
        dialoog = "Ik ben moe, ik ga even rusten."
    else:
        actie = "beweeg"
        dialoog = "Ik ga even een stukje lopen."

    return SimBeslissing(
        actie=actie,
        reden="Fallback op basis van vaste spellogica.",
        dialoog=dialoog,
        emotie=stemming if stemming in Emotie.__args__ else "neutraal",
        gebruikt_rag=False,
    )


def kies_actie_voor_sim_dict(
    sim: Dict,
    objecten_nabij: List[str],
    tijdperk: str = "prehistorie",
    instructie: str = "geen",
    gebruik_rag: bool = False,
    debug: bool = False,
    laatste_nieuws: str = "Geen nieuws.",
    buur_naam: str = "geen",
    buur_dialoog: str = "Niets bijzonders.",
    gesprek_context: str = "Nog geen gesprekken.",
) -> Dict | str:
    """Hoofdfunctie: vraagt via LangChain een beslissing op voor één Sim."""
    veilige_instructie = _veilige_instructie(instructie)
    honger_int = int(sim.get("honger", 0))

    input_data: SimInput = {
        "naam": sim["naam"],
        "persoonlijkheid": sim["persoonlijkheid"],
        "honger": honger_int,
        "honger_label": honger_label(honger_int),   # ← label, geen getal
        "stemming": sim["stemming"],
        "objecten_nabij": ", ".join(objecten_nabij) if objecten_nabij else "niets",
        "instructie": veilige_instructie,
        "tijdperk": tijdperk,
        "herinneringen": geef_memory_context(sim["naam"]),
        "laatste_nieuws": laatste_nieuws,
        "buur_naam": buur_naam,
        "buur_dialoog": buur_dialoog,
        "gesprek_context": gesprek_context,
    }

    try:
        beslissing = sim_decision_agent.invoke(input_data, gebruik_rag=gebruik_rag)
        actie = valideer_kindvriendelijke_actie.invoke(beslissing.actie)
        beslissing.actie = actie
        fallback = False
    except Exception as exc:
        beslissing = _fallback_beslissing(sim, objecten_nabij)
        fallback = True
        if debug:
            return beslissing.model_dump() | {"error": str(exc), "fallback": True}

    voeg_memory_toe(
        sim["naam"],
        input_text=f"Honger: {honger_label(honger_int)}, objecten: {objecten_nabij}",
        output_text=f"Koos '{beslissing.actie}' omdat: {beslissing.reden}",
    )

    if debug:
        return beslissing.model_dump() | {"fallback": fallback}
    return beslissing.actie


def registreer_resultaat(naam: str, gebeurtenis: str) -> None:
    """Sla een actieresultaat achteraf op in het memory van de Sim."""
    voeg_memory_toe(naam, "Resultaat na actie", gebeurtenis)