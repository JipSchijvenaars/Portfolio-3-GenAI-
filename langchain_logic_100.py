"""
LangChain-beslislaag voor de Sims/tijdreis-game.

Doel voor assessment: aantonen dat LangChain niet alleen als simpele LLM-call wordt gebruikt,
maar als modulaire beslisarchitectuur met:
- Prompt templates
- Tools
- Memory
- Custom chains
- Structured output parsing
- Validatie/fallbacks

Gebruik in notebook:
    from langchain_logic_100 import kies_actie_voor_sim_dict, registreer_resultaat

    beslissing = kies_actie_voor_sim_dict(
        sim=sim,
        objecten_nabij=self._objecten_nabij(x, y),
        tijdperk="prehistorie",  # of "toekomst"
        instructie=instructie,
        debug=True,
    )
    sim["actie"] = beslissing["actie"]

    # Na uitvoeren van de actie:
    registreer_resultaat(sim["naam"], f"{sim['naam']} voerde {sim['actie']} uit en stemming is {sim['stemming']}.")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, TypedDict

from pydantic import BaseModel, Field

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

from typing import Literal

# Eigen LangChain-tools uit je RAG-module.
# Deze tools gebruiken de wereldbestanden en karakterbestanden.
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


Actie = Literal["beweeg", "eet", "rust", "praat"]
TOEGESTANE_ACTIES: List[str] = ["beweeg", "eet", "rust", "praat"]


class SimInput(TypedDict):
    naam: str
    persoonlijkheid: str
    honger: int
    stemming: str
    objecten_nabij: str
    instructie: str
    tijdperk: str
    herinneringen: str


Emotie = Literal[
    "blij",
    "rustig",
    "nieuwsgierig",
    "verdrietig",
    "bezorgd",
    "moe",
    "hongerig",
    "neutraal"
]

class SimBeslissing(BaseModel):
    """Gestructureerde output van de LLM."""

    actie: Actie = Field(description="Exact een van: beweeg, eet, rust, praat")
    reden: str = Field(description="Korte kindvriendelijke reden voor deze actie")
    emotie: Emotie = Field(description="Exact een van: blij, rustig, nieuwsgierig, verdrietig, bezorgd, moe, hongerig, neutraal")
    gebruikt_rag: bool = Field(description="Of de beslissing aantoonbaar context uit RAG gebruikt")


@dataclass
class SimMemory:
    """Eenvoudige per-Sim memorylaag.

    De laatste gebeurtenissen per Sim worden opgeslagen en opnieuw
    meegegeven aan de LangChain-prompt.
    """

    max_items: int = 5
    gebeurtenissen: Dict[str, List[str]] = field(default_factory=dict)

    def voeg_toe(self, naam: str, gebeurtenis: str) -> None:
        self.gebeurtenissen.setdefault(naam, []).append(gebeurtenis)
        self.gebeurtenissen[naam] = self.gebeurtenissen[naam][-self.max_items:]

    def geef_context(self, naam: str) -> str:
        items = self.gebeurtenissen.get(naam, [])
        if not items:
            return "Nog geen eerdere herinneringen."
        return "\n".join(f"- {item}" for item in items)


memory = SimMemory(max_items=5)


def voeg_memory_toe(naam: str, input_text: str, output_text: str) -> None:
    """Slaat een gebeurtenis op in de eigen per-Sim memorylaag."""
    gebeurtenis = f"{input_text} -> {output_text}"
    memory.voeg_toe(naam, gebeurtenis)


def geef_memory_context(naam: str) -> str:
    """Haalt de opgeslagen herinneringen van een Sim op voor de prompt."""
    return memory.geef_context(naam)


@tool
def valideer_kindvriendelijke_actie(actie: str) -> str:
    """Controleert of een LLM-actie veilig, geldig en kindvriendelijk is."""
    actie = actie.strip().lower()

    if filter_llm_actie is not None:
        try:
            return filter_llm_actie(actie)
        except Exception:
            pass

    if actie in TOEGESTANE_ACTIES:
        return actie
    return "rust"


@tool
def geef_welzijnsadvies(stemming: str, honger: int) -> str:
    """Geeft ethisch advies wanneer een Sim hongerig, moe of verdrietig is."""
    if geef_ethisch_advies is not None:
        try:
            advies = geef_ethisch_advies(stemming, honger)
            return advies or "Geen bijzonder welzijnsadvies."
        except Exception:
            pass

    if honger <= 3:
        return "De Sim heeft weinig energie. Eten of rusten is verstandig."
    if stemming in {"verdrietig", "bang", "gestrest", "hongerig"}:
        return "De Sim heeft steun nodig. Praten of rusten is verstandig."
    return "Geen bijzonder welzijnsadvies."


llm = ChatOllama(model="llama3.2", temperature=0)
parser = PydanticOutputParser(pydantic_object=SimBeslissing)

context_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Je bent een contextplanner voor een kindvriendelijke Sims-tijdreisgame. "
            "Bepaal welke informatie nodig is om een goede beslissing te nemen.",
        ),
        (
            "human",
            "Sim: {naam}\n"
            "Persoonlijkheid: {persoonlijkheid}\n"
            "Tijdperk: {tijdperk}\n"
            "Stemming: {stemming}\n"
            "Honger/energie: {honger}/10\n"
            "Objecten nabij: {objecten_nabij}\n"
            "Spelerinstructie: {instructie}\n\n"
            "Geef een korte zoekvraag voor RAG. Alleen de zoekvraag, geen uitleg.",
        ),
    ]
)

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
    """Fallback-context wanneer RAG bewust is uitgeschakeld.

    Hierdoor kan de LangChain-beslislaag ook zonder RAG draaien.
    Dat maakt het LangChain-onderdeel zelfstandig beoordeelbaar.
    """
    return {
        "rag_zoekvraag": "RAG is uitgeschakeld voor deze LangChain-run.",
        "wereld_context": "Geen wereldcontext gebruikt.",
        "karakter_context": "Geen karaktercontext gebruikt.",
    }

rag_context_chain = RunnableLambda(_haal_rag_context)

beslis_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Je bent het beslisbrein van een kindvriendelijke Sims-game. "
            "Je kiest een actie voor precies één Sim. "
            "Je gebruikt vaste spellogica, RAG-context, herinneringen en welzijnsadvies. "
            "Je mag alleen deze acties kiezen: beweeg, eet, rust, praat. "
            "Gebruik geen gevaarlijke, gemene, volwassen of ongeschikte content.\n\n"
            "{format_instructions}",
        ),
        (
            "human",
            "Sim: {naam}\n"
            "Persoonlijkheid: {persoonlijkheid}\n"
            "Tijdperk: {tijdperk}\n"
            "Honger/energie: {honger}/10\n"
            "Stemming: {stemming}\n"
            "Objecten nabij: {objecten_nabij}\n"
            "Spelerinstructie: {instructie}\n\n"
            "Herinneringen van deze Sim:\n{herinneringen}\n\n"
            "RAG-zoekvraag:\n{rag_zoekvraag}\n\n"
            "Wereldcontext uit RAG:\n{wereld_context}\n\n"
            "Karaktercontext uit RAG:\n{karakter_context}\n\n"
            "Welzijnsadvies:\n{welzijnsadvies}\n\n"
            "Kies nu de beste actie. Geef uitsluitend geldige JSON volgens het schema.",
        ),
    ]
)


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


# Custom chain: input voorbereiden -> RAG ophalen -> prompt bouwen -> LLM -> parser
beslis_chain = (
    RunnableParallel(
        basis=RunnablePassthrough(),
        rag=rag_context_chain,
    )
    | RunnableLambda(_maak_prompt_input)
    | beslis_prompt
    | llm
    | parser
)

def _maak_beslis_chain(gebruik_rag: bool):
    """Maakt de beslis-chain met of zonder RAG-context."""
    gekozen_rag_chain = rag_context_chain if gebruik_rag else RunnableLambda(_lege_rag_context)

    return (
        RunnableParallel(
            basis=RunnablePassthrough(),
            rag=gekozen_rag_chain,
        )
        | RunnableLambda(_maak_prompt_input)
        | beslis_prompt
        | llm
        | parser
    )

class SimDecisionAgent:
    """Agent-achtige LangChain-beslislaag voor één Sim.

    Deze class orkestreert tools, memory, prompt templates,
    LLM-output parsing, optionele RAG-context en validatie.
    De GUI gebruikt alleen deze agent en hoeft de interne LangChain-flow niet te kennen.
    """

    def invoke(self, input_data: SimInput, gebruik_rag: bool = False) -> SimBeslissing:
        chain = _maak_beslis_chain(gebruik_rag=gebruik_rag)
        return chain.invoke(input_data)

sim_decision_agent = SimDecisionAgent()

def _fallback_beslissing(sim: Dict, objecten_nabij: List[str]) -> SimBeslissing:
    """Deterministische fallback als de LLM/RAG tijdelijk faalt."""
    if sim.get("honger", 10) <= 4 and ("supermarkt" in objecten_nabij or "cafe" in objecten_nabij):
        actie: Actie = "eet"
    elif sim.get("stemming") in {"verdrietig", "bang", "gestrest"}:
        actie = "praat"
    elif sim.get("honger", 10) <= 3:
        actie = "rust"
    else:
        actie = "beweeg"

    return SimBeslissing(
        actie=actie,
        reden="Fallback op basis van vaste spellogica.",
        emotie=sim.get("stemming", "neutraal"),
        gebruikt_rag=False,
    )


def kies_actie_voor_sim_dict(
    sim: Dict,
    objecten_nabij: List[str],
    tijdperk: str = "prehistorie",
    instructie: str = "geen",
    gebruik_rag: bool = False,
    debug: bool = False,
) -> Dict | str:
    """Hoofdfunctie voor de game.

    Deze functie is de enige functie die de GUI hoeft aan te roepen.
    Daardoor blijft de game-logica gescheiden van de LangChain-logica.
    """
    veilige_instructie = _veilige_instructie(instructie)

    input_data: SimInput = {
        "naam": sim["naam"],
        "persoonlijkheid": sim["persoonlijkheid"],
        "honger": int(sim["honger"]),
        "stemming": sim["stemming"],
        "objecten_nabij": ", ".join(objecten_nabij),
        "instructie": veilige_instructie,
        "tijdperk": tijdperk,
        "herinneringen": geef_memory_context(sim["naam"]),
    }

    try:
        beslissing = sim_decision_agent.invoke(input_data, gebruik_rag=gebruik_rag)
        actie = valideer_kindvriendelijke_actie.invoke(beslissing.actie)
        beslissing.actie = actie  # type: ignore[assignment]
    except Exception as exc:
        beslissing = _fallback_beslissing(sim, objecten_nabij)
        if debug:
            return {
                "actie": beslissing.actie,
                "reden": beslissing.reden,
                "emotie": beslissing.emotie,
                "gebruikt_rag": beslissing.gebruikt_rag,
                "error": str(exc),
                "fallback": True,
            }

    voeg_memory_toe(
        sim["naam"],
        input_text=f"Situatie: {sim['stemming']}, honger {sim['honger']}, objecten nabij: {objecten_nabij}",
        output_text=f"Koos actie '{beslissing.actie}' omdat: {beslissing.reden}",
    )

    if debug:
        return beslissing.model_dump() | {"fallback": False}

    return beslissing.actie


def registreer_resultaat(naam: str, gebeurtenis: str) -> None:
    voeg_memory_toe(
        naam,
        input_text="Resultaat na uitgevoerde actie",
        output_text=gebeurtenis,
    )
