from __future__ import annotations 

from typing import Literal, Dict
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_ollama import ChatOllama

import os
import requests
from dotenv import load_dotenv

load_dotenv()

NEWS_API_KEY = os.getenv("NEWS_API_KEY")


# Probeer memoryfunctie uit je bestaande LangChain-bestand te importeren
try:
    from langchain_logic_100 import registreer_resultaat
except Exception:
    registreer_resultaat = None


NieuwsEmotie = Literal[
    "blij",
    "verdrietig",
    "bezorgd",
    "enthousiast",
    "rustig",
    "nieuwsgierig"
]


class NieuwsImpact(BaseModel):
    emotie: NieuwsEmotie = Field(
        description="De kindvriendelijke emotie die het nieuws oproept"
    )
    intensiteit: int = Field(
        description="Hoe sterk de emotie is, van 1 tot 5"
    )
    kindvriendelijke_samenvatting: str = Field(
        description="Zeer algemene, veilige samenvatting zonder heftige details"
    )


llm = ChatOllama(model="llama3.2", temperature=0)

nieuws_parser = PydanticOutputParser(pydantic_object=NieuwsImpact)

nieuws_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Je analyseert nieuws voor een kindvriendelijke dorpssimulatie. "
            "Geef nooit heftige details door zoals geweld, dood, misdaad, oorlog, rampen of volwassen thema's. "
            "Vertaal het nieuws alleen naar een algemene emotie en een veilige, abstracte samenvatting. "
            "De dorpsbewoners mogen alleen de emotionele lading ervaren, niet het letterlijke nieuws.\n\n"
            "{format_instructions}"
        ),
        (
            "human",
            "Nieuwsbericht:\n{nieuws_text}\n\n"
            "Bepaal de emotie, intensiteit en veilige samenvatting."
        ),
    ]
)

nieuws_chain = nieuws_prompt | llm | nieuws_parser


def breng_nieuws_naar_dorp(
    nieuws_text: str,
    ontvanger: Dict,
) -> Dict:
    """
    Eén dorpsbewoner ontvangt nieuws van buiten het dorp.
    Het letterlijke nieuws wordt niet verspreid; alleen de emotie.
    """

    impact = nieuws_chain.invoke(
        {
            "nieuws_text": nieuws_text,
            "format_instructions": nieuws_parser.get_format_instructions(),
        }
    )

    ontvanger["stemming"] = impact.emotie

    if registreer_resultaat is not None:
        registreer_resultaat(
            ontvanger["naam"],
            f"{ontvanger['naam']} hoorde nieuws van buiten het dorp en voelde zich {impact.emotie}."
        )

    return {
        "ontvanger": ontvanger["naam"],
        "emotie": impact.emotie,
        "intensiteit": impact.intensiteit,
        "kindvriendelijke_samenvatting": impact.kindvriendelijke_samenvatting,
    }


def deel_nieuws_emotie(
    bron_sim: Dict,
    doel_sim: Dict,
) -> None:
    """
    De bron-Sim deelt niet het letterlijke nieuws, maar alleen de emotie.
    """

    doel_sim["stemming"] = bron_sim["stemming"]

    if registreer_resultaat is not None:
        registreer_resultaat(
            doel_sim["naam"],
            f"{doel_sim['naam']} sprak met {bron_sim['naam']} en voelde zich daarna {doel_sim['stemming']}."
        )

def haal_nieuws_op(query="ruimtevaart"):
    import os
    import requests

    api_key = os.getenv("NEWS_API_KEY")

    if not api_key:
        return "Geen NEWS_API_KEY gevonden. Controleer je .env bestand."

    url = "https://newsapi.org/v2/everything"

    params = {
        "q": query,
        "language": "nl",
        "pageSize": 1,
        "sortBy": "publishedAt",
        "apiKey": api_key,
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
    except Exception as e:
        return f"Nieuws ophalen mislukt: {e}"

    data = response.json()
    artikelen = data.get("articles", [])

    if not artikelen:
        return "Geen nieuws gevonden."

    artikel = artikelen[0]

    titel = artikel.get("title", "")
    beschrijving = artikel.get("description", "")

    return f"{titel}. {beschrijving}"

def bepaal_dorpsemotie_uit_nieuws(nieuws_tekst):
    """
    Zet een nieuwsbericht om naar alleen een veilige dorpsemotie.
    Het nieuws zelf wordt niet gedeeld met het dorp.
    """

    try:
        from langchain_ollama import ChatOllama
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser

        prompt = ChatPromptTemplate.from_template("""
Je bent een neutrale nieuwsduider voor een kindvriendelijke Sims-wereld.

Lees het nieuwsbericht, maar herhaal geen details uit het nieuws.
Bepaal alleen welke algemene emotie het dorp hiervan krijgt.

Kies precies één emotie uit deze lijst:
- blij
- nieuwsgierig
- bezorgd
- rustig
- verdrietig
- hoopvol

Regels:
- Geef alleen het ene emotiewoord terug.
- Geen uitleg.
- Geen nieuwsdetails.
- Geen politieke of gewelddadige details.
- Houd het geschikt voor kinderen vanaf 8 jaar.

Nieuwsbericht:
{nieuws}
""")

        llm = ChatOllama(model="llama3.2", temperature=0.2)

        chain = prompt | llm | StrOutputParser()

        emotie = chain.invoke({"nieuws": nieuws_tekst}).strip().lower()

        toegestane_emoties = [
            "blij",
            "nieuwsgierig",
            "bezorgd",
            "rustig",
            "verdrietig",
            "hoopvol",
        ]

        if emotie not in toegestane_emoties:
            return "nieuwsgierig"

        return emotie

    except Exception:
        return "nieuwsgierig"