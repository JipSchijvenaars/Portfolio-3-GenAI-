import os
# from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# load_dotenv()
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_API_KEY= "JOUW-API-KEY-HIER"

TOEGESTANE_ACTIES = ["beweeg", "eet", "rust", "praat"]
USE_MOCK_LLM = True

def maak_actie_geldig(llm_output):
    actie = llm_output.strip().lower()

    if actie in TOEGESTANE_ACTIES:
        return actie

    return "rust"

actie_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Je bent {naam}, een bewoner van een klein dorp. "
     "Jouw persoonlijkheid: {persoonlijkheid}. "
     "Gedraag je zoals jouw persoonlijkheid dat zou doen. "
     "Gebruik altijd vriendelijke, kindvriendelijke taal."),
    ("human",
     "Jouw situatie:\n"
     "- Honger: {honger}/10\n"
     "- Stemming: {stemming}\n"
     "- Objecten in de buurt: {objecten_nabij}\n"
     "- Instructie van speler: {instructie}\n\n"
     "Kies precies één actie: beweeg, eet, rust, of praat. "
     "Geef alleen dat ene woord terug, verder niets.")
])

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
      google_api_key=GOOGLE_API_KEY,
        temperature=0.5
        )

kies_actie_chain = actie_prompt | llm | StrOutputParser()

def kies_actie_voor_sim(naam, persoonlijkheid, honger, stemming, objecten_nabij, instructie, debug=False):
    if USE_MOCK_LLM:
        llm_output = "eet" if honger <= 4 and "supermarkt" in objecten_nabij else "beweeg"
    else:
        llm_output = kies_actie_chain.invoke({
            "naam": naam,
            "persoonlijkheid": persoonlijkheid,
            "honger": honger,
            "stemming": stemming,
            "objecten_nabij": objecten_nabij,
            "instructie": instructie,
        })

    geldige_actie = maak_actie_geldig(llm_output)

    if debug:
        return {
            "llm_output": llm_output,
            "actie": geldige_actie,
            "mock": USE_MOCK_LLM
        }

    return geldige_actie

def kies_actie_voor_sim_dict(sim, objecten_nabij, instructie="geen", debug=False):

    return kies_actie_voor_sim(

        naam=sim["naam"],

        persoonlijkheid=sim["persoonlijkheid"],

        honger=sim["honger"],

        stemming=sim["stemming"],

        objecten_nabij=objecten_nabij,

        instructie=instructie,

        debug=debug

    )