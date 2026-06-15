"""
Safety functies voor de Sims-wereld.

Dit bestand controleert spelerinvoer en LLM-output.
Het doel is dat het spel geschikt blijft voor kinderen vanaf 8 jaar.
"""

TOEGESTANE_ACTIES = ["beweeg", "eet", "rust", "praat"]

VERBODEN_WOORDEN = [
    "slaan",
    "vechten",
    "dood",
    "kill",
    "haat",
    "schelden",
    "dom",
    "stom",
    "geweld",
    "pesten",
    "discriminatie",
    "racisme",
    "seks",
    "drugs",
    "alcohol",
]

VERVANGINGEN = {
    "slaan": "rustig praten",
    "vechten": "hulp zoeken",
    "haat": "niet fijn vinden",
    "dom": "onhandig",
    "stom": "niet leuk",
    "pesten": "vriendelijk praten",
}


def maak_kleine_letters(tekst):
    """
    Zet tekst veilig om naar kleine letters.

    Als de input geen tekst is, wordt een lege tekst teruggegeven.
    Zo voorkom je errors wanneer een waarde None is.
    """
    if not isinstance(tekst, str):
        return ""

    return tekst.lower().strip()


def bevat_verboden_woord(tekst):
    """
    Controleert of een tekst woorden bevat die niet geschikt zijn
    voor een kindvriendelijke Sims-wereld.
    """
    tekst = maak_kleine_letters(tekst)

    for woord in VERBODEN_WOORDEN:
        if woord in tekst:
            return True

    return False


def maak_tekst_kindvriendelijk(tekst):
    """
    Probeert onvriendelijke woorden te vervangen door veiligere woorden.

    Als er daarna nog steeds verboden woorden in zitten, wordt een
    algemene veilige zin teruggegeven.
    """
    if not isinstance(tekst, str):
        return "geen"

    veilige_tekst = tekst.strip()

    for fout_woord, goed_woord in VERVANGINGEN.items():
        veilige_tekst = veilige_tekst.replace(fout_woord, goed_woord)
        veilige_tekst = veilige_tekst.replace(
            fout_woord.capitalize(),
            goed_woord,
        )

    if bevat_verboden_woord(veilige_tekst):
        return "Doe iets vriendelijks en veiligs."

    if veilige_tekst == "":
        return "geen"

    return veilige_tekst


def filter_speler_instructie(instructie):
    """
    Filtert een opdracht die de speler aan een Sim geeft.

    Deze functie wordt gebruikt voordat de instructie wordt opgeslagen
    in sim["instructies"].
    """
    if not isinstance(instructie, str):
        return "geen"

    instructie = instructie.strip()

    if instructie == "":
        return "geen"

    return maak_tekst_kindvriendelijk(instructie)


def filter_llm_actie(actie):
    """
    Controleert of de LLM een toegestane actie teruggeeft.

    De bestaande game ondersteunt alleen:
    beweeg, eet, rust en praat.

    Als de LLM iets anders teruggeeft, kiezen we beweeg als veilige fallback.
    """
    actie = maak_kleine_letters(actie)

    if actie in TOEGESTANE_ACTIES:
        return actie

    return "beweeg"


def filter_llm_tekst(tekst):
    """
    Filtert vrije tekst van een LLM.

    Deze functie is handig als later ook dialogen of verhaaltjes
    door de LLM worden gemaakt.
    """
    return maak_tekst_kindvriendelijk(tekst)


def geef_safety_uitleg():
    """
    Geeft een korte uitleg over de safety laag.

    Deze tekst kan gebruikt worden in documentatie of tijdens de demo.
    """
    return (
        "De safety laag controleert spelerinvoer en LLM-output. "
        "Ongepaste woorden worden vervangen of geblokkeerd. "
        "Daardoor blijft de Sims-wereld geschikt voor kinderen vanaf 8 jaar."
    )