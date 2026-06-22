"""
Ethische beslissingslaag voor de Sims-wereld.

Dit bestand helpt Sims om positieve, veilige en kindvriendelijke
keuzes te maken voordat een LLM een beslissing neemt.
"""


POSITIEVE_ACTIES = {
    "verdrietig": "praat",
    "eenzaam": "praat",
    "bang": "praat",
    "boos": "rust",
    "gestrest": "rust",
    "moe": "rust",
    "hongerig": "eet",
}


def normaliseer_stemming(stemming):
    """
    Zet een stemming om naar kleine letters.
    """
    if not isinstance(stemming, str):
        return "neutraal"

    return stemming.lower().strip()


def heeft_veel_honger(honger):
    """
    Controleert of een Sim erg veel honger heeft.

    In de huidige game loopt honger meestal van 0 tot 10.
    Een lage waarde betekent dat de Sim hongeriger is.
    """
    return honger <= 3


def heeft_honger(honger):
    """
    Controleert of een Sim honger heeft.
    """
    return honger <= 5


def geef_ethisch_advies(stemming, honger):
    """
    Geeft een aanbevolen actie terug.

    Deze functie probeert eenvoudige welzijnsregels toe te passen
    voordat de LLM-keuze wordt uitgevoerd.
    """
    stemming = normaliseer_stemming(stemming)

    if heeft_veel_honger(honger):
        return "eet"

    if stemming in POSITIEVE_ACTIES:
        return POSITIEVE_ACTIES[stemming]

    if heeft_honger(honger) and stemming in {"neutraal", "rustig"}:
        return "eet"

    return None


def beschrijf_ethische_keuze(stemming, honger):
    """
    Geeft uitleg waarom een bepaalde keuze wordt aanbevolen.
    """
    advies = geef_ethisch_advies(stemming, honger)

    if advies is None:
        return (
            "Er is geen directe ethische voorkeur. "
            "De Sim mag zelf een passende keuze maken."
        )

    if advies == "eet":
        return (
            "De Sim heeft honger en wordt aangemoedigd "
            "om eerst goed voor zichzelf te zorgen."
        )

    if advies == "rust":
        return (
            "De Sim heeft rust nodig en wordt aangemoedigd "
            "om even te ontspannen."
        )

    if advies == "praat":
        return (
            "De Sim voelt zich niet goed en wordt aangemoedigd "
            "om vriendelijk contact te zoeken."
        )

    return "Er is een positieve actie aanbevolen."


def is_welzijns_actie(actie):
    """
    Controleert of een actie bijdraagt aan gezondheid of sociaal welzijn.
    """
    return actie in {
        "eet",
        "rust",
        "praat",
    }


def score_actie(actie):
    """
    Geeft een eenvoudige welzijnsscore terug.

    Kan later gebruikt worden voor statistieken of evaluatie van Sim-gedrag.
    """
    scores = {
        "eet": 3,
        "rust": 3,
        "praat": 2,
        "beweeg": 1,
    }

    return scores.get(actie, 0)


def geef_ethiek_uitleg():
    """
    Korte uitleg voor documentatie en demo.
    """
    return (
        "De ethics engine stimuleert positieve keuzes. "
        "Wanneer een Sim hongerig, verdrietig, eenzaam, bang, boos, "
        "gestrest of moe is, wordt een veilige welzijnsactie aanbevolen."
    )