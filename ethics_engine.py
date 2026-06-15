"""
Ethische beslissingslaag voor de Sims-wereld.

Dit bestand helpt Sims om positieve en kindvriendelijke keuzes
te maken voordat een LLM een beslissing neemt.
"""


POSITIEVE_ACTIES = {
    "verdrietig": "praat",
    "eenzaam": "praat",
    "moe": "rust",
}


def heeft_veel_honger(honger):
    """
    Controleert of een Sim erg veel honger heeft.
    """
    return honger >= 80


def heeft_honger(honger):
    """
    Controleert of een Sim honger heeft.
    """
    return honger >= 60


def geef_ethisch_advies(stemming, honger):
    """
    Geeft een aanbevolen actie terug.

    Deze functie probeert eerst eenvoudige
    welzijnsregels toe te passen voordat de
    LLM een keuze maakt.
    """
    if heeft_veel_honger(honger):
        return "eet"

    stemming = str(stemming).lower().strip()

    if stemming in POSITIEVE_ACTIES:
        return POSITIEVE_ACTIES[stemming]

    return None


def beschrijf_ethische_keuze(stemming, honger):
    """
    Geeft uitleg waarom een bepaalde keuze
    wordt aanbevolen.
    """
    advies = geef_ethisch_advies(stemming, honger)

    if advies is None:
        return (
            "Er is geen directe ethische voorkeur. "
            "De Sim mag zelf een keuze maken."
        )

    if advies == "eet":
        return (
            "De Sim heeft veel honger en wordt "
            "aangemoedigd om eerst te eten."
        )

    if advies == "rust":
        return (
            "De Sim voelt zich moe en wordt "
            "aangemoedigd om uit te rusten."
        )

    if advies == "praat":
        return (
            "De Sim voelt zich niet goed en wordt "
            "aangemoedigd om sociaal contact te zoeken."
        )

    return "Er is een positieve actie aanbevolen."


def is_welzijns_actie(actie):
    """
    Controleert of een actie bijdraagt aan
    gezondheid of sociaal welzijn.
    """
    return actie in {
        "eet",
        "rust",
        "praat",
    }


def score_actie(actie):
    """
    Geeft een eenvoudige welzijnsscore terug.

    Kan later gebruikt worden voor statistieken
    of evaluatie van Sim-gedrag.
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
        "Wanneer een Sim erg hongerig, moe of verdrietig is, "
        "wordt een actie aanbevolen die het welzijn verbetert."
    )