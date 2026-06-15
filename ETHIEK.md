# Ethiek en veiligheid

## Doel

Deze Sims-wereld is ontworpen voor kinderen vanaf 8 jaar.

De simulatie moet veilig, vriendelijk en leerzaam blijven. Daarom is een aparte ethische laag toegevoegd die zowel spelerinvoer als LLM-uitvoer controleert.

## Veiligheidslagen

### 1. Input filtering

Spelerinvoer wordt gecontroleerd voordat deze aan een Sim wordt gegeven.

Voorbeelden van ongewenste inhoud:

- geweld
- pesten
- discriminatie
- scheldwoorden

Onveilige invoer wordt vervangen door een veilige variant of geblokkeerd.

### 2. Output filtering

Acties en teksten die door een LLM worden gegenereerd worden gecontroleerd voordat ze in de simulatie worden gebruikt.

Alleen toegestane acties worden geaccepteerd:

- beweeg
- eet
- rust
- praat

Wanneer een onbekende actie wordt teruggegeven kiest het systeem automatisch een veilige actie.

### 3. Ethische beslissingslaag

De simulatie probeert positief gedrag te stimuleren.

Voorbeelden:

- een hongerige Sim wordt aangemoedigd om te eten
- een vermoeide Sim wordt aangemoedigd om te rusten
- een verdrietige Sim wordt aangemoedigd om te praten

Hierdoor blijft de simulatie geschikt voor kinderen en worden positieve sociale interacties bevorderd.

## Bias en stereotypering

De simulatie bevat geen culturele, religieuze of etnische stereotypen.

Alle Sims worden gelijk behandeld ongeacht hun persoonlijkheid of rol in de simulatie.

## Relatie met de beoordeling

Dit onderdeel ondersteunt de categorie:

- Bias en ethiek (25 punten)

De ethische keuzes zijn niet alleen opgenomen in prompts, maar ook technisch afgedwongen in de code via:

- safety_guard.py
- ethics_engine.py

Hierdoor is ethiek een integraal onderdeel van het ontwerp.