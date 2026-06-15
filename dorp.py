# dorp.py

# Kaart- en GUI-configuratie
GRID_SIZE = 15
CELL_SIZE = 40
DEFAULT_SPEED = 2000

# Speldata Objecten
OBJECTEN = {
    "🌳": "boom",
    "🏠": "huis",
    "☕": "cafe",
    "🛒": "supermarkt",
    "📰": "kiosk",
    "🌿": "park",
}

# De 5 bewoners van het dorp
BEWONERS = [
    {"naam": "Emma",   "icon": "😊", "persoonlijkheid": "optimistisch en sociaal"},
    {"naam": "Lars",   "icon": "😏", "persoonlijkheid": "cynisch maar grappig"},
    {"naam": "Fatima", "icon": "🌸", "persoonlijkheid": "zorgzaam en empathisch"},
    {"naam": "Daan",   "icon": "😐", "persoonlijkheid": "nuchter en praktisch"},
    {"naam": "Sofia",  "icon": "🎨", "persoonlijkheid": "creatief en gevoelig"},
]

# Vaste posities van alle gebouwen en bomen op de kaart
VASTE_OBJECTEN = [
    # Bomen in de hoeken
    (0, 0, "🌳"), (14, 0, "🌳"), (0, 14, "🌳"), (14, 14, "🌳"),
    (1, 0, "🌳"), (0, 1, "🌳"), (13, 0, "🌳"), (14, 1, "🌳"),
    # Park in het midden
    (6, 6, "🌿"), (7, 6, "🌿"), (8, 6, "🌿"),
    (6, 7, "🌿"), (7, 7, "🌿"), (8, 7, "🌿"),
    (6, 8, "🌿"), (7, 8, "🌿"), (8, 8, "🌿"),
    # Café
    (3, 3, "☕"),
    # Supermarkt
    (11, 3, "🛒"),
    # Kiosk
    (7, 2, "📰"),
    # Huizen
    (2, 11, "🏠"), (5, 12, "🏠"), (9, 12, "🏠"), (12, 11, "🏠"), (7, 11, "🏠"),
]

# Startposities voor de 5 bewoners
STARTPOSITIES = [(3, 7), (7, 4), (11, 7), (5, 5), (10, 10)]