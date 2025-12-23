"""
Utility constants and functions for the medical chatbot.
"""

# Keywords for triage classification
SEVERE_KEYWORDS = [
    "chest pain", "difficulty breathing", "shortness of breath", "unconscious",
    "loss of consciousness", "stroke", "slurred speech", "sudden weakness",
    "severe bleeding", "not breathing"
]

URGENT_KEYWORDS = [
    "high fever", "fever", "persistent vomiting", "severe pain", "deep cut",
    "possible fracture", "infected", "worsening", "dehydration"
]

RECORD_ESCALATORS = [
    "heart disease", "myocardial", "anticoagulant", "blood thinner",
    "immunocompromised", "chemotherapy", "pregnant", "organ transplant",
    "arrhythmia", "stroke history"
]

# Lists based on ccmedicalcenter.com article: Top reasons people visit the ER
MENS_TREAT_AND_RELEASE = [
    "open wounds to the head",
    "open wounds to the neck",
    "open wounds to the limbs",
    "head injury",
    "neck injury",
    "limb injury"
]

WOMENS_TREAT_AND_RELEASE = [
    "urinary tract infection",
    "UTI",
    "headache",
    "migraine",
    "pregnancy-related issues",
    "pregnancy complications"
]

ER = [
    "allergic reactions with trouble breathing",
    "severe swelling from allergic reaction",
    "broken bones",
    "dislocations",
    "loss of vision",
    "double vision",
    "choking",
    "electric shock",
    "severe head injury",
    "heart attack symptoms",
    "chest pain with shortness of breath",
    "high fever over 103F",
    "fever with rash",
    "loss of consciousness",
    "mental health crisis",
    "self-harm",
    "harm to others",
    "poisoning",
    "seizures",
    "severe abdominal pain"
]

URGENT_CARE = [
    "minor cuts",
    "minor wounds",
    "superficial injuries",
    "mild fever",
    "upper respiratory infection",
    "bronchiolitis",
    "mild asthma attack",
    "middle ear infection",
    "viral infection",
    "minor musculoskeletal pain",
    "minor back pain",
    "non-severe headache",
    "mild abdominal pain",
    "vomiting without severe dehydration"
]
