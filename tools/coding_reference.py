"""
Mock ICD-10 / CPT reference tables.

In production these would be sourced from a licensed coding database
(e.g. 3M, AAPC, or CMS code sets). For this demonstration platform a
curated subset is embedded directly so the Medical Coding Agent and
lookup tools function fully offline.
"""

from __future__ import annotations

ICD10_REFERENCE: dict[str, dict[str, list[str]]] = {
    "chest pain": {"code": "R07.9", "description": "Chest pain, unspecified", "keywords": ["chest pain", "chest discomfort"]},
    "type 2 diabetes": {"code": "E11.9", "description": "Type 2 diabetes mellitus without complications", "keywords": ["type 2 diabetes", "t2dm", "diabetes mellitus type 2"]},
    "essential hypertension": {"code": "I10", "description": "Essential (primary) hypertension", "keywords": ["hypertension", "high blood pressure", "htn"]},
    "acute bronchitis": {"code": "J20.9", "description": "Acute bronchitis, unspecified", "keywords": ["bronchitis"]},
    "lower back pain": {"code": "M54.50", "description": "Low back pain, unspecified", "keywords": ["low back pain", "lower back pain", "lumbago"]},
    "acute appendicitis": {"code": "K35.80", "description": "Unspecified acute appendicitis", "keywords": ["appendicitis"]},
    "pneumonia": {"code": "J18.9", "description": "Pneumonia, unspecified organism", "keywords": ["pneumonia"]},
    "atrial fibrillation": {"code": "I48.91", "description": "Unspecified atrial fibrillation", "keywords": ["atrial fibrillation", "afib", "a-fib"]},
    "acute kidney injury": {"code": "N17.9", "description": "Acute kidney failure, unspecified", "keywords": ["acute kidney injury", "aki", "acute renal failure"]},
    "migraine": {"code": "G43.909", "description": "Migraine, unspecified, not intractable, without status migrainosus", "keywords": ["migraine"]},
    "gastroenteritis": {"code": "K52.9", "description": "Noninfective gastroenteritis and colitis, unspecified", "keywords": ["gastroenteritis"]},
    "fracture wrist": {"code": "S62.90XA", "description": "Unspecified fracture of wrist and hand, initial encounter", "keywords": ["wrist fracture", "fractured wrist"]},
    "asthma": {"code": "J45.909", "description": "Unspecified asthma, uncomplicated", "keywords": ["asthma"]},
    "urinary tract infection": {"code": "N39.0", "description": "Urinary tract infection, site not specified", "keywords": ["uti", "urinary tract infection"]},
    "concussion": {"code": "S06.0X0A", "description": "Concussion without loss of consciousness, initial encounter", "keywords": ["concussion"]},
}

CPT_REFERENCE: dict[str, dict[str, str]] = {
    "office visit level 3": {"code": "99213", "description": "Office/outpatient visit, established patient, low complexity"},
    "office visit level 4": {"code": "99214", "description": "Office/outpatient visit, established patient, moderate complexity"},
    "office visit new level 3": {"code": "99203", "description": "Office/outpatient visit, new patient, low complexity"},
    "emergency dept level 4": {"code": "99284", "description": "Emergency department visit, high severity"},
    "emergency dept level 5": {"code": "99285", "description": "Emergency department visit, high severity, threat to life"},
    "chest x-ray": {"code": "71046", "description": "Radiologic examination, chest, 2 views"},
    "ct chest": {"code": "71260", "description": "CT thorax with contrast"},
    "ct abdomen pelvis": {"code": "74177", "description": "CT abdomen and pelvis with contrast"},
    "mri brain": {"code": "70551", "description": "MRI brain without contrast"},
    "ekg": {"code": "93000", "description": "Electrocardiogram, routine, with interpretation and report"},
    "basic metabolic panel": {"code": "80048", "description": "Basic metabolic panel"},
    "complete blood count": {"code": "85025", "description": "Complete blood count with differential"},
    "appendectomy": {"code": "44970", "description": "Laparoscopic appendectomy"},
    "wrist x-ray": {"code": "73100", "description": "Radiologic examination, wrist, 2 views"},
    "closed treatment wrist fracture": {"code": "25600", "description": "Closed treatment of distal radial fracture"},
    "nebulizer treatment": {"code": "94640", "description": "Pressurized or nonpressurized inhalation treatment"},
    "urinalysis": {"code": "81003", "description": "Urinalysis, automated, without microscopy"},
}


def lookup_icd10(term: str) -> dict | None:
    term_lower = term.lower().strip()
    for entry in ICD10_REFERENCE.values():
        if term_lower == entry["code"].lower():
            return entry
    for entry in ICD10_REFERENCE.values():
        if any(kw in term_lower or term_lower in kw for kw in entry["keywords"]):
            return entry
    return None


def lookup_cpt(term: str) -> dict | None:
    term_lower = term.lower().strip()
    for entry in CPT_REFERENCE.values():
        if term_lower == entry["code"].lower():
            return entry
    for key, entry in CPT_REFERENCE.items():
        if key in term_lower or term_lower in key:
            return entry
    return None
