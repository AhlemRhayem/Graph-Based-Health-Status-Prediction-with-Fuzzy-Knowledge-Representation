from src.fuzzy.fuzzy_logic_config import fuzzy_sets
from fuzzy.fuzzy_logic import FuzzyLogic
from fuzzy.fuzzy_set import FuzzySet

import pandas as pd 
from models.patient import Patient

from ontology.health_ontology import HealthOntology

if __name__ == "__main__":
    data = pd.read_excel("personal_health_data.xlsx")

    from src.fuzzy.fuzzy_logic_config import fuzzy_sets   

    ontology = HealthOntology("ontology/v1.ttl")

    for idx, row in data.iterrows():
        patient = Patient(row)
        ontology.add_patient(patient, idx, fuzzy_sets)

    ontology.save("fullKG.ttl")
