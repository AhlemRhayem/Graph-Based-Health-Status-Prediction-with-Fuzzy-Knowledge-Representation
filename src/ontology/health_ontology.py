import pandas as pd
from rdflib import Graph, Namespace, RDF, RDFS, OWL, Literal, XSD
from src.fuzzy.fuzzy_logic import FuzzyLogic

class HealthOntology:

    def __init__(self, ontology_file: str):
        self.g = Graph()
        self.g.parse(ontology_file)
        self.EX = Namespace("https://w3id.org/HealthIoT#")
        self.SAREF = Namespace("https://saref.etsi.org/core#")
        self.device_instances = {}
        self._bind_namespaces()
        self._define_labels()

    def _bind_namespaces(self):
        self.g.bind("ex", self.EX)
        self.g.bind("saref", self.SAREF)
        self.g.bind("rdf", RDF)
        self.g.bind("rdfs", RDFS)
        self.g.bind("owl", OWL)
        self.g.bind("xsd", XSD)

    def _define_labels(self):
        labels = [
            "Young", "Adult", "Old",
            "LowHeartRate", "MediumHeartRate", "HighHeartRate",
            "LowBloodOxygen", "MediumBloodOxygen", "HighBloodOxygen",
            "LowCalorieIntake", "MediumCalorieIntake", "HighCalorieIntake",
            "LowTemperature", "MediumTemperature", "HighTemperature",
            "LowBodyFat", "MediumBodyFat", "HighBodyFat",
            "LowMuscleMass", "MediumMuscleMass", "HighMuscleMass",
            "LowWaterIntake", "MediumWaterIntake", "HighWaterIntake"
        ]
        for label in labels:
            self.g.add((self.EX[label], RDF.type, OWL.AnnotationProperty))

    def add_patient(self, patient, idx, fuzzy_sets):
        person_uri = self.EX[patient.id]
        self.g.add((person_uri, RDF.type, self.EX.Patient))
        self.g.add((person_uri, self.EX.age, Literal(patient.age, datatype=XSD.integer)))
        self.g.add((person_uri, self.EX.patientId, Literal(patient.id, datatype=XSD.string)))
        self.g.add((person_uri, self.EX.weight, Literal(patient.weight, datatype=XSD.float)))
        self.g.add((person_uri, self.EX.height, Literal(patient.height, datatype=XSD.float)))
        self.g.add((person_uri, self.EX.stressLevel, self.EX[patient.stress]))
        self.g.add((person_uri, self.EX.mood, self.EX[patient.mood]))
        if pd.notna(patient.disease):
            self.g.add((person_uri, self.EX.medicalCondition, self.EX[patient.disease]))

        # Handle alcohol consumption
        if patient.alcohol == "Moderate":
            self.g.add((person_uri, self.EX.alcoholConsumption, self.EX.ModerateAlcoholConsumption))
        elif patient.alcohol == "Heavy":
            self.g.add((person_uri, self.EX.alcoholConsumption, self.EX.HeavyAlcoholConsumption))

        # Add heart rate and temperature measurements
        self.add_measurement(person_uri, patient.heart, patient.devheart, "HeartRate", idx, patient.timestamp)
        self.add_measurement(person_uri, patient.temp, patient.devtem, "Temperature", idx, patient.timestamp)
        self.add_measurement(person_uri, patient.blood, patient.devblood, "BloodOxygen", idx, patient.timestamp)
        self.add_measurement(person_uri, patient.calorie, patient.devcalorie, "CalorieIntake", idx, patient.timestamp)
        self.add_measurement(person_uri, patient.bodyfat, patient.devmuscle, "BodyFat", idx, patient.timestamp)
        self.add_measurement(person_uri, patient.muscle, patient.devmuscle, "MuscleMass", idx, patient.timestamp)
        self.add_measurement(person_uri, patient.waterIntake, patient.devsleep, "WaterIntake", idx, patient.timestamp)
        # Add fuzzy scores
        self._add_fuzzy_scores(person_uri, patient, idx, fuzzy_sets)

    def add_measurement(self, person_uri, value, device_id, property_name, idx, timestamp):
        """
                Generic function to add any measurement with a device.
            """
        if pd.notna(value):
            measurement_uri = self.EX[f"{property_name}Reading{idx + 1}"]
            self.g.add((measurement_uri, RDF.type, self.EX.Measurement))
            self.g.add((measurement_uri, self.EX.value, Literal(value)))
            self.g.add((measurement_uri, self.EX.timestamp, Literal(timestamp, datatype=XSD.dateTimeStamp)))
            self.g.add((measurement_uri, self.SAREF.relatesToProperty, self.EX[property_name]))
            self.g.add((self.EX[property_name], self.SAREF.relatesToMeasurement, measurement_uri))

            if device_id not in self.device_instances:
                device_uri = self.EX[f"Device{device_id}"]
                self.g.add((device_uri, RDF.type, self.EX.MedicalObject))
                self.g.add((device_uri, self.EX.deviceId, Literal(device_id)))
                self.device_instances[device_id] = device_uri
            else:
                device_uri = self.device_instances[device_id]

            self.g.add((device_uri, self.SAREF.makesMeasurement, measurement_uri))
            self.g.add((person_uri, self.EX.hasMeasurement, measurement_uri))
            self.g.add((person_uri, self.EX.wears, device_uri))

            return measurement_uri
        return None

    
    def _add_fuzzy_scores(self, person_uri, patient, idx, fuzzy_sets):
        scores_age = FuzzyLogic.get_membership_degrees(patient.age, fuzzy_sets["age"])
        scores_heart = FuzzyLogic.get_membership_degrees(patient.heart, fuzzy_sets["heart"])
        scores_temperature = FuzzyLogic.get_membership_degrees(patient.temp, fuzzy_sets["temperature"])
        scores_blood = FuzzyLogic.get_membership_degrees(patient.blood, fuzzy_sets["bloodoxygen"])
        scores_calorie = FuzzyLogic.get_membership_degrees(patient.calorie, fuzzy_sets["calorie"])
        scores_bodyfat = FuzzyLogic.get_membership_degrees(patient.bodyfat, fuzzy_sets["bodyfat"])
        scores_muscle = FuzzyLogic.get_membership_degrees(patient.muscle, fuzzy_sets["musclemass"])
        scores_water = FuzzyLogic.get_membership_degrees(patient.waterIntake, fuzzy_sets["waterIntake"])

        scores_normalsleep = FuzzyLogic.get_membership_degrees(patient.normalsleep, fuzzy_sets["normalsleep"])
        scores_deepsleep = FuzzyLogic.get_membership_degrees(patient.deepsleep, fuzzy_sets["deepsleep"])
        scores_remsleep = FuzzyLogic.get_membership_degrees(patient.remsleep, fuzzy_sets["remsleep"])
        scores_wakeupduration = FuzzyLogic.get_membership_degrees(patient.wakeup, fuzzy_sets["wakeup"])
        scores_healthscore = FuzzyLogic.get_membership_degrees(patient.score, fuzzy_sets["score"])

        for label, score in scores_age.items():
            self.g.add((person_uri, self.EX[label], Literal(score, datatype=XSD.float)))
        for label, score in scores_heart.items():
            self.g.add((self.EX[f"HeartRateReading{idx + 1}"], self.EX[label], Literal(score, datatype=XSD.float)))
        for label, score in scores_temperature.items():
            self.g.add((self.EX[f"TemperatureReading{idx + 1}"], self.EX[label], Literal(score, datatype=XSD.float)))
        for label, score in scores_blood.items():
            self.g.add((self.EX[f"BloodOxygenReading{idx + 1}"], self.EX[label], Literal(score, datatype=XSD.float)))
        for label, score in scores_calorie.items():
            self.g.add((self.EX[f"CalorieIntake{idx + 1}"], self.EX[label], Literal(score, datatype=XSD.float)))
        for label, score in scores_bodyfat.items():
            self.g.add((self.EX[f"BodyFatReading{idx + 1}"], self.EX[label], Literal(score, datatype=XSD.float)))
        for label, score in scores_muscle.items():
            self.g.add((self.EX[f"MuscleMassReading{idx + 1}"], self.EX[label], Literal(score, datatype=XSD.float)))
        for label, score in scores_water.items():
            self.g.add((self.EX[f"WaterIntakeReading{idx + 1}"], self.EX[label], Literal(score, datatype=XSD.float)))
        for label, score in scores_normalsleep.items():
            self.g.add((self.EX[f"NormalSleepReading{idx + 1}"], self.EX[label], Literal(score, datatype=XSD.float)))
        for label, score in scores_deepsleep.items():
            self.g.add((self.EX[f"DeepSleepReading{idx + 1}"], self.EX[label], Literal(score, datatype=XSD.float)))
        for label, score in scores_remsleep.items():
            self.g.add((self.EX[f"REMSleepReading{idx + 1}"], self.EX[label], Literal(score, datatype=XSD.float)))
        for label, score in scores_wakeupduration.items():
            self.g.add((self.EX[f"WakeupsReading{idx + 1}"], self.EX[label], Literal(score, datatype=XSD.float)))
        for label, score in scores_healthscore.items():
            self.g.add((person_uri, self.EX[label], Literal(score, datatype=XSD.float)))

    def save(self, file_path="fullKG.ttl"):
        self.g.serialize(destination=file_path, format="turtle")
