import pandas as pd
import numpy as np
from scipy.stats import norm
from rdflib import Graph, Namespace, RDF, RDFS, OWL, Literal, XSD


# ------------------------------
# Fuzzy Logic Module
# ------------------------------
class FuzzySet:
    def __init__(self, label, a, b, c, shape):
        self.label = label
        self.a = a
        self.b = b
        self.c = c
        self.shape = shape


class FuzzyLogic:
    @staticmethod
    def classify_value(x, group1_range, group2_range):
        mu1 = (group1_range[0] + group1_range[1]) / 2
        sigma1 = (group1_range[1] - group1_range[0]) / 4
        mu2 = (group2_range[0] + group2_range[1]) / 2
        sigma2 = (group2_range[1] - group2_range[0]) / 4

        pdf1 = norm.pdf(x, mu1, sigma1)
        pdf2 = norm.pdf(x, mu2, sigma2)
        total = pdf1 + pdf2
        p1 = pdf1 / total if total > 0 else 0
        p2 = pdf2 / total if total > 0 else 0
        return p1, p2

    @staticmethod
    def get_membership_degrees(value, fuzzy_sets):
        result = {}
        left = next(fs for fs in fuzzy_sets if fs["shape"] == "left_shoulder")
        medium = next(fs for fs in fuzzy_sets if fs["shape"] == "triangle")
        right = next(fs for fs in fuzzy_sets if fs["shape"] == "right_shoulder")

        if value < medium["a"]:
            result[left["label"]] = 1.0
        elif medium["a"] <= value < left["c"]:
            degree_left, degree_medium = FuzzyLogic.classify_value(value, (left["b"], left["c"]), (medium["a"], medium["b"]))
            result[left["label"]] = round(degree_left, 2)
            result[medium["label"]] = round(degree_medium, 2)
        elif left["c"] <= value < right["a"]:
            result[medium["label"]] = 1.0
        elif right["a"] <= value <= medium["c"]:
            degree_medium, degree_right = FuzzyLogic.classify_value(value, (medium["b"], medium["c"]), (right["a"], right["b"]))
            result[medium["label"]] = round(degree_medium, 2)
            result[right["label"]] = round(degree_right, 2)
        elif value > medium["c"]:
            result[right["label"]] = 1.0

        return result

# ------------------------------
# Patient and Measurement Module
# ------------------------------
class Patient:
    def __init__(self, row):
        self.id = row["User_ID"]
        self.age = row["Age"]
        self.gender = row["Gender"]
        self.weight = row["Weight"]
        self.height = row["Height"]
        self.alcohol = row["Alcohol_Consumption"]
        self.heart = row["Heart_Rate"]
        self.devheart = row["Device_heart"]
        self.devblood = row["Device_blood"]
        self.devcalorie = row["calorie_device"]
        self.devtem = row["tem_dev"]
        self.blood = row["Blood_Oxygen_Level"]
        self.calorie = row["Calories_Intake"]
        self.temp = row["Skin_Temperature"]
        self.bodyfat = row["Body_Fat_Percentage"]
        self.timestamp = row["Timestamp1"]
        self.muscle = row["Muscle_Mass"]
        self.devmuscle = row["muscle_dv"]
        self.stress = row["Stress_Level"]
        self.waterIntake = row["Water_Intake"]
        self.disease = row["Medical_Conditions"]
        self.mood = row["Mood"]
        self.normalsleep = row["Sleep_Duration"]
        self.deepsleep = row["Deep_Sleep_Duration"]
        self.remsleep = row["REM_Sleep_Duration"]
        self.wakeup = row["Wakeups"]
        self.devsleep = row["devsleep"]
        self.score = row["Health_Score"]


# ------------------------------
# Ontology Handler
# ------------------------------
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
            "LowWaterIntake", "MediumWaterIntake", "HighWaterIntake",
            "LowNormalSleepDuration", "MediumNormalSleepDuration", "HighNormalSleepDuration",
            "LowDeepSleepDuration", "MediumDeepSleepDuration", "HighDeepSleepDuration",
            "LowREMSleepDuration", "MediumREMSleepDuration", "HighREMSleepDuration",
            "LowWakeupsDuration", "MediumWakeupsDuration", "HighWakeupsDuration",
            "LowScore", "MediumScore", "HighScore"
        ]
        for label in labels:
            self.g.add((self.EX[label], RDF.type, OWL.AnnotationProperty))

    def add_patient(self, patient: Patient, idx: int, fuzzy_sets):
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


# ------------------------------
# Main Execution
# ------------------------------
if __name__ == "__main__":
    data = pd.read_excel("personal_health_data.xlsx")
    print(data.head)

    fuzzy_sets = {
    "age": [
        {"label": "Young", "a": 10, "b": 25, "c": 35, "shape": "left_shoulder"},
        {"label": "Adult", "a": 30, "b": 47.5, "c": 65, "shape": "triangle"},
        {"label": "Old", "a": 60, "b": 75, "c": 85, "shape": "right_shoulder"},
        ],
    "heart": [
        {"label": "LowHeartRate", "a": 50, "b":60, "c": 75, "shape": "left_shoulder"},
        {"label": "MediumHeartRate", "a": 65, "b": 80, "c": 105, "shape": "triangle"},
        {"label": "HighHeartRate", "a": 95, "b": 130, "c": 180, "shape": "right_shoulder"},
        ],
    "bloodoxygen": [
        {"label": "LowBloodOxygen", "a": 90.0025179, "b":91.5, "c": 94, "shape": "left_shoulder"},
        {"label": "MediumBloodOxygen", "a": 93, "b": 96.5, "c": 99, "shape": "triangle"},
        {"label": "HighBloodOxygen", "a": 98, "b": 100, "c": 100.9990865, "shape": "right_shoulder"},
        ],
    "calorie": [
        {"label": "LowCalorieIntake", "a": 1200.01183, "b":1600.0, "c": 2000.0, "shape": "left_shoulder"},
        {"label": "MediumCalorieIntake", "a": 1800.0, "b": 2200.0, "c": 2600.0, "shape": "triangle"},
        {"label": "HighCalorieIntake", "a": 2400.0 , "b": 2700.0, "c": 2999.733804, "shape": "right_shoulder"},
        ],
    "temperature": [
        {"label": "LowTemperature", "a": 32.0002771, "b":33.75, "c": 35.0, "shape": "left_shoulder"},
        {"label": "MediumTemperature", "a": 34.5, "b": 36.0, "c": 37.5, "shape": "triangle"},
        {"label": "HighTemperature", "a": 37.0, "b": 38.0, "c": 38.99936953, "shape": "right_shoulder"},
        ],
    "bodyfat": [
        {"label": "LowBodyFat", "a": 10.00121675, "b":13.5, "c": 18.0 , "shape": "left_shoulder"},
        {"label": "MediumBodyFat", "a": 16.0, "b": 22.0, "c": 29.0 , "shape": "triangle"},
        {"label": "HighBodyFat", "a": 27.0, "b": 31.5, "c": 34.99782606, "shape": "right_shoulder"},
        ],
    "musclemass": [
        {"label": "LowMuscleMass", "a": 20.00033553, "b":30.0, "c": 45.0 , "shape": "left_shoulder"},
        {"label": "MediumMuscleMass", "a": 40.0, "b": 50.0, "c": 60.0  , "shape": "triangle"},
        {"label": "HighMuscleMass", "a": 55.0, "b": 67.5, "c": 79.99875569, "shape": "right_shoulder"},
    ],
    "waterIntake": [
        {"label": "LowWaterIntake", "a": 0.500078726, "b":1.0, "c": 1.75 , "shape": "left_shoulder"},
        {"label": "MediumWaterIntake", "a": 1.5, "b": 2.25, "c": 3.0  , "shape": "triangle"},
        {"label": "HighWaterIntake", "a": 2.5, "b": 3.0 , "c": 3.499655592, "shape": "right_shoulder"},
    ],
    "normalsleep": [
        {"label": "LowNormalSleepDuration", "a":4.000285234 , "b":5.5, "c":7.0 , "shape": "left_shoulder"},
        {"label": "MediumNormalSleepDuration", "a": 6.0, "b":7.5 , "c": 9.0, "shape": "triangle"},
        {"label": "HighNormalSleepDuration", "a": 8.0, "b": 9.25 , "c":9.999370248 , "shape": "right_shoulder"},
    ],
    "deepsleep": [
        {"label": "LowDeepSleepDuration", "a": 0.500597801, "b":2.0, "c":4.0  , "shape": "left_shoulder"},
        {"label": "MediumDeepSleepDuration", "a":3.0 , "b":5.5 , "c":  7.5 , "shape": "triangle"},
        {"label": "HighDeepSleepDuration", "a": 6.5, "b": 8.5  , "c":9.93064243 , "shape": "right_shoulder"},
    ],   

    "wakeupduration": [
        {"label": "LowWakeupsDuration", "a":0 , "b":1.0, "c": 2.0, "shape": "left_shoulder"},
        {"label": "MediumWakeupsDuration", "a": 1.5, "b": 2.5, "c": 3.5  , "shape": "triangle"},
        {"label": "HighWakeupsDuration", "a":3.0, "b": 3.5 , "c":4 , "shape": "right_shoulder"},
    ],
    "remsleep": [
        {"label": "LowWakeupsDuration", "a":0.000805135 , "b":2.0, "c":4.0 , "shape": "left_shoulder"},
        {"label": "MediumWakeupsDuration", "a": 3.0, "b": 5.0, "c": 7.0  , "shape": "triangle"},
        {"label": "HighWakeupsDuration", "a":6.0, "b": 7.75, "c":9.448512283 , "shape": "right_shoulder"},
    ],
    "wakeup": [
        {"label": "LowWakeupsDuration", "a":0.000805135 , "b":2.0, "c":4.0 , "shape": "left_shoulder"},
        {"label": "MediumWakeupsDuration", "a": 3.0, "b": 5.0, "c": 7.0  , "shape": "triangle"},
        {"label": "HighWakeupsDuration", "a":6.0, "b": 7.75, "c":9.448512283 , "shape": "right_shoulder"},
    ],
    "score": [
        {"label": "LowScore", "a":0 , "b":20, "c":40 , "shape": "left_shoulder"},
        {"label": "MediumScore", "a": 30, "b": 50, "c": 70  , "shape": "triangle"},
        {"label": "HighScore", "a":60, "b": 80, "c":100, "shape": "right_shoulder"},
    ]

                }

    ontology = HealthOntology("ontology/v1.ttl")
    for idx, row in data.iterrows():
        patient = Patient(row)
        ontology.add_patient(patient, idx, fuzzy_sets)

    ontology.save("fullKG.ttl")