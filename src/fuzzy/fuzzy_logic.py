import numpy as np
from scipy.stats import norm

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
            degree_left, degree_medium = FuzzyLogic.classify_value(
                value, (left["b"], left["c"]), (medium["a"], medium["b"])
            )
            result[left["label"]] = round(degree_left, 2)
            result[medium["label"]] = round(degree_medium, 2)
        elif left["c"] <= value < right["a"]:
            result[medium["label"]] = 1.0
        elif right["a"] <= value <= medium["c"]:
            degree_medium, degree_right = FuzzyLogic.classify_value(
                value, (medium["b"], medium["c"]), (right["a"], right["b"])
            )
            result[medium["label"]] = round(degree_medium, 2)
            result[right["label"]] = round(degree_right, 2)
        elif value > medium["c"]:
            result[right["label"]] = 1.0

        return result
