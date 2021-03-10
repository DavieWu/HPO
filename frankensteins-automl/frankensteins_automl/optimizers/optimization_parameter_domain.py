import heapq
import logging
import numpy
import random
import uuid


logger = logging.getLogger(__name__)


class OptimizationParameterDomain(object):
    def __init__(self, component_mapping):
        self.component_mapping = component_mapping
        self.parameter_descriptions = {}
        self.has_parameter = False
        for component_id, component in self.component_mapping.items():
            description = component.get_parameter_description()
            self.parameter_descriptions[component_id] = description
            if len(description) > 0:
                self.has_parameter = True
        self.results = []
        self.id_to_scores_mapping = {}
        self.id_to_vector_mapping = {}
        self.vector_string_to_id_mapping = {}
        (
            self.min_vector,
            self.max_vector,
        ) = self._calculate_min_and_max_vectors()

    def _calculate_min_and_max_vectors(self):
        min_vector = []
        max_vector = []
        for component in list(self.component_mapping.values()):
            for parameter in list(
                component.get_parameter_description().values()
            ):
                parameter_type = parameter["type"]
                if parameter_type in ["int", "double"]:
                    min_vector.append(float(parameter["min"]))
                    max_vector.append(float(parameter["max"]))
                elif parameter_type == "bool":
                    min_vector.append(0.0)
                    max_vector.append(2.0)
                elif parameter_type == "cat":
                    min_vector.append(0.0)
                    max_vector.append(float(len(parameter["values"]) - 1))
        return numpy.array(min_vector), numpy.array(max_vector)

    def get_has_parameter(self):
        return self.has_parameter

    def get_parameter_descriptions(self):
        return self.parameter_descriptions

    def get_default_config(self):
        config = {}
        for component_id, component in self.component_mapping.items():
            config[component_id] = component.create_default_parameter_config()
        return self.config_to_vector(config)

    def draw_random_config(self):
        config = {}
        for component_id, component in self.component_mapping.items():
            config[component_id] = component.draw_random_parameter_config()
        return self.config_to_vector(config)

    def add_result(self, vector, score):
        vector_str = str(vector)
        if vector_str not in self.vector_string_to_id_mapping:
            vector_id = str(uuid.uuid1())
            self.vector_string_to_id_mapping[vector_str] = vector_id
            self.id_to_scores_mapping[vector_id] = score
            self.id_to_vector_mapping[vector_id] = vector
            heapq.heappush(self.results, (score, vector_id))

    def get_top_results(self, top_n):
        results = heapq.nlargest(top_n, self.results)
        result_vectors = []
        for result in results:
            result_vectors.append(
                (result[0], self.id_to_vector_mapping[result[1]])
            )
        return result_vectors

    def get_random_result(self):
        candidate_score, candidate_id = random.choice(self.results)
        return candidate_score, self.id_to_vector_mapping[candidate_id]

    def get_score_of_result(self, result):
        result_str = str(result)
        if result_str in self.vector_string_to_id_mapping:
            vector_id = self.vector_string_to_id_mapping[result_str]
            return self.id_to_scores_mapping[vector_id]
        return None

    def has_results(self):
        return len(self.results) > 0

    def get_min_vector(self):
        return self.min_vector

    def get_max_vector(self):
        return self.max_vector

    def config_to_vector(self, config):
        vector = []
        for component_id, component in self.component_mapping.items():
            for (
                parameter_id,
                parameter,
            ) in component.get_parameter_description().items():
                parameter_type = parameter["type"]
                value = config[component_id][parameter_id]
                if parameter_type == "int":
                    value = float(value)
                elif parameter_type == "bool":
                    if value:
                        value = 1.0
                    else:
                        value = 0.0
                elif parameter_type == "cat":
                    value = float(parameter["values"].index(value))
                vector.append(value)
        return numpy.array(vector)

    def config_from_vector(self, vector):
        config = {}
        index = 0
        for component_id, component in self.component_mapping.items():
            config[component_id] = {}
            for (
                parameter_id,
                parameter,
            ) in component.get_parameter_description().items():
                value = vector[index]
                index = index + 1
                parameter_type = parameter["type"]
                if parameter_type == "int":
                    value = int(value)
                elif parameter_type == "bool":
                    if value < 1.0:
                        value = False
                    else:
                        value = True
                elif parameter_type == "cat":
                    if value < 0:
                        value = 0
                    if value >= len(parameter["values"]):
                        value = len(parameter["values"] - 1)
                    value = parameter["values"][int(round(value))]
                config[component_id][parameter_id] = value
        return config
