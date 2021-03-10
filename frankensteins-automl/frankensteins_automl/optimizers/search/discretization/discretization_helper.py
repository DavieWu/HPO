import copy
import logging

logger = logging.getLogger(__name__)

minimum_split_ratio = 0.1


class Discretization:
    def __init__(self, parameter_domain):
        self.parameter_list = []
        self.selected_values = []
        self.parameter_descriptions = parameter_domain
        self.refinement_index = 0
        for component in self.parameter_descriptions.values():
            for parameter in component.values():
                self.parameter_list.append(parameter)
                if parameter["type"] == "int":
                    self.selected_values.append(
                        {
                            "lower_split_bound": parameter["min"],
                            "upper_split_bound": parameter["max"],
                        }
                    )
                elif parameter["type"] == "double":
                    minimum_split_size = (
                        parameter["max"] - parameter["min"]
                    ) * minimum_split_ratio
                    self.selected_values.append(
                        {
                            "lower_split_bound": parameter["min"],
                            "upper_split_bound": parameter["max"],
                            "minimum_split_size": minimum_split_size,
                        }
                    )
                else:
                    self.selected_values.append(None)

    @classmethod
    def from_previous_discretization_by_refinement(
        cls, discretization, refinement
    ):
        d = Discretization(discretization.parameter_descriptions)
        d.parameter_list = discretization.parameter_list
        d.selected_values = copy.deepcopy(discretization.selected_values)
        for i in range(len(d.selected_values)):
            if d.selected_values[i] is None or isinstance(
                d.selected_values[i], dict
            ):
                d.selected_values[i] = refinement
                d.refinement_index = i + 1
                break
        return d

    @classmethod
    def from_previous_discretization_by_split(cls, discretization, new_split):
        d = Discretization(discretization.parameter_descriptions)
        d.parameter_list = discretization.parameter_list
        d.selected_values = copy.deepcopy(discretization.selected_values)
        d.refinement_index = discretization.refinement_index
        d.selected_values[d.refinement_index] = new_split
        return d

    def is_atomic(self):
        return self.refinement_index >= len(self.parameter_list)

    def get_config(self):
        if self.is_atomic():
            i = 0
            config = {}
            for c_name, c in self.parameter_descriptions.items():
                c_config = {}
                for parameter in c.keys():
                    c_config[parameter] = self.selected_values[i]
                    i = i + 1
                config[c_name] = c_config
            return config
        else:
            return None


def refine_discretization(discretization):
    refinements = []
    if not discretization.is_atomic():
        param_description = discretization.parameter_list[
            discretization.refinement_index
        ]
        param_type = param_description["type"]
        if param_type == "bool":
            refinements.append(
                Discretization.from_previous_discretization_by_refinement(
                    discretization, True
                )
            )
            refinements.append(
                Discretization.from_previous_discretization_by_refinement(
                    discretization, False
                )
            )
        elif param_type == "cat":
            for cat in param_description["values"]:
                refinements.append(
                    Discretization.from_previous_discretization_by_refinement(
                        discretization, cat
                    )
                )
        elif param_type == "int":
            current_split = discretization.selected_values[
                discretization.refinement_index
            ]
            lower_split_bound = current_split["lower_split_bound"]
            upper_split_bound = current_split["upper_split_bound"]
            if lower_split_bound == upper_split_bound:
                refinements.append(
                    Discretization.from_previous_discretization_by_refinement(
                        discretization, lower_split_bound
                    )
                )
            else:
                split_point = (upper_split_bound - lower_split_bound) * 0.5
                split_point = int(lower_split_bound + split_point)
                refinements.append(
                    Discretization.from_previous_discretization_by_split(
                        discretization,
                        {
                            "lower_split_bound": lower_split_bound,
                            "upper_split_bound": split_point,
                        },
                    )
                )
                refinements.append(
                    Discretization.from_previous_discretization_by_split(
                        discretization,
                        {
                            "lower_split_bound": split_point + 1,
                            "upper_split_bound": upper_split_bound,
                        },
                    )
                )
        elif param_type == "double":
            current_split = discretization.selected_values[
                discretization.refinement_index
            ]
            lower_split_bound = current_split["lower_split_bound"]
            upper_split_bound = current_split["upper_split_bound"]
            minimum_split_size = current_split["minimum_split_size"]
            if (upper_split_bound - lower_split_bound) <= minimum_split_size:
                refinement_value = (
                    upper_split_bound - lower_split_bound
                ) * 0.5
                refinement_value = refinement_value + lower_split_bound
                refinements.append(
                    Discretization.from_previous_discretization_by_refinement(
                        discretization, refinement_value
                    )
                )
            else:
                split_point = (upper_split_bound - lower_split_bound) * 0.5
                split_point = lower_split_bound + split_point
                refinements.append(
                    Discretization.from_previous_discretization_by_split(
                        discretization,
                        {
                            "lower_split_bound": lower_split_bound,
                            "upper_split_bound": split_point,
                            "minimum_split_size": minimum_split_size,
                        },
                    )
                )
                refinements.append(
                    Discretization.from_previous_discretization_by_split(
                        discretization,
                        {
                            "lower_split_bound": split_point,
                            "upper_split_bound": upper_split_bound,
                            "minimum_split_size": minimum_split_size,
                        },
                    )
                )
    return refinements
