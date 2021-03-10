import copy
import logging
import random
import uuid


logger = logging.getLogger(__name__)


class SearchSpaceComponent(object):
    def __init__(self, description):
        self.name = description["name"]
        self.id = str(uuid.uuid1())
        logger.debug(f"Create search space component {self.name}")
        self.provided_interfaces = []
        if "providedInterface" in description:
            self.provided_interfaces = description["providedInterface"]
        self.required_interfaces = {}
        if "requiredInterface" in description:
            self.required_interfaces = description["requiredInterface"]
        self.function_pointer = "function_pointer" in description and (
            description["function_pointer"]
        )
        self.params = {}
        if "parameter" in description:
            for p in description["parameter"]:
                logger.debug(f"Init param {p}")
                param_domain = copy.deepcopy(p)
                del param_domain["name"]
                self.params[p["name"]] = param_domain

    def get_name(self):
        return self.name

    def get_id(self):
        return self.id

    def get_provided_interfaces(self):
        return self.provided_interfaces

    def get_required_interfaces(self):
        return self.required_interfaces

    def has_required_interfaces(self):
        return len(self.required_interfaces) > 0

    def is_function_pointer(self):
        return self.function_pointer

    def has_parameter(self):
        return len(self.params) > 0

    def get_parameter_description(self):
        return self.params

    def create_default_parameter_config(self):
        config = {}
        for name, domain in self.params.items():
            if "default" in domain:
                config[name] = domain["default"]
            else:
                logger.warning(f"Parameter {name} has no default value")
                config[name] = self._draw_random_parameter_value(domain)
        return config

    def draw_random_parameter_config(self):
        config = {}
        for name, domain in self.params.items():
            config[name] = self._draw_random_parameter_value(domain)
        return config

    def _draw_random_parameter_value(self, domain):
        param_type = domain["type"]
        if param_type == "double":
            return random.uniform(domain["min"], domain["max"])
        elif param_type == "int":
            return random.randint(domain["min"], domain["max"])
        elif param_type == "cat":
            return random.choice(domain["values"])
        elif param_type == "bool":
            return bool(random.getrandbits(1))
        else:
            logger.warning(
                f"Unknown parameter type {param_type}, so returning None"
            )
            return None

    def validate_parameter_config(self, config):
        logger.debug(f"Validate param config of {self.name} with {config}")
        # Check if config has as many members as parameters needed
        if len(config) != len(self.params):
            logger.debug("Not valid because config has wrong number of params")
            return False
        # Check given values for each param
        for param, domain in self.params.items():
            # Check if the config has this param
            if param in config:
                logger.debug(f"Checking param {param}")
                param_type = domain["type"]
                # Type is a category
                if param_type == "cat":
                    # Check if the set value is one of the allowed categories
                    if config[param] not in domain["values"]:
                        logger.debug(
                            f"Not valid because {config[param]} is unknown"
                        )
                        return False
                # Type is a boolean
                elif param_type == "bool":
                    if not isinstance(config[param], bool):
                        return False
                # Type is a number
                elif param_type == "double" or param_type == "int":
                    value = config[param]
                    # Check if value is smaller than allowed minimum
                    if value < domain["min"]:
                        logger.debug("Not valid because value is too small")
                        return False
                    # Check if value is bigger than allowd maximum
                    if value > domain["max"]:
                        logger.debug("Not valid because value is too big")
                        return False
                    # If type is double, check if value is float
                    if param_type == "double":
                        if not isinstance(value, float):
                            logger.debug(
                                "Not valid because a double was expected"
                            )
                            return False
                    # If type is int, check if the value is int
                    elif param_type == "int":
                        if not isinstance(value, int):
                            logger.debug(
                                "Not valid because an int was expected"
                            )
                            return False
                # One param type is unknown and cannot be validated
                else:
                    logger.warning(f"Unknown param type {param_type}")
            # Config is missing a paramter and is not valid
            else:
                logger.debug(
                    f"Not valid because config does not include {param}"
                )
                return False
        # Validation of all config params was successfull
        return True

    def create_construction_args_from_config(
        self, parameter_config, required_interfaces
    ):
        positional_args = []
        keyword_args = {}
        if parameter_config is not None:
            for parameter in self.params:
                key = parameter
                if "construction_key" in self.params[parameter]:
                    key = self.params[parameter]["construction_key"]
                if isinstance(key, str):
                    keyword_args[key] = parameter_config[parameter]
                elif isinstance(key, int):
                    positional_args.insert(key, parameter_config[parameter])
        if required_interfaces is not None:
            for interface in self.required_interfaces:
                logger.debug(interface)
                key = interface["construction_key"]
                if isinstance(key, str):
                    keyword_args[key] = required_interfaces[interface["id"]]
                elif isinstance(key, int):
                    positional_args.insert(
                        key, required_interfaces[interface["id"]]
                    )
        return positional_args, keyword_args
