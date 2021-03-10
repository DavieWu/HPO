import copy
import logging
from sklearn.pipeline import make_pipeline, make_union
from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401
from frankensteins_automl.search_space.search_space_component_instance import (
    SearchSpaceComponentInstance,
)


logger = logging.getLogger(__name__)


def construct_pipeline(start_component_name, rest_problem, parameter):
    component_mapping = copy.deepcopy(rest_problem.get_component_mapping())
    required_interfaces = copy.deepcopy(rest_problem.get_required_interfaces())
    for component_id, component in component_mapping.items():
        if component.get_name() == start_component_name:
            return _assemble_component_instance(
                component_id, component_mapping, required_interfaces, parameter
            )
    logger.warning(f"No component found as start {start_component_name}")
    return None


def _assemble_component_instance(
    component_id, component_mapping, required_interfaces, parameter
):
    logger.debug(f"Assembling {component_id}")
    logger.debug(f"Params for assembling: {parameter}")
    logger.debug(f"Components for assembling: {component_mapping}")
    logger.debug(f"Required interfaces for assembling: {required_interfaces}")
    params = {}
    if component_id in parameter:
        params = parameter[component_id]
    component = component_mapping[component_id]
    instance = SearchSpaceComponentInstance(component)
    # Create required interfaces
    if component.has_required_interfaces():
        # Get all interfaces for the current component
        interfaces = list(
            filter(
                lambda i: i["component_id"] == component_id,
                required_interfaces,
            )
        )
        # Create an instance for this component
        interface_instances = {}
        for interface in interfaces:
            required_interfaces.remove(interface)
            created_element = _assemble_component_instance(
                interface["satisfied_with"],
                component_mapping,
                required_interfaces,
                parameter,
            )
            if created_element is not None:
                interface_instances[
                    interface["interface"]["id"]
                ] = created_element
            else:
                logger.warning(f"No instance of {interface} could be created")
        if not instance.set_required_interfaces(interface_instances):
            logger.error(
                f"Created interfaces were invalid for {component.get_name()}"
            )
            return None
    # Set parameter and create element from configured instance
    if not instance.set_parameter_config(params):
        logger.error(
            f"Given parameter were invalid for f{component.get_name()}"
        )
        return None
    else:
        return instance.construct_pipeline_element()


def build_topology(topology_array):
    return make_pipeline(*topology_array)


def preprocessor_union(preprocessor_1, preprocessor_2):
    return make_union(*preprocessor_1, *preprocessor_2)


def topology_union(*topologies):
    union_array = []
    for topology in topologies:
        if isinstance(topology, list):
            union_array.extend(topology)
        else:
            union_array.append(topology)
    return union_array
