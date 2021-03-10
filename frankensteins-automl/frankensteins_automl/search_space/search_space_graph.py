import copy
import logging
import uuid
from frankensteins_automl.event_listener import event_topics
from frankensteins_automl.search_space.graphs import GraphGenerator, GraphNode


logger = logging.getLogger(__name__)

topic = event_topics.SEARCH_GRAPH_TOPIC


class SearchSpaceRestProblem(object):
    def __init__(self, required_interfaces, component_mapping):
        self.required_interfaces = required_interfaces
        self.component_mapping = component_mapping

    @classmethod
    def from_previous_rest_problem(
        cls, rest_problem, satisfied_interface_component
    ):
        ri = copy.deepcopy(rest_problem.get_required_interfaces())
        component_mapping = copy.deepcopy(rest_problem.get_component_mapping())
        new_component_id = str(uuid.uuid1())
        component_mapping[new_component_id] = satisfied_interface_component
        for i, interface in enumerate(ri):
            if not interface["satisfied"]:
                interface["satisfied"] = True
                interface["satisfied_with"] = new_component_id
                break
        if satisfied_interface_component.get_required_interfaces() is not None:
            for (
                interface
            ) in satisfied_interface_component.get_required_interfaces():
                ri.append(
                    {
                        "interface": interface,
                        "satisfied": False,
                        "component_id": new_component_id,
                    }
                )
        return SearchSpaceRestProblem(ri, component_mapping)

    def is_satisfied(self):
        for interface in self.required_interfaces:
            if not interface["satisfied"]:
                return False
        logger.debug("All interfaces satisfied in rest problem")
        return True

    def get_required_interfaces(self):
        return self.required_interfaces

    def get_first_unsatisfied_required_interface(self):
        for interface in self.required_interfaces:
            if not interface["satisfied"]:
                return interface["interface"]
        return None

    def get_component_mapping(self):
        return self.component_mapping


class SearchSpaceGraphNode(GraphNode):
    def __init__(self, predecessor, rest_problem, specified_interface):
        super().__init__(predecessor)
        self.rest_problem = rest_problem
        self.specified_interface = specified_interface

    def get_rest_problem(self):
        return self.rest_problem

    def get_specified_interface(self):
        return self.specified_interface

    def is_leaf_node(self):
        return self.rest_problem.is_satisfied()

    def get_event_payload(self):
        predecessor_id = None
        if self.get_predecessor() is not None:
            predecessor_id = self.predecessor.get_node_id()
        return {
            "id": self.node_id,
            "predecessor": predecessor_id,
            "specified_interface": self.specified_interface,
        }


class SearchSpaceGraphGenerator(GraphGenerator):
    def __init__(self, search_space, initial_component_name):
        self.search_space = search_space
        self.initial_component_name = initial_component_name
        root_component = self.search_space.get_components_by_name(
            self.initial_component_name
        )[0]
        root_component_id = str(uuid.uuid1())
        unsatisfied_interfaces = []
        component_mapping = {}
        component_mapping[root_component_id] = root_component
        if root_component.get_required_interfaces() is not None:
            for ri in root_component.get_required_interfaces():
                unsatisfied_interfaces.append(
                    {
                        "interface": ri,
                        "satisfied": False,
                        "component_id": root_component_id,
                    }
                )
        rest_problem = SearchSpaceRestProblem(
            unsatisfied_interfaces, component_mapping
        )
        self.root_node = SearchSpaceGraphNode(None, rest_problem, None)

    def get_root_node(self):
        return self.root_node

    def get_node_successors(self, node):
        rest_problem = node.get_rest_problem()
        interface = rest_problem.get_first_unsatisfied_required_interface()[
            "name"
        ]
        components = self.search_space.get_components_providing_interface(
            interface
        )
        successors = []
        for component in components:
            logger.debug(f"Generating a successor with {component.get_name()}")
            successor_rp = SearchSpaceRestProblem.from_previous_rest_problem(
                rest_problem, component
            )
            specified_interface_name = component.get_name().split(".")
            specified_interface_name = specified_interface_name.pop()
            specified_interface_name = f"{specified_interface_name}"
            successor = SearchSpaceGraphNode(
                node, successor_rp, specified_interface_name
            )
            successors.append(successor)
        node.set_successors(successors)
        logger.debug(f"{len(successors)} successors for interface {interface}")
        return successors
