import logging

logger = logging.getLogger(__name__)


class SearchSpace(object):
    def __init__(self, components):
        self.components = components
        self.components_by_name = {}
        self.components_by_id = {}
        self.components_providing_interface = {}
        for component in components:
            component_id = component.get_id()
            component_name = component.get_name()
            provided_interfaces = component.get_provided_interfaces()
            if component_name not in self.components_by_name:
                self.components_by_name[component_name] = []
            self.components_by_name[component_name].append(component)
            self.components_by_id[component_id] = component
            for interface in provided_interfaces:
                if interface not in self.components_providing_interface:
                    self.components_providing_interface[interface] = []
                if (
                    component_id
                    not in self.components_providing_interface[interface]
                ):
                    self.components_providing_interface[interface].append(
                        component_id
                    )

    def get_components_by_name(self, component_name):
        if component_name in self.components_by_name:
            return self.components_by_name[component_name]
        else:
            return None

    def get_components_providing_interface(self, interface_name):
        if interface_name in self.components_providing_interface:
            component_ids = self.components_providing_interface[interface_name]
            components = []
            for component_id in component_ids:
                components.append(self.components_by_id[component_id])
            return components
        logger.warning(
            f"No provided interfaces for unknown component {interface_name}"
        )
        return None
