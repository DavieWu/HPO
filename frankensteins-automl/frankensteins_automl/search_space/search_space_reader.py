import json
import logging
import os
import uuid
from .search_space_component import SearchSpaceComponent
from .search_space import SearchSpace


logger = logging.getLogger(__name__)


def create_search_space(*paths):
    logger.debug(f"Creating search space from the files {paths}")
    components = []
    for path in paths:
        if isinstance(path, str):
            name, extension = os.path.splitext(path)
            if extension == ".json":
                logger.debug(f"Adding components from {path}")
                with open(path) as search_space_json:
                    search_space = json.load(search_space_json)
                    search_space_components = search_space["components"]
                    for component in search_space_components:
                        if (
                            "requiredInterface" in component
                            and component["requiredInterface"] is not None
                        ):
                            for interface in component["requiredInterface"]:
                                interface["id"] = str(uuid.uuid1())
                        new_component = SearchSpaceComponent(component)
                        logger.debug(
                            f"Add component {new_component.get_name()}"
                        )
                        components.append(new_component)
            else:
                logger.warning(
                    f"Given path {path} is not a JSON file. It will be ignored"
                )
        else:
            logger.warning(
                f"Given path {path} is not a string. It will be ignored"
            )
    if len(components) > 0:
        logger.debug(
            f"Creating search space with {len(components)} components"
        )
        return SearchSpace(components)
    else:
        return None
