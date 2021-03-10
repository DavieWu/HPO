import json
import logging
from pubsub import pub
from frankensteins_automl.event_listener import event_topics

logger = None


def activate():
    global logger
    logger = logging.getLogger("event-logger")
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler("logs/event-log.log", "w")
    formatter = logging.Formatter("%(asctime)s %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info("Start of event log:\n")
    for topic in event_topics.ALL_TOPICS:
        logger.info(f"Log for topic: {topic}")
        pub.subscribe(_listener, topic)


def _listener(payload, topic=pub.AUTO_TOPIC):
    logger.info(f"{topic.getName()}:\n{json.dumps(payload, indent=4)}\n\n")
