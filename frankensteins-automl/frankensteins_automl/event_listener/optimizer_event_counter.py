from pubsub import pub
from frankensteins_automl.event_listener import event_topics

optimizer_call_counts = {}


def activate():
    pub.subscribe(_listener, event_topics.OPTIMIZATION_TOPIC)


def get_optimizer_call_counts():
    return optimizer_call_counts


def _listener(payload, topic=pub.AUTO_TOPIC):
    optimizer = payload["optimizer_class"]
    if optimizer in optimizer_call_counts:
        current_count = optimizer_call_counts[optimizer]
        optimizer_call_counts[optimizer] = current_count + 1
    else:
        optimizer_call_counts[optimizer] = 1
