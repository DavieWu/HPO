import requests
import uuid
from pubsub import pub
from frankensteins_automl.event_listener import event_topics

relevant_topics = [event_topics.MCTS_TOPIC, event_topics.SEARCH_GRAPH_TOPIC]
url = None
is_sending = False


def activate(rest_url):
    global url
    global is_sending
    url = rest_url
    is_sending = True
    for topic in relevant_topics:
        pub.subscribe(_listener, topic)


def _listener(payload, topic=pub.AUTO_TOPIC):
    global is_sending
    payload["message_id"] = str(uuid.uuid1())
    if is_sending:
        try:
            requests.post(url, json=payload)
        except Exception:
            is_sending = False
