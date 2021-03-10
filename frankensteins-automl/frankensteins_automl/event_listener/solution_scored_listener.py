import logging
import pickle
from pubsub import pub
from frankensteins_automl.event_listener import event_topics

logger = logging.getLogger(__name__)

best_pipeline_data = None
best_score = 0.0


def activate():
    pub.subscribe(_listener, event_topics.FRANKENSTEINS_AUTOML_TOPIC)


def get_best_solution():
    pipeline = None
    if best_pipeline_data is not None:
        pipeline = pickle.loads(bytes.fromhex(best_pipeline_data))
    return pipeline, best_score


def _listener(payload, topic=pub.AUTO_TOPIC):
    global best_score
    global best_pipeline_data
    if "event_type" in payload and payload["event_type"] == "SOLUTION_SCORED":
        score = payload["score"]
        pipeline_data = payload["pipeline_data"]
        if (score > best_score) and (pipeline_data is not None):
            best_score = score
            best_pipeline_data = pipeline_data
