import confluent_kafka
import prometheus_client
import psycopg2
import datetime
import json
import utils
from psycopg2.extras import DictCursor

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

utils.check_connection_status("postgres", 5432)
p = psycopg2.connect(
    host="postgres",
    user="root",
    port=5432,
    database="W9sV6cL2dX",
    password="E5rG7tY3fH",
)
k = confluent_kafka.Consumer(
    {
        "bootstrap.servers": "kafka:29092",
        "group.id": "logs-group-1",
        "auto.offset.reset": "earliest",
    }
)
k.subscribe(["logs"])
p.autocommit = True


PG_INSERTS = prometheus_client.Counter("pg_inserts", "Postgres Inserts")
PG_INSERTS_TIME_SPENT = prometheus_client.Counter(
    "pg_inserts_time_spent", "Postgres Inserts Time Spent"
)
PG_ERRORS = prometheus_client.Counter("pg_errors", "Postgres Errors")
TINY_STOP_EVENTS = prometheus_client.Counter("tiny_stop_events", "Stop Events")


def insert_to_postgres(store):
    insert_query = """
    INSERT INTO fct_hourly_metric (
      date_stamp,
      time_stamp,
      evnt_stamp,
      user_id,
      session_id,
      evt_type,
      item_id
    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
  """

    insert_data = []

    for evt_log in store:
        cur_date = datetime.datetime.fromtimestamp(evt_log["ts"])

        insert_data.append(
            (
                cur_date.date(),
                cur_date.replace(minute=0, second=0, microsecond=0),
                evt_log.get("ts"),
                evt_log.get("user_id"),
                evt_log.get("session"),
                evt_log.get("type"),
                evt_log.get("item_id"),
            )
        )

    try:
        cursor = p.cursor()
        cursor.executemany(insert_query, insert_data)
        PG_INSERTS.inc(len(insert_data))
    except Exception as e:
        PG_ERRORS.inc()
        print("Worker Log error", e)

def get_session(session_id):
    get_query = f"SELECT * FROM fct_hourly_metric WHERE session_id = '{session_id}' ORDER BY evnt_stamp ASC"
    cursor = p.cursor(cursor_factory=DictCursor)
    cursor.execute(get_query)
    evt_logs = cursor.fetchall()
    return evt_logs

def calculate_time_spent(session_logs):
    time_spent_per_item = []
    for i in range(len(session_logs) - 1):
        current_event = session_logs[i]
        next_event = session_logs[i + 1]

        if current_event["evt_type"] == "view":
            time_spent = next_event["evnt_stamp"] - current_event["evnt_stamp"]
            is_stop = next_event["evt_type"] == "stop"
            time_spent_per_item.append(
                (
                    current_event["user_id"],
                    current_event["session_id"],
                    current_event["item_id"],
                    time_spent,
                    is_stop,
                )
            )
    return time_spent_per_item


def insert_time_spent(time_spent_per_item):
    insert_query = """
        INSERT INTO time_spent (
        user_id,
        session_id,
        item_id,
        time_spent,
        last_item
        ) VALUES (%s, %s, %s, %s, %s)
    """
    cursor = p.cursor()
    cursor.executemany(insert_query, time_spent_per_item)
    p.commit()


def check_session_data(store):
    try:
        stop_events = [item for item in store if item.get("type") == "stop"]
        logger.info(f"There are {len(stop_events)} stop events.")
        TINY_STOP_EVENTS.inc(len(stop_events))

        for event in stop_events:
            session_logs = get_session(event["session"])
            logger.info(
                f"There are {len(session_logs)} events in session {event['session']}"
            )

            time_spent_per_item = calculate_time_spent(session_logs)
            logger.info(
                f"There are {len(time_spent_per_item)} view events in {event['session']}."
            )
            insert_time_spent(time_spent_per_item)
    except Exception as e:
        logger.error(f"Worker Log error: {e}")


def main():
    store = []
    while True:
        msg = k.poll(1.0)
        if msg is None:
            continue
        if msg.error():
            continue
        raw_res = msg.value().decode("utf-8")
        cur_res = json.loads(raw_res)
        store.append(cur_res)
        # if there are more than 5 messages in kafka consumer queue
        if len(store) > 5:
            insert_to_postgres(store)
            check_session_data(store)
            store = []


if __name__ == "__main__":
    print("Starting Kafka to Postgres logs worker")
    prometheus_client.start_http_server(9965)
    main()
