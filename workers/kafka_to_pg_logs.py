import confluent_kafka
import prometheus_client
import psycopg2
import datetime
import json
import utils
from psycopg2.extras import DictCursor

utils.check_connection_status("postgres", 5432)
p = psycopg2.connect(host="postgres", user="root", port=5432, database="W9sV6cL2dX", password="E5rG7tY3fH")
k = confluent_kafka.Consumer({"bootstrap.servers": "kafka:29092", "group.id": "logs-group-1", "auto.offset.reset": "earliest"})
k.subscribe(["logs"])
p.autocommit = True


PG_INSERTS = prometheus_client.Counter("pg_inserts", "Postgres Inserts")
PG_INSERTS_TIME_SPENT = prometheus_client.Counter("pg_inserts_time_spent", "Postgres Inserts Time Spent")
PG_ERRORS = prometheus_client.Counter("pg_errors", "Postgres Errors")


def insert_to_postgres(store):
  print('calling insert postgres')
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

    insert_data.append((
      cur_date.date(),
      cur_date.replace(minute=0, second=0, microsecond=0),
      evt_log.get("ts"),
      evt_log.get("user_id"),
      evt_log.get("session"),
      evt_log.get("type"),
      evt_log.get("item_id")
    ))

  try:
    cursor = p.cursor()
    cursor.executemany(insert_query, insert_data)
    PG_INSERTS.inc(len(insert_data))
  except Exception as e:
    PG_ERRORS.inc()
    print("Worker Log error", e)

def compute_time_spent(store):
  insert_time_spent_query = """
  INSERT INTO time_spent_on_item (
    user_id,
    item_id,
    time_spent,
    last_item,
    evnt_stamp
  ) VALUES (%s, %s, %s, %s, %s)
  """

  try:
      time_spent_data = [] 
      for evt_log in store:
          session_id = evt_log['session']
          get_query = f"SELECT * FROM fct_hourly_metric WHERE session_id = '{session_id}'"
          cursor = p.cursor(cursor_factory=DictCursor)
          cursor.execute(get_query)
          evt_logs = cursor.fetchall()
          print(f'*** evt_logs for session_id {session_id}', evt_logs)

          length = len(evt_logs)
          if length == 1:
            print("*** Length is 1, skipping")
            continue
          else:
              penultimate_event = evt_logs[-2]
              last_event = evt_logs[-1]

              penultimate_event_timestamp = penultimate_event["evnt_stamp"]
              print(f'*** penultimate_event_timestamp', penultimate_event_timestamp)

              time_diff = last_event["evnt_stamp"] - penultimate_event["evnt_stamp"]
              item_id = penultimate_event["item_id"] # not last_event because last_event can be null with evt_type stop
              is_last = True if last_event["evt_type"] == "stop" else False

              time_spent_data.append((
                  last_event["user_id"],
                  item_id,
                  time_diff,
                  is_last,
                  penultimate_event_timestamp
              ))
          
      print(f'*** *** time_spent_data', time_spent_data)
      cursor_insert = p.cursor()
      cursor_insert.executemany(insert_time_spent_query, time_spent_data)
      PG_INSERTS.inc(len(time_spent_data))

  except Exception as e:
      PG_ERRORS.inc()
      print("Worker Time Spent error", e)

def main():
  store = []
  while True:
    msg = k.poll(1.0)
    if msg is None: continue
    if msg.error(): continue
    raw_res = msg.value().decode("utf-8")
    cur_res = json.loads(raw_res)
    store.append(cur_res)
    # if there are more than 5 messages in kafka consumer queue
    if len(store) > 5: 
      insert_to_postgres(store)
      # compute_time_spent(store)
      store = []


if __name__ == "__main__":
  print("Starting Kafka to Postgres logs worker")
  prometheus_client.start_http_server(9965)
  main()