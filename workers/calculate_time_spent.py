#%%
import psycopg2
from psycopg2.extras import DictCursor
from tqdm import tqdm
from itertools import groupby
from operator import itemgetter
from uuid import UUID

#%%
def create_pg_connection():
    connection = psycopg2.connect(host="localhost", user="root", port=5432, database="W9sV6cL2dX", password="E5rG7tY3fH")
    connection.autocommit = True
    return connection
    
def get_stop_events(connection):
    with connection.cursor(cursor_factory=DictCursor) as cursor:
        cursor.execute("SELECT * FROM fct_hourly_metric WHERE evt_type = 'stop' AND evnt_stamp < 1690974000 LIMIT 10")
        return cursor.fetchall()
    
def get_session(session_id, connection):
    get_query = f"SELECT * FROM fct_hourly_metric WHERE session_id = '{session_id}' ORDER BY evnt_stamp ASC"
    with connection.cursor(cursor_factory=DictCursor) as cursor:
        cursor.execute(get_query)
        evt_logs = cursor.fetchall()
        return evt_logs
    
def get_sessions(session_ids, connection):
    get_query = f"SELECT * FROM fct_hourly_metric WHERE session_id = ANY(%s) ORDER BY session_id, evnt_stamp ASC"
    with connection.cursor(cursor_factory=DictCursor) as cursor:
        cursor.execute(get_query, (session_ids,))
        return cursor.fetchall()

def get_sessions_gains(connection):
    query = """
    SELECT user_id, session_id, item_id, evt_type, evnt_stamp
    FROM fct_hourly_metric
    WHERE session_id IN (
        SELECT session_id
        FROM (
            SELECT *
            FROM fct_hourly_metric
            WHERE evt_type = 'stop'
        ) AS temp_table
    )
    ORDER BY session_id, evt_type DESC, evnt_stamp ASC;
    """
    with connection.cursor(cursor_factory=DictCursor) as cursor:
        cursor.execute(query)
        return cursor.fetchall()

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

def insert_time_spent(time_spent_per_item, connection):
    insert_query = """
        INSERT INTO time_spent (
        user_id,
        session_id,
        item_id,
        time_spent,
        last_item
        ) VALUES (%s, %s, %s, %s, %s) ON CONFLICT (user_id, session_id, item_id) DO NOTHING
    """
    with connection.cursor() as cursor:
        total_items = len(time_spent_per_item)
        print(f"Total number of items to insert: {total_items}")
        
        batch_size = 10000
        session_batches = chunks(time_spent_per_item, batch_size)

        for i, session_batch in enumerate(session_batches, 1):
            print(f"Processing batch {i}, Number of sessions in batch: {len(session_batch)}")
            cursor.executemany(insert_query, session_batch)
            print(f"Inserted {i * batch_size} items which is {i * batch_size / total_items * 100}% of total items")


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def handle_calculate_time_spent():
    connection = create_pg_connection()

    batch_size = 10000
    all_time_spent_per_item = []
    time_spent_count = 0

    sessions = get_sessions_gains(connection)
    print(f"Number of sessions: {len(sessions)}")

    session_batches = chunks(sessions, batch_size)

    for i, session_batch in tqdm(enumerate(session_batches, 1)):
        print(f"Processing batch {i}, Number of sessions in batch: {len(session_batch)}")

        for session_id, group in groupby(sessions, key=itemgetter("session_id")):
            session_logs = list(group)

            time_spent_per_item = calculate_time_spent(session_logs)

            all_time_spent_per_item.extend(time_spent_per_item)
            time_spent_count += len(time_spent_per_item)


    print(f"Total number of view events: {time_spent_count}")
    insert_time_spent(all_time_spent_per_item, connection)

# %%
handle_calculate_time_spent()
