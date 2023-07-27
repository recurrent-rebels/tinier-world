import psycopg2


connection = psycopg2.connect(
    host="localhost",
    user="root",
    port=5432,
    database="W9sV6cL2dX",
    password="E5rG7tY3fH",
)
connection.autocommit = True
cursor = connection.cursor()


cursor.execute(
    """
  BEGIN;

  CREATE TABLE IF NOT EXISTS items (
    item_key   UUID        PRIMARY KEY,
    created_at TIMESTAMP   DEFAULT NOW(),
    user_id    INTEGER     NOT NULL,
    bucket_key VARCHAR     NOT NULL,
    type       VARCHAR(5)  NOT NULL
  );

  CREATE TABLE IF NOT EXISTS users (
    id        INTEGER PRIMARY KEY,
    gender    VARCHAR(6),
    country   VARCHAR(20),
    age       INTEGER
  );

  CREATE TABLE IF NOT EXISTS time_spent_on_item (
    user_id     INTEGER,
    item_id     UUID,
    time_spent  INTEGER,
    last_item   BOOLEAN,
    evnt_stamp INTEGER
  );

  CREATE TABLE IF NOT EXISTS item_embeddings (
    item_id     UUID,
    embedding   FLOAT[],
    time_stamp  TIMESTAMP DEFAULT NOW()
  );

  CREATE TABLE IF NOT EXISTS fct_hourly_metric (
    date_stamp DATE      NOT NULL DEFAULT CURRENT_DATE,
    time_stamp TIMESTAMP NOT NULL DEFAULT date_trunc('hour', CURRENT_TIMESTAMP AT TIME ZONE 'UTC'),
    evnt_stamp INTEGER            DEFAULT extract(epoch from NOW()),
    user_id    INTEGER,
    item_id    UUID,
    evt_type   VARCHAR(20),
    session_id UUID
  ) PARTITION BY RANGE (date_stamp);

  DO $$
  DECLARE
    i INTEGER;
    j INTEGER;
  BEGIN
    FOR i IN -5..50 LOOP
      DECLARE
        start_date DATE := (CURRENT_DATE + i) :: DATE;
        endin_date DATE := (CURRENT_DATE + i + 1) :: DATE;
        start_date_time TIMESTAMP := start_date :: TIMESTAMP;
        endin_date_time TIMESTAMP := endin_date :: TIMESTAMP;
      BEGIN
        -- RAISE NOTICE 'ITEM: %', i;
        -- RAISE NOTICE 'DATE: %', TO_CHAR(start_date, 'DD_MM_YYYY');

        EXECUTE format(
          '
            CREATE TABLE IF NOT EXISTS fct_hourly_metric_%s
            PARTITION OF fct_hourly_metric FOR VALUES FROM (%L) TO (%L)
            PARTITION BY RANGE (time_stamp)
          ',
          TO_CHAR(start_date, 'DD_MM_YYYY'),
          start_date,
          endin_date
        );

        FOR j IN 1..24 LOOP
          DECLARE
            start_time TIMESTAMP := (start_date :: TIMESTAMP) + (j - 1) * INTERVAL '1 hour';
            endin_time TIMESTAMP := (start_date :: TIMESTAMP) + j * INTERVAL '1 hour';
          BEGIN
            -- RAISE NOTICE 'TIME: %', start_time;
            -- RAISE NOTICE 'TIME: %', TO_CHAR(start_time, 'DD_MM_YYYY_HH24');

            EXECUTE format(
              '
                CREATE TABLE IF NOT EXISTS fct_hourly_metric_%s
                PARTITION OF fct_hourly_metric_%s FOR VALUES FROM (%L) TO (%L)
              ',
              TO_CHAR(start_time, 'DD_MM_YYYY_HH24'),
              TO_CHAR(start_date, 'DD_MM_YYYY'),
              start_time,
              endin_time
            );
          END;
        END LOOP;
      END;
    END LOOP;
  END;
  $$ LANGUAGE plpgsql;

  COMMIT;
"""
)


cursor.close()
connection.close()
