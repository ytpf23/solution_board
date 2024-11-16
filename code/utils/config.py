import os 

class params:
    psql_host = os.getenv("PSQL_HOST")
    psql_port = int(os.getenv("PSQL_PORT",'5432'))
    psql_user = os.getenv("PSQL_USER", "techcombinator")
    psql_password = os.getenv("PSQL_PASSWORD")
    psql_db = os.getenv("PSQL_DB", "postgres")
    psql_schema = os.getenv("PSQL_SCHEMA", "techcombinator")
    psql_conn_string='postgresql+psycopg2://{0}:{1}@{2}:{3}/{4}?options=-csearch_path%3Ddbo,{5}'.format(psql_user,psql_password,psql_host,psql_port,psql_db,psql_schema)
    collection_name = os.getenv("COLLECTION NAME", "test_paper")