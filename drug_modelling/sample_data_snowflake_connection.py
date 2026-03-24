import os
import json
import snowflake.connector
import pandas as pd
import logging
import sys
import boto3

# ✅ Send logs to stdout so SageMaker can capture them
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

def get_snowflake_connector(secret_name="snowflake/credentials", region_name="us-east-1"):
    """
    Fetch Snowflake credentials from AWS Secrets Manager and return a live connection.
    The secret must contain keys: user, password, account, warehouse, database, schema, role
    """
    # 1️⃣ Initialize Secrets Manager client
    client = boto3.client("secretsmanager", region_name=region_name)

    # 2️⃣ Retrieve the secret value
    secret_value = client.get_secret_value(SecretId=secret_name)
    secret_dict = json.loads(secret_value["SecretString"])

    # 3️⃣ Connect to Snowflake using secrets
    conn = snowflake.connector.connect(
        user=secret_dict["user"],
        password=secret_dict["password"],
        account=secret_dict["account"],
        warehouse=secret_dict.get("warehouse"),
        database=secret_dict.get("database"),
        schema=secret_dict.get("schema"),
        role=secret_dict.get("role"),
    )
    logging.info("✅ Connected to Snowflake.")
    return conn


def main():
    logging.info("🔹 Starting Snowflake connection test...")
    conn = get_snowflake_connector()
    cur = conn.cursor()

    cur.execute("SELECT CURRENT_TIMESTAMP()")
    snowflake_time = cur.fetchone()[0]  # this is a datetime object
    logging.info(f"Snowflake current time: {snowflake_time}")
    #print("snowflake_time")
    cur.close()


    query = f"""
        SELECT *
        FROM TABLE_NAME
        LIMIT 10
    """
    df = pd.read_sql(query, conn)
    logging.info("✅ Query executed successfully, showing top 5 rows:")
    print(df)

    conn.close()
    logging.info("🔚 Connection closed successfully.")


if __name__ == "__main__":
    main()