import pandas as pd
import numpy as np
import os
#import snowflake.connector
import boto3
from snowflake.connector.pandas_tools import write_pandas
#from dotenv import load_dotenv
from utils import *

def main():
    try:
        conn = get_snowflake_connector_rw()

        env = os.getenv("ENV")
        if env is None:
            raise RuntimeError("ENV is not set")

        env = env.lower()

        schema_map = {
            "dev":  "ANALYTICS.FINANCE_SCENARIOS",
            "uat":  "ANALYTICS.FINANCE_SCENARIOS_QA",
            "prod": "ANALYTICS.FINANCE_SCENARIOS_PROD",
        }

        if env not in schema_map:
            raise RuntimeError(f"Invalid ENV value: {env}")

        scenarios_schema = schema_map[env]

        forecasting_schema_map = {
            "dev":  "ANALYTICS.FINANCE_FORECASTING",
            "uat":  "ANALYTICS.FINANCE_FORECASTING_QA",
            "prod": "ANALYTICS.FINANCE_FORECASTING_PROD",
        }

        forecasting_schema = forecasting_schema_map[env]

        with open("data_pull.sql", "r") as f:
            sql_script = f.read()

        sql_script = (
            sql_script
            .replace("{{SCENARIOS_SCHEMA}}", scenarios_schema)
            .replace("{{FORECASTING_SCHEMA}}", forecasting_schema)
        )

        cur = conn.cursor()
        cur.execute(sql_script)
        cur.close()

        print(f"✅ Initial Data loading for Model training is successfully completed. You can start train your models ENV={env}")
        print(f" SCENARIOS → {scenarios_schema}")
        print(f" FORECASTING → {forecasting_schema}")

        return True

    except Exception as e:
        print("❌ Initial Data loading failed:", str(e))
        return False

if __name__ == "__main__":
    main()