from sqlalchemy import create_engine
import os
from dotenv import load_dotenv

load_dotenv()  # Loads .env from the current directory
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://asiabarbato:asiaamstel@localhost/designstreamdb")
engine = create_engine(DATABASE_URL)

with engine.connect() as connection:
    print("Database connection successful")