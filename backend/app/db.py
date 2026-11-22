from collections.abc import Iterator

import psycopg
from pgvector.psycopg import register_vector
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from app.config import get_settings
from app.models import Base


engine = create_engine(get_settings().database_url, pool_pre_ping=True)


@event.listens_for(engine, "connect")
def _register_vector(dbapi_connection, connection_record):
    register_vector(dbapi_connection)


SessionLocal = sessionmaker(engine)


def init_db() -> None:
    database_url = get_settings().database_url.replace("postgresql+psycopg://", "postgresql://", 1)
    with psycopg.connect(database_url) as connection:
        connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
    Base.metadata.create_all(engine)


def get_session() -> Iterator[Session]:
    with SessionLocal() as session:
        yield session
