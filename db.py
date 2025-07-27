"""
This module contains the database schema and functions to interact with the database.
"""

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
    create_engine,
    text,
)
from sqlalchemy.orm import relationship, sessionmaker

if TYPE_CHECKING:
    from sqlalchemy.orm import DeclarativeBase

    Base = DeclarativeBase

DEFAULT_AREA_PRIOR = 0.15
DEFAULT_ENTITY_WEIGHT = 0.85
DEFAULT_ENTITY_PROB_GIVEN_TRUE = 0.8
DEFAULT_ENTITY_PROB_GIVEN_FALSE = 0.05
DB_PATH = "sqlite:///area_occupancy.db"


class AreaOccupancyStorage:
    """
    A class to manage area occupancy database operations.
    """

    def __init__(self, db_path: str | None = None):
        """
        Initialize the AreaOccupancyStorage with the specified database path.

        Args:
            db_path: Database path (optional, defaults to DB_PATH)
        """
        self.db_path = db_path or DB_PATH
        self.engine = create_engine(
            self.db_path,
            echo=False,
            pool_pre_ping=True,
            connect_args={
                "check_same_thread": False,
                "timeout": 30,
            },
        )

        # Check if database exists and initialize if needed
        self._ensure_db_exists()

    class AreaOccupancy(Base):
        """
        A table to store the area occupancy information.
        """

        __tablename__ = "area_occupancy"
        entry_id = Column(String, primary_key=True)
        area_name = Column(String, nullable=False)
        purpose = Column(String, nullable=False)
        threshold = Column(Float, nullable=False)
        area_prior = Column(
            Float,
            nullable=False,
            default=DEFAULT_AREA_PRIOR,
            server_default=text(str(DEFAULT_AREA_PRIOR)),
        )
        created_at = Column(DateTime, nullable=False)
        updated_at = Column(DateTime, nullable=False)
        entities = relationship("AreaEntityConfig", back_populates="area")
        priors = relationship("AreaTimePriors", back_populates="area")

    class Entity(Base):
        """
        A table to store the entity information.
        """

        __tablename__ = "entities"
        entity_id = Column(String, primary_key=True)
        last_seen = Column(DateTime, nullable=False)
        created_at = Column(DateTime, nullable=False)
        intervals = relationship("StateInterval", back_populates="entity")
        configs = relationship("AreaEntityConfig", back_populates="entity")

    class AreaEntityConfig(Base):
        """
        A table to store the area entity configuration.
        """

        __tablename__ = "area_entity_config"
        entry_id = Column(
            String, ForeignKey("area_occupancy.entry_id"), primary_key=True
        )
        entity_id = Column(String, ForeignKey("entities.entity_id"), primary_key=True)
        entity_type = Column(String, nullable=False)
        weight = Column(Float, nullable=False, default=DEFAULT_ENTITY_WEIGHT)
        prob_given_true = Column(
            Float, nullable=False, default=DEFAULT_ENTITY_PROB_GIVEN_TRUE
        )
        prob_given_false = Column(
            Float, nullable=False, default=DEFAULT_ENTITY_PROB_GIVEN_FALSE
        )
        last_updated = Column(DateTime, nullable=False)
        area = relationship("AreaOccupancy", back_populates="entities")
        entity = relationship("Entity", back_populates="configs")

        __table_args__ = (
            Index("idx_area_entity_entry", "entry_id"),
            Index("idx_area_entity_type", "entry_id", "entity_type"),
        )

    class AreaTimePriors(Base):
        """
        A table to store the area time priors.
        """

        __tablename__ = "area_time_priors"
        entry_id = Column(
            String, ForeignKey("area_occupancy.entry_id"), primary_key=True
        )
        day_of_week = Column(Integer, primary_key=True)
        time_slot = Column(Integer, primary_key=True)
        prior_value = Column(Float, nullable=False)
        data_points = Column(Integer, nullable=False)
        last_updated = Column(DateTime, nullable=False)
        area = relationship("AreaOccupancy", back_populates="priors")

        __table_args__ = (
            Index("idx_area_time_priors_entry", "entry_id"),
            Index("idx_area_time_priors_day_slot", "day_of_week", "time_slot"),
            Index("idx_area_time_priors_last_updated", "last_updated"),
        )

    class StateInterval(Base):
        """
        A table to store the state intervals.
        """

        __tablename__ = "state_intervals"
        id = Column(Integer, primary_key=True)
        entity_id = Column(String, ForeignKey("entities.entity_id"), nullable=False)
        state = Column(String, nullable=False)
        start_time = Column(DateTime, nullable=False)
        end_time = Column(DateTime, nullable=False)
        duration_seconds = Column(Float, nullable=False)
        created_at = Column(DateTime, nullable=False)
        entity = relationship("Entity", back_populates="intervals")

        # Add unique constraint on (entity_id, start_time, end_time)
        __table_args__ = (
            UniqueConstraint(
                "entity_id", "start_time", "end_time", name="uq_intervals_entity_time"
            ),
            # Performance indexes
            Index("idx_state_intervals_entity", "entity_id"),
            Index(
                "idx_state_intervals_entity_time", "entity_id", "start_time", "end_time"
            ),
            Index("idx_state_intervals_start_time", "start_time"),
            Index("idx_state_intervals_end_time", "end_time"),
        )

    class Metadata(Base):
        """
        A table to store the metadata.
        """

        __tablename__ = "metadata"
        key = Column(String, primary_key=True)
        value = Column(String, nullable=False)

    def _ensure_db_exists(self):
        """
        Check if the database exists and initialize it if needed.
        """
        # Check if any tables exist by trying to query the metadata table
        try:
            with self.engine.connect() as conn:
                # Try to query a table to see if the database is initialized
                conn.execute(
                    text("SELECT name FROM sqlite_master WHERE type='table' LIMIT 1")
                )
        except Exception:
            # Database doesn't exist or is not initialized, create it
            self.init_db()

    def get_engine(self):
        """
        Get the engine for the database with optimized settings.
        """
        return self.engine

    def get_session(self):
        """
        Get the session for the database.
        """
        engine = self.get_engine()
        session = sessionmaker(bind=engine)
        return session()

    def init_db(self):
        """
        Initialize the database.
        """
        Base.metadata.create_all(self.engine)

    def reset_learned_parameters(self):
        """
        Delete all time priors and reset entity likelihoods to default values.

        Returns:
            tuple: (priors_deleted: int, configs_reset: int)
        """
        session = self.get_session()

        try:
            # Delete all time priors
            priors_deleted = session.query(self.AreaTimePriors).delete()

            # Reset likelihoods to default values
            configs = session.query(self.AreaEntityConfig).all()
            now = datetime.now(timezone.utc)
            for config in configs:
                config.prob_given_true = self.DEFAULT_ENTITY_PROB_GIVEN_TRUE
                config.prob_given_false = self.DEFAULT_ENTITY_PROB_GIVEN_FALSE
                config.last_updated = now

            session.commit()
            return priors_deleted, len(configs)

        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def import_from_sql(self, file_path: str):
        """
        Import data from SQL file into the database.

        Args:
            file_path: Path to the SQL file to import

        Returns:
            tuple: (success: bool, message: str)
        """
        if not Path(file_path).exists():
            return False, f"File '{file_path}' not found."

        try:
            with Path(file_path).open() as f:
                sql_content = f.read()

            # Split by semicolon to handle multiple statements
            statements = [
                stmt.strip() for stmt in sql_content.split(";") if stmt.strip()
            ]

            # Regex to match legacy area_occupancy INSERTs without column list, with or without quotes
            area_occ_insert_re = re.compile(
                r"^INSERT INTO [\"']?area_occupancy[\"']? VALUES ?\(([^)]*)\)$",
                re.IGNORECASE,
            )
            area_occ_columns = (
                "(entry_id, area_name, purpose, threshold, created_at, updated_at)"
            )

            with self.engine.connect() as conn:
                # Start a transaction
                trans = conn.begin()
                try:
                    for statement in statements:
                        if statement:  # Skip empty statements
                            # Skip transaction control statements
                            if statement.upper().startswith(
                                ("BEGIN", "COMMIT", "ROLLBACK")
                            ):
                                continue
                            # Rewrite legacy area_occupancy INSERTs
                            m = area_occ_insert_re.match(statement)
                            if m:
                                values = m.group(1)
                                rewritten_statement = f"INSERT INTO area_occupancy {area_occ_columns} VALUES ({values})"
                            else:
                                rewritten_statement = statement
                            # Execute the statement using text()
                            conn.execute(text(rewritten_statement))
                    # Commit the transaction
                    trans.commit()
                    return True, f"Successfully imported data from '{file_path}'."
                except Exception as e:
                    trans.rollback()
                    raise e

        except Exception as e:
            return False, f"Error importing data: {e}"

    def bulk_insert_state_intervals(self, intervals_data: list[dict]):
        """
        Efficiently bulk insert state intervals using raw SQL for maximum performance.

        Args:
            intervals_data: List of dicts with keys: entity_id, state, start_time, end_time, duration_seconds, created_at

        Returns:
            int: Number of intervals inserted
        """
        if not intervals_data:
            return 0

        # Use bulk insert for better performance
        with self.engine.begin() as conn:
            result = conn.execute(
                text("""
                    INSERT OR IGNORE INTO state_intervals
                    (entity_id, state, start_time, end_time, duration_seconds, created_at)
                    VALUES (:entity_id, :state, :start_time, :end_time, :duration_seconds, :created_at)
                """),
                intervals_data,
            )

            return result.rowcount

    def bulk_update_entity_likelihoods(self, configs_data: list[dict]):
        """
        Efficiently bulk update entity likelihood configurations.

        Args:
            configs_data: List of dicts with keys: entry_id, entity_id, prob_given_true, prob_given_false, last_updated

        Returns:
            int: Number of configs updated
        """
        if not configs_data:
            return 0

        with self.engine.begin() as conn:
            result = conn.execute(
                text("""
                    UPDATE area_entity_config
                    SET prob_given_true = :prob_given_true,
                        prob_given_false = :prob_given_false,
                        last_updated = :last_updated
                    WHERE entry_id = :entry_id AND entity_id = :entity_id
                """),
                configs_data,
            )

            return result.rowcount
