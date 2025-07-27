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
    func,
    text,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

if TYPE_CHECKING:
    from sqlalchemy.orm import DeclarativeBase

    Base = DeclarativeBase
else:
    Base = declarative_base()

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

        # Create and store session
        self.session = self.get_session()

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

    def commit(self):
        """
        Commit the current session.
        """
        if self.session:
            self.session.commit()

    def rollback(self):
        """
        Rollback the current session.
        """
        if self.session:
            self.session.rollback()

    def close(self):
        """
        Close the current session.
        """
        if self.session:
            self.session.close()
            self.session = None

    def refresh_session(self):
        """
        Create a new session if the current one is closed.
        """
        if not self.session:
            self.session = self.get_session()

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
        try:
            # Delete all time priors
            priors_deleted = self.session.query(self.AreaTimePriors).delete()

            # Reset likelihoods to default values
            configs = self.session.query(self.AreaEntityConfig).all()
            now = datetime.now(timezone.utc)
            for config in configs:
                config.prob_given_true = DEFAULT_ENTITY_PROB_GIVEN_TRUE
                config.prob_given_false = DEFAULT_ENTITY_PROB_GIVEN_FALSE
                config.last_updated = now

            self.commit()
            return priors_deleted, len(configs)

        except Exception as e:
            self.rollback()
            raise e

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

    def get_area_by_identifier(self, area_identifier: str):
        """
        Get an area by entry_id or area_name (case-insensitive).

        Args:
            area_identifier: Area entry_id or area_name

        Returns:
            AreaOccupancy object or None if not found
        """
        # First try to find by entry_id (UUID) - exact match
        area = (
            self.session.query(self.AreaOccupancy)
            .filter_by(entry_id=area_identifier)
            .first()
        )
        if area:
            return area

        area = (
            self.session.query(self.AreaOccupancy)
            .filter(func.lower(self.AreaOccupancy.area_name) == area_identifier.lower())
            .first()
        )
        return area

    def get_all_areas(self):
        """
        Get all areas in the database.

        Returns:
            List of AreaOccupancy objects
        """
        return self.session.query(self.AreaOccupancy).all()

    def get_area_entity_configs(self, entry_id: str):
        """
        Get all entity configurations for a specific area.

        Args:
            entry_id: Area entry ID

        Returns:
            List of AreaEntityConfig objects
        """
        return (
            self.session.query(self.AreaEntityConfig).filter_by(entry_id=entry_id).all()
        )

    def get_motion_sensor_intervals(self, entry_id: str):
        """
        Get all motion sensor intervals for a specific area.

        Args:
            entry_id: Area entry ID

        Returns:
            List of (start_time, end_time) tuples
        """
        intervals = (
            self.session.query(
                self.StateInterval.start_time, self.StateInterval.end_time
            )
            .join(
                self.AreaEntityConfig,
                self.StateInterval.entity_id == self.AreaEntityConfig.entity_id,
            )
            .filter(
                self.AreaEntityConfig.entry_id == entry_id,
                self.AreaEntityConfig.entity_type == "motion",
                self.StateInterval.state == "on",
            )
            .order_by(self.StateInterval.start_time)
            .all()
        )
        return [(start, end) for start, end in intervals]

    def get_interval_aggregates(self, entry_id: str, slot_minutes: int = 60):
        """
        Get aggregated interval data for time prior computation.

        Args:
            entry_id: Area entry ID
            slot_minutes: Time slot size in minutes

        Returns:
            List of (day_of_week, time_slot, total_occupied_seconds) tuples
        """

        interval_aggregates = (
            self.session.query(
                func.extract("dow", self.StateInterval.start_time).label("day_of_week"),
                func.floor(
                    (
                        func.extract("hour", self.StateInterval.start_time) * 60
                        + func.extract("minute", self.StateInterval.start_time)
                    )
                    / slot_minutes
                ).label("time_slot"),
                func.sum(self.StateInterval.duration_seconds).label(
                    "total_occupied_seconds"
                ),
            )
            .join(
                self.AreaEntityConfig,
                self.StateInterval.entity_id == self.AreaEntityConfig.entity_id,
            )
            .filter(
                self.AreaEntityConfig.entry_id == entry_id,
                self.AreaEntityConfig.entity_type == "motion",
                self.StateInterval.state == "on",
            )
            .group_by("day_of_week", "time_slot")
            .all()
        )
        return [
            (day, slot, total_seconds)
            for day, slot, total_seconds in interval_aggregates
        ]

    def get_time_bounds(self, entry_id: str):
        """
        Get time bounds for a specific area.

        Args:
            entry_id: Area entry ID

        Returns:
            Tuple of (first_time, last_time) or (None, None) if no data
        """

        time_bounds = (
            self.session.query(
                func.min(self.StateInterval.start_time).label("first"),
                func.max(self.StateInterval.end_time).label("last"),
            )
            .join(
                self.AreaEntityConfig,
                self.StateInterval.entity_id == self.AreaEntityConfig.entity_id,
            )
            .filter(self.AreaEntityConfig.entry_id == entry_id)
            .first()
        )
        return (time_bounds.first, time_bounds.last) if time_bounds else (None, None)

    def get_sensor_intervals(self, entity_ids: list[str]):
        """
        Get all intervals for specific sensor entities.

        Args:
            entity_ids: List of entity IDs

        Returns:
            List of StateInterval objects
        """
        return (
            self.session.query(self.StateInterval)
            .filter(self.StateInterval.entity_id.in_(entity_ids))
            .all()
        )

    def get_sensor_intervals_for_area(self, entry_id: str):
        """
        Get all sensor intervals for a specific area.

        Args:
            entry_id: Area entry ID

        Returns:
            List of StateInterval objects
        """
        return (
            self.session.query(self.StateInterval)
            .join(
                self.AreaEntityConfig,
                self.StateInterval.entity_id == self.AreaEntityConfig.entity_id,
            )
            .filter(self.AreaEntityConfig.entry_id == entry_id)
            .all()
        )

    def get_sensor_state_at_time(self, entity_id: str, timestamp):
        """
        Get the sensor state at a specific timestamp.

        Args:
            entity_id: Entity ID
            timestamp: Timestamp to check

        Returns:
            str: Sensor state ("on" or "off") or None if no data
        """
        interval = (
            self.session.query(self.StateInterval)
            .filter(
                self.StateInterval.entity_id == entity_id,
                self.StateInterval.start_time <= timestamp,
                self.StateInterval.end_time > timestamp,
            )
            .first()
        )
        return interval.state if interval else None

    def get_total_occupied_seconds(self, entry_id: str):
        """
        Get total occupied seconds for an area.

        Args:
            entry_id: Area entry ID

        Returns:
            Total occupied seconds or 0 if no data
        """

        occupied_result = (
            self.session.query(func.sum(self.StateInterval.duration_seconds))
            .join(
                self.AreaEntityConfig,
                self.StateInterval.entity_id == self.AreaEntityConfig.entity_id,
            )
            .filter(
                self.AreaEntityConfig.entry_id == entry_id,
                self.AreaEntityConfig.entity_type == "motion",
                self.StateInterval.state == "on",
            )
            .scalar()
        )
        return float(occupied_result or 0)

    def delete_area_time_priors(self, entry_id: str):
        """
        Delete all time priors for a specific area.

        Args:
            entry_id: Area entry ID

        Returns:
            Number of priors deleted
        """
        try:
            deleted = (
                self.session.query(self.AreaTimePriors)
                .filter_by(entry_id=entry_id)
                .delete()
            )
            self.commit()
            return deleted
        except Exception as e:
            self.rollback()
            raise e

    def add_time_priors(self, priors: list):
        """
        Add time priors to the database.

        Args:
            priors: List of AreaTimePriors objects
        """
        try:
            self.session.add_all(priors)
            self.commit()
        except Exception as e:
            self.rollback()
            raise e

    def update_area_prior(self, entry_id: str, prior_value: float):
        """
        Update the area prior for a specific area.

        Args:
            entry_id: Area entry ID
            prior_value: New prior value
        """
        try:
            area = (
                self.session.query(self.AreaOccupancy)
                .filter_by(entry_id=entry_id)
                .first()
            )
            if area:
                area.area_prior = prior_value
                area.updated_at = datetime.now(timezone.utc)
                self.commit()
        except Exception as e:
            self.rollback()
            raise e
