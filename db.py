"""
This module contains the database schema and functions to interact with the database.
"""

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
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

from orm import AreaOccupancyORM

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

        # Initialize ORM
        self._init_orm()

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
            # Update ORM session
            self.orm.session = self.session

    def init_db(self):
        """
        Initialize the database.
        """
        Base.metadata.create_all(self.engine)

        # Re-initialize ORM after creating tables
        self._init_orm()

    def _init_orm(self):
        """
        Initialize the ORM with session and model classes.
        """
        # Create model classes dictionary for ORM
        self.model_classes = {
            "AreaOccupancy": self.AreaOccupancy,
            "Entity": self.Entity,
            "AreaEntityConfig": self.AreaEntityConfig,
            "AreaTimePriors": self.AreaTimePriors,
            "StateInterval": self.StateInterval,
            "Metadata": self.Metadata,
        }

        # Initialize ORM
        self.orm = AreaOccupancyORM(self.session, self.model_classes)

    def reset_learned_parameters(self):
        """Delegate to ORM."""
        return self.orm.reset_learned_parameters()

    def import_from_sql(self, file_path: str):
        """Delegate to ORM."""
        return self.orm.import_from_sql(self.engine, file_path)

    def bulk_insert_state_intervals(self, intervals_data: list[dict]):
        """Delegate to ORM."""
        return self.orm.bulk_insert_state_intervals(self.engine, intervals_data)

    def bulk_update_entity_likelihoods(self, configs_data: list[dict]):
        """Delegate to ORM."""
        return self.orm.bulk_update_entity_likelihoods(self.engine, configs_data)

    def get_area_by_identifier(self, area_identifier: str):
        """Delegate to ORM."""
        return self.orm.get_area_by_identifier(area_identifier)

    def get_all_areas(self):
        """Delegate to ORM."""
        return self.orm.get_all_areas()

    def get_area_entity_configs(self, entry_id: str):
        """Delegate to ORM."""
        return self.orm.get_area_entity_configs(entry_id)

    def get_motion_sensor_intervals(self, entry_id: str):
        """Delegate to ORM."""
        return self.orm.get_motion_sensor_intervals(entry_id)

    def get_interval_aggregates(self, entry_id: str, slot_minutes: int = 60):
        """Delegate to ORM."""
        return self.orm.get_interval_aggregates(entry_id, slot_minutes)

    def get_time_bounds(self, entry_id: str):
        """Delegate to ORM."""
        return self.orm.get_time_bounds(entry_id)

    def get_sensor_intervals(self, entity_ids: list[str]):
        """Delegate to ORM."""
        return self.orm.get_sensor_intervals(entity_ids)

    def get_sensor_intervals_for_area(self, entry_id: str):
        """Delegate to ORM."""
        return self.orm.get_sensor_intervals_for_area(entry_id)

    def get_sensor_state_at_time(self, entity_id: str, timestamp):
        """Delegate to ORM."""
        return self.orm.get_sensor_state_at_time(entity_id, timestamp)

    def get_total_occupied_seconds(self, entry_id: str):
        """Delegate to ORM."""
        return self.orm.get_total_occupied_seconds(entry_id)

    def delete_area_time_priors(self, entry_id: str):
        """Delegate to ORM."""
        return self.orm.delete_area_time_priors(entry_id)

    def add_time_priors(self, priors: list):
        """Delegate to ORM."""
        return self.orm.add_time_priors(priors)

    def get_time_prior_at_timestamp(
        self, entry_id: str, timestamp, slot_minutes: int = 60
    ):
        """Delegate to ORM."""
        return self.orm.get_time_prior_at_timestamp(entry_id, timestamp, slot_minutes)

    def update_area_prior(self, entry_id: str, prior_value: float):
        """Delegate to ORM."""
        return self.orm.update_area_prior(entry_id, prior_value)
