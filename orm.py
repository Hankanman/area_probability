"""
This module contains the ORM operations for the area occupancy database.
"""

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from sqlalchemy import func, text

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

from db import DEFAULT_ENTITY_PROB_GIVEN_FALSE, DEFAULT_ENTITY_PROB_GIVEN_TRUE


class AreaOccupancyORM:
    """
    A class to manage area occupancy database ORM operations.
    """

    def __init__(self, session: "Session", model_classes):
        """
        Initialize the AreaOccupancyORM with session and model classes.

        Args:
            session: SQLAlchemy session
            model_classes: Dictionary containing model classes
        """
        self.session = session
        self.AreaOccupancy = model_classes["AreaOccupancy"]
        self.Entity = model_classes["Entity"]
        self.AreaEntityConfig = model_classes["AreaEntityConfig"]
        self.AreaTimePriors = model_classes["AreaTimePriors"]
        self.StateInterval = model_classes["StateInterval"]
        self.Metadata = model_classes["Metadata"]

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

            self.session.commit()
            return priors_deleted, len(configs)

        except Exception as e:
            self.session.rollback()
            raise e

    def import_from_sql(self, engine, file_path: str):
        """
        Import data from SQL file into the database.

        Args:
            engine: SQLAlchemy engine
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

            with engine.connect() as conn:
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

    def bulk_insert_state_intervals(self, engine, intervals_data: list[dict]):
        """
        Efficiently bulk insert state intervals using raw SQL for maximum performance.

        Args:
            engine: SQLAlchemy engine
            intervals_data: List of dicts with keys: entity_id, state, start_time, end_time, duration_seconds, created_at

        Returns:
            int: Number of intervals inserted
        """
        if not intervals_data:
            return 0

        # Use bulk insert for better performance
        with engine.begin() as conn:
            result = conn.execute(
                text("""
                    INSERT OR IGNORE INTO state_intervals
                    (entity_id, state, start_time, end_time, duration_seconds, created_at)
                    VALUES (:entity_id, :state, :start_time, :end_time, :duration_seconds, :created_at)
                """),
                intervals_data,
            )

            return result.rowcount

    def bulk_update_entity_likelihoods(self, engine, configs_data: list[dict]):
        """
        Efficiently bulk update entity likelihood configurations.

        Args:
            engine: SQLAlchemy engine
            configs_data: List of dicts with keys: entry_id, entity_id, prob_given_true, prob_given_false, last_updated

        Returns:
            int: Number of configs updated
        """
        if not configs_data:
            return 0

        with engine.begin() as conn:
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
            self.session.commit()
            return deleted
        except Exception as e:
            self.session.rollback()
            raise e

    def add_time_priors(self, priors: list):
        """
        Add time priors to the database.

        Args:
            priors: List of AreaTimePriors objects
        """
        try:
            self.session.add_all(priors)
            self.session.commit()
        except Exception as e:
            self.session.rollback()
            raise e

    def get_time_prior_at_timestamp(
        self, entry_id: str, timestamp, slot_minutes: int = 60
    ):
        """
        Get the time prior for a specific timestamp.

        Args:
            entry_id: Area entry ID
            timestamp: Timestamp to get prior for
            slot_minutes: Time slot size in minutes

        Returns:
            float: Time prior value or default 0.5 if not found
        """
        # Calculate day of week and time slot
        day_of_week = timestamp.weekday()  # 0=Monday, 6=Sunday
        time_slot = (timestamp.hour * 60 + timestamp.minute) // slot_minutes

        # Query for the specific time prior
        prior = (
            self.session.query(self.AreaTimePriors)
            .filter_by(entry_id=entry_id, day_of_week=day_of_week, time_slot=time_slot)
            .first()
        )

        return prior.prior_value if prior else 0.5

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
                self.session.commit()
        except Exception as e:
            self.session.rollback()
            raise e
