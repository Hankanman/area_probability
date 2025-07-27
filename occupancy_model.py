"""
This module contains the occupancy model.
"""

import bisect
import math
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict

from db import AreaOccupancyStorage
from mock_types import Entity, EntityType, InputType, Likelihood


def compute_time_priors(
    storage: AreaOccupancyStorage, entry_id: str, slot_minutes: int = 60
):
    """
    Estimate P(occupied) per day_of_week and time_slot from motion sensor intervals.
    """
    # Get aggregated interval data
    interval_aggregates = storage.get_interval_aggregates(entry_id, slot_minutes)

    # Get time bounds
    first_time, last_time = storage.get_time_bounds(entry_id)

    if not first_time or not last_time:
        return []  # No data available

    # Calculate total time period
    days = (last_time.date() - first_time.date()).days + 1
    slots_per_day = (24 * 60) // slot_minutes
    slot_duration_seconds = slot_minutes * 60.0

    # Create lookup dictionary from aggregated results
    occupied_seconds = {}
    for day, slot, total_seconds in interval_aggregates:
        # Convert PostgreSQL day_of_week (0=Sunday) to Python weekday (0=Monday)
        python_weekday = (int(day) + 6) % 7
        occupied_seconds[(python_weekday, int(slot))] = float(total_seconds or 0)

    # Generate priors for all time slots
    now = datetime.now(timezone.utc)
    priors = []

    for day in range(7):
        for slot in range(slots_per_day):
            total_slot_seconds = days * slot_duration_seconds
            occupied_slot_seconds = occupied_seconds.get((day, slot), 0.0)

            # Calculate probability
            p = (
                occupied_slot_seconds / total_slot_seconds
                if total_slot_seconds > 0
                else 0.0
            )

            # Create the prior object within the storage's session
            prior = storage.AreaTimePriors(
                entry_id=entry_id,
                day_of_week=day,
                time_slot=slot,
                prior_value=p,
                data_points=int(total_slot_seconds),
                last_updated=now,
            )
            storage.session.add(prior)
            priors.append(prior)

    return priors


def compute_entity_likelihoods(storage: AreaOccupancyStorage, entry_id: str):
    """
    Compute P(sensor=true|occupied) and P(sensor=true|empty) per sensor.
    Use motion-based labels for 'occupied'.
    """
    # Get all sensor configs for this area
    sensors = storage.get_area_entity_configs(entry_id)

    # Get truth timeline from motion sensors (sorted for binary search)
    occupied_times = storage.get_motion_sensor_intervals(entry_id)

    def is_occupied_optimized(ts):
        """Efficiently check if timestamp falls within any occupied interval using binary search."""
        if not occupied_times:
            return False

        # Binary search to find the rightmost interval that starts <= ts
        idx = bisect.bisect_right([start for start, _ in occupied_times], ts)

        # Check if ts falls within the interval found
        if idx > 0:
            start, end = occupied_times[idx - 1]
            if start <= ts < end:
                return True

        return False

    sensor_entity_ids = [cfg.entity_id for cfg in sensors]
    if not sensor_entity_ids:
        return sensors

    # Get all intervals for all sensors in one query
    all_intervals = storage.get_sensor_intervals(sensor_entity_ids)

    # Group intervals by entity_id for processing
    intervals_by_entity = defaultdict(list)
    for interval in all_intervals:
        intervals_by_entity[interval.entity_id].append(interval)

    # Process each sensor's intervals
    now = datetime.now(timezone.utc)
    for cfg in sensors:
        # Get intervals for this specific sensor
        intervals = intervals_by_entity[cfg.entity_id]

        # Count true readings during occupied vs empty
        true_occ = false_occ = true_empty = false_empty = 0

        for iv in intervals:
            occ = is_occupied_optimized(iv.start_time)
            if iv.state == "on":
                if occ:
                    true_occ += iv.duration_seconds
                else:
                    true_empty += iv.duration_seconds
            elif occ:
                false_occ += iv.duration_seconds
            else:
                false_empty += iv.duration_seconds

        # Avoid division by zero
        pt = true_occ / (true_occ + false_occ) if (true_occ + false_occ) > 0 else 0.5
        pf = (
            true_empty / (true_empty + false_empty)
            if (true_empty + false_empty) > 0
            else 0.5
        )
        cfg.prob_given_true = pt
        cfg.prob_given_false = pf
        cfg.last_updated = now

    return sensors


def naive_bayes_predict(
    entities: Dict[str, Entity], area_prior: float = 0.5, time_prior: float = 0.5
):
    """
    Compute posterior probability of occupancy given current features, area prior, and time prior.

    Args:
        entities: Dict mapping entity_id to Entity objects containing evidence and likelihood
        area_prior: Base prior probability of occupancy for this area (default: 0.5)
        time_prior: Time-based modifier for the prior (default: 0.5)
    """
    # Clamp priors to avoid log(0) or log(1)
    area_prior = max(0.001, min(0.999, area_prior))
    time_prior = max(0.001, min(0.999, time_prior))

    # Combine area prior with time prior modifier
    # Use time_prior as a multiplier on the area_prior
    combined_prior = area_prior * time_prior / 0.5  # Normalize by default time prior

    # Clamp combined prior
    combined_prior = max(0.001, min(0.999, combined_prior))

    # log-space for numerical stability
    log_true = math.log(combined_prior)
    log_false = math.log(1 - combined_prior)

    for entity in entities.values():
        value = entity.evidence
        decay_factor = entity.decay.decay_factor

        # Apply decay to evidence strength
        # When evidence is True: decay_factor = 1.0 (full strength)
        # When evidence is False: decay_factor reduces over time (weakening negative evidence)
        if value:
            # Evidence is present - use full strength
            p_t = entity.likelihood.prob_given_true
            p_f = entity.likelihood.prob_given_false
        else:
            # Evidence is absent - apply decay to weaken the negative evidence
            # Interpolate between neutral (0.5) and full negative evidence based on decay
            neutral_prob = 0.5
            full_negative_t = 1 - entity.likelihood.prob_given_true
            full_negative_f = 1 - entity.likelihood.prob_given_false

            p_t = neutral_prob + (full_negative_t - neutral_prob) * decay_factor
            p_f = neutral_prob + (full_negative_f - neutral_prob) * decay_factor

        # Clamp probabilities to avoid log(0) or log(1)
        p_t = max(0.001, min(0.999, p_t))
        p_f = max(0.001, min(0.999, p_f))

        log_true += math.log(p_t) * entity.entity_type.weight
        log_false += math.log(p_f) * entity.entity_type.weight

    # convert back
    max_log = max(log_true, log_false)
    true_prob = math.exp(log_true - max_log)
    false_prob = math.exp(log_false - max_log)
    return true_prob / (true_prob + false_prob)


def calculate_area_prior(storage: AreaOccupancyStorage, entry_id: str) -> float:
    """
    Calculate the overall occupancy prior for an area based on historical motion sensor data.

    Args:
        storage: AreaOccupancyStorage instance
        entry_id: Area entry ID

    Returns:
        float: Prior probability of occupancy (0.0 to 1.0)
    """
    # Get total occupied time from motion sensors
    total_occupied_seconds = storage.get_total_occupied_seconds(entry_id)

    # Get total time period
    first_time, last_time = storage.get_time_bounds(entry_id)

    if not first_time or not last_time or total_occupied_seconds == 0:
        return 0.5  # Default prior if no data

    total_seconds = (last_time - first_time).total_seconds()

    if total_seconds <= 0:
        return 0.5

    return total_occupied_seconds / total_seconds


def update_area_prior(storage: AreaOccupancyStorage, entry_id: str) -> float:
    """
    Calculate and update the area prior for a specific area.

    Args:
        storage: AreaOccupancyStorage instance
        entry_id: Area entry ID

    Returns:
        float: The calculated prior value
    """
    prior_value = calculate_area_prior(storage, entry_id)
    storage.update_area_prior(entry_id, prior_value)
    return prior_value


def create_features_from_entities(entities: Dict[str, Entity]) -> Dict[str, Entity]:
    """
    Create features dictionary from Entity objects for use with naive_bayes_predict.

    Args:
        entities: Dictionary mapping entity_id to Entity objects

    Returns:
        Dict[str, Entity]: Features dictionary mapping entity_id to Entity objects
    """
    features = {}
    for entity_id, entity_data in entities.items():
        # Create Feature object from Entity data
        features[entity_id] = entity_data

    return features


def generate_entities_from_db(
    storage: AreaOccupancyStorage, entry_id: str
) -> Dict[str, Entity]:
    """
    Generate Entity objects from database configurations for an area.

    Args:
        storage: AreaOccupancyStorage instance
        entry_id: Area entry ID

    Returns:
        Dict[str, Entity]: Dictionary mapping entity_id to Entity objects
    """
    # Get all sensor configs for this area
    configs = storage.get_area_entity_configs(entry_id)

    entities = {}
    for cfg in configs:
        # Convert string entity_type to InputType enum
        try:
            input_type = InputType(cfg.entity_type)
        except ValueError:
            # Default to MOTION if entity_type is not in InputType enum
            input_type = InputType.MOTION

        # Create EntityType from config
        entity_type = EntityType(input_type=input_type, weight=cfg.weight)

        # Create Likelihood from config
        likelihood = Likelihood(
            prob_given_true=cfg.prob_given_true,
            prob_given_false=cfg.prob_given_false,
        )

        # Create Entity object
        entity = Entity(
            entity_id=cfg.entity_id,
            entity_type=entity_type,
            weight=cfg.weight,
            likelihood=likelihood,
        )

        entities[cfg.entity_id] = entity

    return entities


def prepare_features_for_prediction(
    storage: AreaOccupancyStorage, entry_id: str, entities: Dict[str, Entity]
) -> Dict[str, Entity]:
    """
    Prepare complete features dictionary for naive_bayes_predict.
    Ensures all required sensors for the area are present with default values for missing ones.

    Args:
        storage: AreaOccupancyStorage instance
        entry_id: Area entry ID
        entities: Dictionary mapping entity_id to Entity objects

    Returns:
        Dict[str, Entity]: Complete features dictionary with all required sensors
    """
    # Get all sensor configs for this area
    configs = storage.get_area_entity_configs(entry_id)

    # Create features from provided entities
    features = create_features_from_entities(entities)

    # Ensure all required sensors are present with default values
    for cfg in configs:
        if cfg.entity_id not in features:
            # Add missing sensor with default False evidence
            features[cfg.entity_id] = Entity(
                entity_id=cfg.entity_id,
                entity_type=cfg.entity_type,
                weight=cfg.weight,
                likelihood=Likelihood(
                    prob_given_true=cfg.prob_given_true,
                    prob_given_false=cfg.prob_given_false,
                ),
            )

    return features


def update_all_area_priors(storage: AreaOccupancyStorage) -> dict[str, float]:
    """
    Calculate and update area priors for all areas in the database.

    Args:
        storage: AreaOccupancyStorage instance

    Returns:
        dict: Mapping of entry_id to prior value
    """
    areas = storage.get_all_areas()
    results = {}

    for area in areas:
        prior_value = update_area_prior(storage, area.entry_id)
        results[area.entry_id] = prior_value

    return results
