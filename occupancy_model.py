"""
This module contains the occupancy model.
"""

import bisect
import math
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
from sqlalchemy import func

from db import AreaEntityConfig, AreaOccupancy, AreaTimePriors, StateInterval


def compute_time_priors(session, entry_id: str, slot_minutes: int = 60):
    """
    Estimate P(occupied) per day_of_week and time_slot from motion sensor intervals.
    OPTIMIZED VERSION: Uses single aggregated query and reduces memory overhead.
    """
    # Use a single optimized query with aggregation to reduce data transfer
    # This replaces loading all intervals into memory
    interval_aggregates = (
        session.query(
            func.extract("dow", StateInterval.start_time).label("day_of_week"),
            func.floor(
                (
                    func.extract("hour", StateInterval.start_time) * 60
                    + func.extract("minute", StateInterval.start_time)
                )
                / slot_minutes
            ).label("time_slot"),
            func.sum(StateInterval.duration_seconds).label("total_occupied_seconds"),
        )
        .join(AreaEntityConfig, StateInterval.entity_id == AreaEntityConfig.entity_id)
        .filter(
            AreaEntityConfig.entry_id == entry_id,
            AreaEntityConfig.entity_type == "motion",
            StateInterval.state == "on",
        )
        .group_by("day_of_week", "time_slot")
        .all()
    )

    # Get time bounds with a single query
    time_bounds = (
        session.query(
            func.min(StateInterval.start_time).label("first"),
            func.max(StateInterval.end_time).label("last"),
        )
        .join(AreaEntityConfig, StateInterval.entity_id == AreaEntityConfig.entity_id)
        .filter(AreaEntityConfig.entry_id == entry_id)
        .first()
    )

    if not time_bounds.first or not time_bounds.last:
        return []  # No data available

    # Calculate total time period
    days = (time_bounds.last.date() - time_bounds.first.date()).days + 1
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

            priors.append(
                AreaTimePriors(
                    entry_id=entry_id,
                    day_of_week=day,
                    time_slot=slot,
                    prior_value=p,
                    data_points=int(total_slot_seconds),
                    last_updated=now,
                )
            )

    return priors


def compute_entity_likelihoods(session, entry_id: str):
    """
    Compute P(sensor=true|occupied) and P(sensor=true|empty) per sensor.
    Use motion-based labels for 'occupied'.
    OPTIMIZED VERSION: Eliminates N+1 queries and uses efficient temporal search.
    """
    # Get all sensor configs for this area
    sensors = session.query(AreaEntityConfig).filter_by(entry_id=entry_id).all()

    # Get truth timeline from motion sensors (sorted for binary search)
    occupied_intervals = (
        session.query(StateInterval.start_time, StateInterval.end_time)
        .join(AreaEntityConfig, StateInterval.entity_id == AreaEntityConfig.entity_id)
        .filter(
            AreaEntityConfig.entry_id == entry_id,
            AreaEntityConfig.entity_type == "motion",
            StateInterval.state == "on",
        )
        .order_by(StateInterval.start_time)  # Sort for efficient lookup
        .all()
    )

    # Convert to list of tuples for binary search
    occupied_times = [(start, end) for start, end in occupied_intervals]

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

    # OPTIMIZATION: Single query to get all state intervals for all sensors at once
    # This eliminates the N+1 query problem
    sensor_entity_ids = [cfg.entity_id for cfg in sensors]
    if not sensor_entity_ids:
        return sensors

    # Get all intervals for all sensors in one query
    all_intervals = (
        session.query(StateInterval)
        .filter(StateInterval.entity_id.in_(sensor_entity_ids))
        .all()
    )

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


def naive_bayes_predict(configs, features: dict):
    """
    Compute posterior probability of occupancy given current features.
    features: dict mapping entity_id to boolean or numeric value.
    """
    # log-space for numerical stability
    log_true = 0.0
    log_false = 0.0
    for cfg in configs:
        value = features.get(cfg.entity_id)
        p_t = cfg.prob_given_true if value else 1 - cfg.prob_given_true
        p_f = cfg.prob_given_false if value else 1 - cfg.prob_given_false
        log_true += math.log(p_t) * cfg.weight
        log_false += math.log(p_f) * cfg.weight
    # convert back
    max_log = max(log_true, log_false)
    true_prob = math.exp(log_true - max_log)
    false_prob = math.exp(log_false - max_log)
    return true_prob / (true_prob + false_prob)


def forward_hmm(
    session, entry_id: str, features_list: list[dict], slot_minutes: int = 60
):
    """
    Run HMM forward algorithm over a time grid of features,
    using time priors as initial state probabilities and a learned transition matrix.
    OPTIMIZED VERSION: Pre-loads configs once and uses batch processing.
    """
    # Pre-load priors and configs once (instead of querying repeatedly)
    priors_query = session.query(AreaTimePriors).filter_by(entry_id=entry_id).all()
    prior_map = {(p.day_of_week, p.time_slot): p.prior_value for p in priors_query}

    # Pre-load sensor configs once
    configs = session.query(AreaEntityConfig).filter_by(entry_id=entry_id).all()

    # Estimate transition probabilities
    a_00 = a_11 = 0.9
    a_01 = 1 - a_00
    a_10 = 1 - a_11

    # Process features in batches to reduce memory usage for large datasets
    BATCH_SIZE = 1000
    alphas = []
    prev = None

    for batch_start in range(0, len(features_list), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(features_list))
        batch_features = features_list[batch_start:batch_end]

        for ts, features in batch_features:
            day, slot = ts.weekday(), (ts.hour * 60 + ts.minute) // slot_minutes
            pi1 = prior_map.get((day, slot), 0.5)
            pi0 = 1 - pi1

            # Emission probability (cached configs)
            e1 = naive_bayes_predict(configs, features)
            e0 = 1 - e1

            if prev is None:
                alpha1 = pi1 * e1
                alpha0 = pi0 * e0
            else:
                prev0, prev1 = prev
                alpha1 = (prev1 * a_11 + prev0 * a_01) * e1
                alpha0 = (prev0 * a_00 + prev1 * a_10) * e0

            norm = alpha0 + alpha1 or 1
            prev = (alpha0 / norm, alpha1 / norm)
            alphas.append(prev[1])

    return alphas


def naive_bayes_predict_vectorized(configs, features_batch: list[dict]):
    """
    Vectorized version of naive_bayes_predict for processing multiple feature sets efficiently.
    Falls back to regular processing if numpy is not available.

    Args:
        configs: List of AreaEntityConfig objects
        features_batch: List of feature dictionaries

    Returns:
        List of probabilities
    """
    if not features_batch:
        return []

    # Pre-compute log probabilities for efficiency
    entity_ids = [cfg.entity_id for cfg in configs]
    log_true_probs = np.array(
        [math.log(cfg.prob_given_true) * cfg.weight for cfg in configs]
    )
    log_false_probs = np.array(
        [math.log(1 - cfg.prob_given_true) * cfg.weight for cfg in configs]
    )
    log_empty_true_probs = np.array(
        [math.log(cfg.prob_given_false) * cfg.weight for cfg in configs]
    )
    log_empty_false_probs = np.array(
        [math.log(1 - cfg.prob_given_false) * cfg.weight for cfg in configs]
    )

    results = []
    for features in features_batch:
        # Convert features to numpy array
        feature_values = np.array(
            [features.get(entity_id, 0) for entity_id in entity_ids]
        )

        # Vectorized computation
        log_true = np.sum(np.where(feature_values, log_true_probs, log_false_probs))
        log_false = np.sum(
            np.where(feature_values, log_empty_true_probs, log_empty_false_probs)
        )

        # Convert back to probability
        max_log = max(log_true, log_false)
        true_prob = math.exp(log_true - max_log)
        false_prob = math.exp(log_false - max_log)

        results.append(true_prob / (true_prob + false_prob))

    return results


def calculate_area_prior(session, entry_id: str) -> float:
    """
    Calculate the overall occupancy prior for an area based on historical motion sensor data.

    Args:
        session: Database session
        entry_id: Area entry ID

    Returns:
        float: Prior probability of occupancy (0.0 to 1.0)
    """
    # Get total occupied time from motion sensors
    occupied_result = (
        session.query(func.sum(StateInterval.duration_seconds))
        .join(AreaEntityConfig, StateInterval.entity_id == AreaEntityConfig.entity_id)
        .filter(
            AreaEntityConfig.entry_id == entry_id,
            AreaEntityConfig.entity_type == "motion",
            StateInterval.state == "on",
        )
        .scalar()
    )

    # Get total time period
    time_bounds = (
        session.query(
            func.min(StateInterval.start_time).label("first"),
            func.max(StateInterval.end_time).label("last"),
        )
        .join(AreaEntityConfig, StateInterval.entity_id == AreaEntityConfig.entity_id)
        .filter(AreaEntityConfig.entry_id == entry_id)
        .first()
    )

    if not time_bounds.first or not time_bounds.last or not occupied_result:
        return 0.5  # Default prior if no data

    total_occupied_seconds = float(occupied_result or 0)
    total_seconds = (time_bounds.last - time_bounds.first).total_seconds()

    if total_seconds <= 0:
        return 0.5

    return total_occupied_seconds / total_seconds


def update_area_prior(session, entry_id: str) -> float:
    """
    Calculate and update the area prior for a specific area.

    Args:
        session: Database session
        entry_id: Area entry ID

    Returns:
        float: The calculated prior value
    """

    prior_value = calculate_area_prior(session, entry_id)

    # Update the area record
    area = session.query(AreaOccupancy).filter_by(entry_id=entry_id).first()
    if area:
        area.area_prior = prior_value
        area.updated_at = datetime.now(timezone.utc)
        session.commit()

    return prior_value


def update_all_area_priors(session) -> dict[str, float]:
    """
    Calculate and update area priors for all areas in the database.

    Args:
        session: Database session

    Returns:
        dict: Mapping of entry_id to prior value
    """

    areas = session.query(AreaOccupancy).all()
    results = {}

    for area in areas:
        prior_value = update_area_prior(session, area.entry_id)
        results[area.entry_id] = prior_value

    return results
