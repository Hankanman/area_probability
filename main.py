"""
This module provides a CLI for initializing, populating, and using the occupancy model.
"""

import argparse
import random
import sys

from db import AreaOccupancyStorage
from occupancy_model import (
    compute_entity_likelihoods,
    compute_time_priors,
    generate_entities_from_db,
    naive_bayes_predict,
    update_area_prior,
)


def resolve_area_id(storage: AreaOccupancyStorage, area_identifier: str) -> str:
    """
    Resolve an area identifier (name or entry_id) to an entry_id.

    Args:
        storage: AreaOccupancyStorage instance
        area_identifier: Area name or entry_id (case-insensitive)

    Returns:
        str: The entry_id for the area

    Raises:
        ValueError: If area is not found
    """
    area = storage.get_area_by_identifier(area_identifier)
    if area:
        return area.entry_id

    # If not found, list available areas
    areas = storage.get_all_areas()
    available = [f"{a.area_name} (ID: {a.entry_id})" for a in areas]
    raise ValueError(
        f"Area '{area_identifier}' not found. Available areas:\n" + "\n".join(available)
    )


def cmd_init(_):
    """Initialize the database schema."""
    storage = AreaOccupancyStorage()
    storage.init_db()
    print("Database initialized.")


def cmd_import(args):
    """Import data from SQL file into the database."""
    storage = AreaOccupancyStorage()
    success, message = storage.import_from_sql(args.file)
    print(message)


def cmd_reset(_):
    """Delete all priors and likelihoods from the database."""
    try:
        storage = AreaOccupancyStorage()
        priors_deleted, configs_reset = storage.reset_learned_parameters()
        print(
            f"Deleted {priors_deleted} time priors and reset {configs_reset} entity likelihoods to default values."
        )
    except Exception as e:
        print(f"Error resetting learned parameters: {e}")


def cmd_help(_):
    """Display detailed help information about the CLI commands."""
    help_text = """
Occupancy Model CLI - Help

This CLI provides tools for managing and analyzing area occupancy data using
probabilistic models including Naïve Bayes and Hidden Markov Models.

Available Commands:

1. init
   Initialize the database schema
   Usage: python main.py init

2. import [--file FILE]
   Import data from SQL file into the database
   Usage: python main.py import [--file export.sql]
   Default file: export.sql

3. priors ENTRY_ID [--slot MINUTES]
   Compute time-of-day occupancy priors for a specific area
   Usage: python main.py priors <entry_id> [--slot 60]
   - entry_id: Area entry ID to compute priors for
   - --slot: Time slot size in minutes (default: 60)

4. likelihoods ENTRY_ID
   Compute sensor likelihoods given occupancy for a specific area
   Usage: python main.py likelihoods <entry_id>
   - entry_id: Area entry ID to compute likelihoods for

5. predict ENTRY_ID --entities ENTITY1 ENTITY2... --values VALUE1 VALUE2...
   Perform Naïve Bayes occupancy prediction using current sensor readings
   Usage: python main.py predict <entry_id> --entities sensor1 sensor2 --values 1 0
   - entry_id: Area entry ID
   - --entities: List of sensor entity IDs
   - --values: List of sensor readings (0 or 1)

6. hmm ENTRY_ID --timeline FILE [--slot MINUTES]
   Run HMM-smoothed occupancy analysis over a timeline CSV
   Usage: python main.py hmm <entry_id> --timeline data.csv [--slot 60]
   - entry_id: Area entry ID
   - --timeline: CSV file with format: timestamp,entity_id,value
   - --slot: Time slot size in minutes (default: 60)

7. area-prior ENTRY_ID
   Compute area prior for a specific area
   Usage: python main.py area-prior <entry_id>
   - entry_id: Area entry ID

8. reset
   Delete all time priors and reset entity likelihoods to default values
   Usage: python main.py reset

9. learn-all [--slot MINUTES]
   Compute priors and likelihoods for all areas in the database
   Usage: python main.py learn-all [--slot 60]
   - --slot: Time slot size in minutes for priors (default: 60)

10. help
    Display this help information
    Usage: python main.py help

Examples:

# Initialize database and import data
python main.py init
python main.py import

# Compute priors and likelihoods for an area
python main.py priors living_room --slot 30
python main.py likelihoods living_room

# Make a prediction with current sensor readings
python main.py predict living_room --entities motion_sensor light_sensor --values 1 0

# Analyze occupancy over time using HMM
python main.py hmm living_room --timeline sensor_data.csv --slot 15

# Compute area prior for a specific area
python main.py area-prior living_room

# Learn priors and likelihoods for all areas at once
python main.py learn-all --slot 30

# Reset all learned parameters
python main.py reset

Database Schema:
- area_occupancy: Area configurations
- entities: Sensor entities
- area_entity_config: Sensor configurations per area
- area_time_priors: Time-based occupancy priors
- state_intervals: Historical sensor state data
- metadata: System metadata
"""
    print(help_text)


def cmd_priors(args):
    """Compute time-of-day occupancy priors for a specific area."""
    storage = AreaOccupancyStorage()
    try:
        entry_id = resolve_area_id(storage, args.entry_id)
        storage.delete_area_time_priors(entry_id)
        priors = compute_time_priors(storage, entry_id, slot_minutes=args.slot)
        storage.commit()
        print(f"Populated {len(priors)} time priors for {args.entry_id}.")
    except ValueError as e:
        print(f"Error: {e}")


def cmd_likelihoods(args):
    """Compute sensor likelihoods given occupancy for a specific area."""
    storage = AreaOccupancyStorage()
    try:
        entry_id = resolve_area_id(storage, args.entry_id)
        configs = compute_entity_likelihoods(storage, entry_id)
        storage.commit()
        print(f"Updated likelihoods for {len(configs)} sensors in {args.entry_id}.")
    except ValueError as e:
        print(f"Error: {e}")


def cmd_area_prior(args):
    """Compute area prior for a specific area."""
    storage = AreaOccupancyStorage()
    try:
        entry_id = resolve_area_id(storage, args.entry_id)
        prior_value = update_area_prior(storage, entry_id)
        print(f"Updated area prior for {args.entry_id}: {prior_value:.3f}")
    except ValueError as e:
        print(f"Error: {e}")


def cmd_predict(args):
    """Perform Naïve Bayes occupancy prediction using random historical data."""
    storage = AreaOccupancyStorage()
    try:
        entry_id = resolve_area_id(storage, args.entry_id)

        # Get a random point in time where there is data for this area
        intervals = storage.get_sensor_intervals_for_area(entry_id)
        if not intervals:
            print(f"No sensor data found for area {args.entry_id}")
            return

        random_interval = random.choice(intervals)
        timestamp = random_interval.start_time

        print(f"Using data from {timestamp}")

        # Generate entities from database
        entities = generate_entities_from_db(storage, entry_id)

        # Get area prior and time prior for this timestamp
        area = storage.get_area_by_identifier(entry_id)
        area_prior = area.area_prior if area else 0.5
        time_prior = storage.get_time_prior_at_timestamp(entry_id, timestamp)
        print(f"Area prior: {area_prior:.3f}")
        print(f"Time prior for this time: {time_prior:.3f}")

        # Calculate combined prior
        combined_prior = area_prior * time_prior / 0.5
        combined_prior = max(0.001, min(0.999, combined_prior))
        print(f"Combined prior: {combined_prior:.3f}")

        # Update evidence based on sensor state at the random timestamp
        for entity_id, entity in entities.items():
            # Get sensor state at this timestamp
            sensor_state = storage.get_sensor_state_at_time(entity_id, timestamp)
            entity.evidence = sensor_state == "on"

        # Display entities and evidence used in a table
        print("\nEntities and evidence used:")
        print(
            f"{'Entity ID':<60} {'Evidence':<8} {'Decay':<6} {'P(True)':<8} {'P(False)':<8}"
        )
        print("-" * 100)

        for entity_id, entity in entities.items():
            evidence_str = "ON" if entity.evidence else "OFF"
            decay_str = f"{entity.decay.decay_factor:.3f}"

            # Calculate probabilities for display
            if entity.evidence:
                p_true = entity.likelihood.prob_given_true
                p_false = entity.likelihood.prob_given_false
            else:
                # Apply decay to negative evidence
                neutral_prob = 0.5
                full_negative_t = 1 - entity.likelihood.prob_given_true
                full_negative_f = 1 - entity.likelihood.prob_given_false

                p_true = (
                    neutral_prob
                    + (full_negative_t - neutral_prob) * entity.decay.decay_factor
                )
                p_false = (
                    neutral_prob
                    + (full_negative_f - neutral_prob) * entity.decay.decay_factor
                )

            print(
                f"{entity_id:<60} {evidence_str:<8} {decay_str:<6} {p_true:<8.3f} {p_false:<8.3f}"
            )

        # Make prediction with area prior and time prior
        p = naive_bayes_predict(entities, area_prior, time_prior)
        print(f"\nNaïve Bayes P(occupied) = {p:.3f}")

    except ValueError as e:
        print(f"Error: {e}")


def cmd_learn_all(args):
    """Compute priors and likelihoods for all areas."""
    storage = AreaOccupancyStorage()
    try:
        # Get all areas
        areas = storage.get_all_areas()
        if not areas:
            print("No areas found in the database.")
            return

        print(f"Found {len(areas)} areas. Computing priors and likelihoods...")

        total_priors = 0
        total_configs = 0
        failed_areas = []

        for i, area in enumerate(areas, 1):
            print(
                f"\n[{i}/{len(areas)}] Processing {area.area_name} ({area.entry_id})..."
            )

            try:
                # Compute priors
                storage.delete_area_time_priors(area.entry_id)
                priors = compute_time_priors(
                    storage, area.entry_id, slot_minutes=args.slot
                )

                # Compute likelihoods
                configs = compute_entity_likelihoods(storage, area.entry_id)

                # Compute area prior
                area_prior = update_area_prior(storage, area.entry_id)

                storage.commit()

                non_zero_priors = len([p for p in priors if p.prior_value > 0])
                updated_configs = len(
                    [
                        c
                        for c in configs
                        if c.prob_given_true != 0.5 or c.prob_given_false != 0.5
                    ]
                )

                print(
                    f"  ✓ Generated {len(priors)} priors ({non_zero_priors} non-zero)"
                )
                print(
                    f"  ✓ Updated {len(configs)} sensor likelihoods ({updated_configs} changed from defaults)"
                )
                print(f"  ✓ Updated area prior: {area_prior:.3f}")

                total_priors += len(priors)
                total_configs += len(configs)

            except Exception as e:
                print(f"  ✗ Failed: {e}")
                storage.rollback()

        # Summary
        print("\n=== SUMMARY ===")
        print(
            f"Successfully processed: {len(areas) - len(failed_areas)}/{len(areas)} areas"
        )
        print(f"Total priors generated: {total_priors}")
        print(f"Total sensor configs updated: {total_configs}")

        if failed_areas:
            print("\nFailed areas:")
            for area_name, error in failed_areas:
                print(f"  - {area_name}: {error}")
        else:
            print("All areas processed successfully!")

    except Exception as e:
        print(f"Error: {e}")
        storage.rollback()


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser("Occupancy CLI")
    sub = parser.add_subparsers(required=True)

    p = sub.add_parser("init", help="Initialize the database schema")
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("import", help="Import data from SQL file")
    p.add_argument(
        "--file", default="export.sql", help="SQL file to import (default: export.sql)"
    )
    p.set_defaults(func=cmd_import)

    p = sub.add_parser("priors", help="Compute time-of-day occupancy priors")
    p.add_argument("entry_id", help="Area entry ID")
    p.add_argument("--slot", type=int, default=60, help="Slot size in minutes")
    p.set_defaults(func=cmd_priors)

    p = sub.add_parser("likelihoods", help="Compute sensor likelihoods given occupancy")
    p.add_argument("entry_id", help="Area entry ID")
    p.set_defaults(func=cmd_likelihoods)

    p = sub.add_parser("area-prior", help="Compute area prior for a specific area")
    p.add_argument("entry_id", help="Area entry ID")
    p.set_defaults(func=cmd_area_prior)

    p = sub.add_parser(
        "predict", help="Naïve Bayes occupancy prediction using random historical data"
    )
    p.add_argument("entry_id", help="Area entry ID")
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser(
        "reset",
        help="Delete all time priors and reset entity likelihoods to default values",
    )
    p.set_defaults(func=cmd_reset)

    p = sub.add_parser(
        "learn-all",
        help="Compute priors and likelihoods for all areas in the database",
    )
    p.add_argument(
        "--slot", type=int, default=60, help="Slot size in minutes for priors"
    )
    p.set_defaults(func=cmd_learn_all)

    p = sub.add_parser(
        "help", help="Display detailed help information about the CLI commands"
    )
    p.set_defaults(func=cmd_help)

    try:
        args = parser.parse_args()
        args.func(args)
    except SystemExit:
        # If no arguments provided, show help

        if len(sys.argv) == 1:
            cmd_help(argparse.Namespace())
        else:
            raise


if __name__ == "__main__":
    main()
