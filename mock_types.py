"""Mock types for occupancy prediction."""

import random
from enum import StrEnum


class InputType(StrEnum):
    """Input type."""

    MOTION = "motion"
    MEDIA = "media"
    APPLIANCE = "appliance"
    DOOR = "door"
    WINDOW = "window"
    TEMPERATURE = "temperature"
    HUMIDITY = "humidity"
    ILLUMINANCE = "illuminance"
    ENVIRONMENTAL = "environmental"


class Likelihood:
    """Type definition for a likelihood used in occupancy prediction."""

    prob_given_true: float
    prob_given_false: float

    def __init__(self, prob_given_true: float, prob_given_false: float):
        self.prob_given_true = prob_given_true
        self.prob_given_false = prob_given_false


class EntityType:
    """Entity type."""

    input_type: InputType
    weight: float

    def __init__(self, input_type: InputType, weight: float):
        self.input_type = input_type
        self.weight = weight


class Entity:
    """Type definition for a feature used in occupancy prediction."""

    entity_id: str
    entity_type: EntityType
    weight: float
    likelihood: Likelihood
    evidence: bool

    def __init__(
        self,
        entity_id: str,
        entity_type: EntityType,
        weight: float,
        likelihood: Likelihood,
    ):
        self.entity_id = entity_id
        self.entity_type = entity_type
        self.weight = weight
        self.likelihood = likelihood
        self.evidence = random.choice([True, False])
