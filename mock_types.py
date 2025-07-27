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


class Decay:
    """Decay."""

    decay_factor: float
    is_decaying: bool

    def __init__(self, decay_factor: float, is_decaying: bool):
        self.decay_factor = decay_factor
        self.is_decaying = is_decaying


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
        self.decay = self.get_decay()

    def get_decay(self):
        """Decay the evidence."""
        if self.evidence:
            return Decay(decay_factor=1.0, is_decaying=False)
        decay_factor = random.uniform(0.0, 1.0)
        is_decaying = decay_factor > 0.0
        return Decay(decay_factor=decay_factor, is_decaying=is_decaying)
