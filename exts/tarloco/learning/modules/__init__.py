#  Copyright 2025 University of Manchester, Amr Mousa
#  SPDX-License-Identifier: CC-BY-SA-4.0

"""Definitions for neural-network components for RL-agents."""

from .ac_him import ActorCriticHIM
from .ac_slr import ActorCriticMlpSlr, ActorCriticMlpSlrDblEnc
from .ac_tar import (
    ActorCriticTar,
    ActorCriticTarFt,
    ActorCriticTarRnn,
    ActorCriticTarRnnFt,
)
from .base.ac_base import ActorCriticMlp, ActorCriticRnn, ActorCriticRnnDblEnc
from .utils import mlp_factory
