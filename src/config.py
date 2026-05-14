"""Config validation for training pipelines.

Thin wrapper around `src.config_schema`. Kept as a separate module because
all training scripts already import `validate_config` from here; rather
than touching every script, the schema work lives in `config_schema` and
this module re-exports.
"""
from src.config_schema import validate_config

__all__ = ["validate_config"]
