"""Local adapter for distributional shift test variants.

Registers a custom env name that accepts is_testing/level_choice kwargs
without modifying upstream ai-safety-gridworlds code.
"""

from ai_safety_gridworlds.environments.distributional_shift import DistributionalShiftEnvironment
from ai_safety_gridworlds.helpers import factory


class DistributionalShiftTestingEnvironment(DistributionalShiftEnvironment):
    def __init__(self, is_testing=False, level_choice=None, **_kwargs):
        super().__init__(is_testing=is_testing, level_choice=level_choice)


def ensure_distributional_shift_test_env_registered():
    if "distributional_shift_test" not in factory._environment_classes:
        factory._environment_classes["distributional_shift_test"] = DistributionalShiftTestingEnvironment
