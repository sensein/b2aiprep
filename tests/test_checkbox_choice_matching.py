"""Unit tests for matching RedCap checkbox option codes to reproschema choice values."""

import pytest

from b2aiprep.prepare.dataset import BIDSDataset


@pytest.mark.parametrize(
    "choice_value, option_code, expected",
    [
        (1, "1", True),                 # plain integer code
        ("1", "1", True),
        ("past", "past", True),         # word code
        ("age_2_4", "age_2_4", True),   # word code containing digits and underscores
        (20297, "2029_7", True),        # OID-style code stored as int by the conversion
        ("20297", "2029_7", True),
        (2029, "2029_7", False),        # truncated value must not match
        (20297, "20297", True),         # literal match still accepted
        ("age_2_4", "age_24", False),   # word codes are never normalised
        (None, "1", False),
        (1, None, False),
    ],
)
def test_checkbox_choice_matches(choice_value, option_code, expected):
    assert BIDSDataset._checkbox_choice_matches(choice_value, option_code) is expected
