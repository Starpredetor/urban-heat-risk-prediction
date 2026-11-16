"""MODIS quality masking and scaling, against hand-computed values (SRS V1)."""

import numpy as np
import pytest

from uhrm.modis import (QcRule, dn_to_celsius, lst_error_class, mandatory_qa,
                        process_composite, quality_mask)

RULE = QcRule()  # defaults: good mandatory QA, error <= 2 K, DN 7500..65535


def test_bit_decoding():
    qc = np.array([0b11000010], dtype=np.uint8)   # error class 3, mandatory QA 2
    assert mandatory_qa(qc)[0] == 2
    assert lst_error_class(qc)[0] == 3


@pytest.mark.parametrize("dn, qc, expected", [
    (15000, 0b00000000, True),     # good, <=1 K
    (15000, 0b01000000, True),     # good, <=2 K (allowed by default)
    (15000, 0b10000000, False),    # error class 2 (<=3 K) exceeds the limit
    (15000, 0b11000000, False),    # error class 3
    (15000, 0b00000001, False),    # produced but "other quality"
    (15000, 0b00000010, False),    # not produced: cloud
    (15000, 0b00000011, False),    # not produced: other
    (0, 0b00000000, False),        # fill / masked pixel reads QC 0 but DN 0
    (7499, 0b00000000, False),     # just below the valid DN range
    (7500, 0b00000000, True),      # lower edge of the valid range
    (65535, 0b00000000, True),     # upper edge
])
def test_quality_mask_truth_table(dn, qc, expected):
    got = quality_mask(np.array([dn], dtype=np.uint16), np.array([qc], dtype=np.uint16), RULE)
    assert bool(got[0]) is expected


def test_error_class_limit_is_configurable():
    strict = QcRule(max_lst_error_class=0)
    dn, qc = np.array([15000], np.uint16), np.array([0b01000000], np.uint16)
    assert not quality_mask(dn, qc, strict)[0]
    assert quality_mask(dn, qc, QcRule(max_lst_error_class=1))[0]


def test_good_qa_requirement_can_be_relaxed():
    dn, qc = np.array([15000], np.uint16), np.array([0b00000001], np.uint16)
    assert not quality_mask(dn, qc, QcRule())[0]
    assert quality_mask(dn, qc, QcRule(require_good_mandatory_qa=False))[0]


def test_scaling_matches_hand_calculation():
    # 15000 * 0.02 = 300 K  ->  300 - 273.15 = 26.85 C
    assert dn_to_celsius(np.array([15000]))[0] == pytest.approx(26.85, abs=1e-4)
    # 15500 * 0.02 = 310 K  ->  36.85 C
    assert dn_to_celsius(np.array([15500]))[0] == pytest.approx(36.85, abs=1e-4)


def test_process_composite_gives_nan_where_invalid_and_never_fills():
    dn = np.array([[15000, 0], [15500, 15000]], dtype=np.uint16)
    qc = np.array([[0, 0], [0, 0b00000010]], dtype=np.uint16)
    out = process_composite(dn, qc, RULE)
    assert out.dtype == np.float32
    assert out[0, 0] == pytest.approx(26.85, abs=1e-4)
    assert out[1, 0] == pytest.approx(36.85, abs=1e-4)
    assert np.isnan(out[0, 1])     # masked pixel
    assert np.isnan(out[1, 1])     # cloud flag


def test_implausible_temperature_is_rejected_even_if_flagged_good():
    # DN 20000 -> 400 K -> 126.85 C: not a surface temperature in India
    out = process_composite(np.array([20000], np.uint16), np.array([0], np.uint16), RULE)
    assert np.isnan(out[0])


def test_rule_validation():
    with pytest.raises(ValueError):
        QcRule(max_lst_error_class=4)
    with pytest.raises(ValueError):
        QcRule(valid_dn_range=(100, 50))
    with pytest.raises(ValueError):
        QcRule(plausible_celsius=(50.0, 10.0))
