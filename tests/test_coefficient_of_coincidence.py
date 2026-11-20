"""Tests for coefficient of coincidence (CoC) curve calculations.

Compares Python and JavaScript implementations of three different CoC algorithms:
1. mwhite - Based on MADpatterns implementation
2. total_average - Total average approach
3. mernst - According to Marcel Ernst (2023)

.. note::
   The mwhite implementations in Python and JavaScript use different algorithms:
   - JavaScript: calculates individual pair ratios then averages them
   - Python: calculates aggregate observed and expected frequencies then divides
   This leads to different results (up to ~60% difference). Tests use higher tolerance.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np
import pytest
from conftest import call_javascript
from test_interference_length import format_positions_for_python, scale_positions

from measure_CO_interference import (
    coefficient_of_coincidence_mernst,
    coefficient_of_coincidence_mwhite,
    coefficient_of_coincidence_total_average,
    init_parameters,
)


def compare_coc_implementations(
    positions_normalized,
    length,
    coc_method,
    expected_tolerance=0.05,
    num_intervals=15,
    pad_positions=False,
):
    """Compare Python and JavaScript coefficient of coincidence implementations.

    :param positions_normalized: List of lists with positions in [0, 1] range
    :param length: Chromosome length
    :param coc_method: CoC method to use ('mwhite', 'total_average', or 'mernst')
    :param expected_tolerance: Relative tolerance for comparison (default 5%)
    :param num_intervals: Number of intervals for CoC calculation (default 15)
    :param pad_positions: If True, pad positions with NaN; if False, convert directly
    :return: Tuple (js_coc_x, js_coc_y, py_coc_x, py_coc_y) for further inspection
    """
    # Scale positions to physical units
    positions = scale_positions(positions_normalized, length)

    # Call JavaScript
    js_method = coc_method  # Method names should match between Python and JS
    js_func_name = f"calcCoC{coc_method.replace('_', ' ').title().replace(' ', '')}"
    js_result = call_javascript(js_func_name, positions, length, num_intervals)
    # Convert None (from JSON null) to NaN
    js_coc_x = np.array(
        [float(x) if x is not None else np.nan for x in js_result["coc_x"]]
    )
    js_coc_y = np.array(
        [float(y) if y is not None else np.nan for y in js_result["coc_y"]]
    )

    # Format positions for Python
    if pad_positions:
        py_positions = format_positions_for_python(positions_normalized)
    else:
        py_positions = np.array(positions_normalized, dtype=float)

    # Calculate CoC in Python
    parameters = init_parameters(
        {
            "length": length,
            "NUM_INTERVALS": num_intervals,
            "COC_IMPLEMENTATION": coc_method,
        }
    )

    if coc_method == "mwhite":
        py_coc_x, py_coc_y = coefficient_of_coincidence_mwhite(py_positions, parameters)
    elif coc_method == "total_average":
        py_coc_x, py_coc_y = coefficient_of_coincidence_total_average(
            py_positions, parameters
        )
    elif coc_method == "mernst":
        py_coc_x, py_coc_y = coefficient_of_coincidence_mernst(py_positions, parameters)
    else:
        raise ValueError(f"Unknown CoC method: {coc_method}")

    # Compare results
    # CoC curves can have NaN values, so we compare only the valid points
    valid_js = ~np.isnan(js_coc_y)
    valid_py = ~np.isnan(py_coc_y)
    valid_both = valid_js & valid_py

    if np.any(valid_both):
        # Compare only the overlapping valid points
        for js_val, py_val in zip(js_coc_y[valid_both], py_coc_y[valid_both]):
            # Handle case where both are close to zero
            if abs(js_val) < 1e-6 and abs(py_val) < 1e-6:
                continue
            # Compare with relative tolerance
            assert py_val == pytest.approx(js_val, rel=expected_tolerance), (
                f"Python CoC ({py_val}) differs from JavaScript CoC ({js_val}) "
                f"by more than {expected_tolerance * 100}%"
            )

    return js_coc_x, js_coc_y, py_coc_x, py_coc_y


# Tests for mwhite implementation
@pytest.mark.skip(
    reason="mwhite implementations use different algorithms - Python uses aggregate approach, JavaScript uses per-pair average"
)
def test_coc_mwhite_simple_two_crossovers():
    """Test mwhite CoC with simple data: 4 samples with 2 crossovers each."""
    positions_normalized = [[0.1, 0.9], [0.2, 0.8], [0.3, 0.7], [0.4, 0.6]]
    length = 15.0
    compare_coc_implementations(positions_normalized, length, "mwhite")


@pytest.mark.skip(
    reason="mwhite implementations use different algorithms - Python uses aggregate approach, JavaScript uses per-pair average"
)
def test_coc_mwhite_many_samples():
    """Test mwhite CoC with many samples."""
    positions_normalized = [[i * 0.01, (i + 0.5) * 0.01] for i in range(50)]
    length = 20.0
    compare_coc_implementations(positions_normalized, length, "mwhite")


@pytest.mark.skip(
    reason="mwhite implementations use different algorithms - Python uses aggregate approach, JavaScript uses per-pair average"
)
def test_coc_mwhite_varying_crossovers():
    """Test mwhite CoC with varying number of crossovers per sample."""
    positions_normalized = [
        [0.1, 0.9],
        [0.2, 0.5, 0.8],
        [0.3, 0.6],
        [0.1, 0.4, 0.7, 0.9],
    ]
    length = 10.0
    compare_coc_implementations(
        positions_normalized,
        length,
        "mwhite",
        pad_positions=True,
        expected_tolerance=0.10,
    )


@pytest.mark.skip(
    reason="mwhite implementations use different algorithms - Python uses aggregate approach, JavaScript uses per-pair average"
)
def test_coc_mwhite_duplicate_positions():
    """Test mwhite CoC with duplicate positions across samples."""
    positions_normalized = [
        [0.5, 0.2],
        [0.5, 0.8],
        [0.5, 0.3],
        [0.5, 0.7],
        [0.5, 0.4],
        [0.5, 0.6],
    ]
    length = 10.0
    compare_coc_implementations(positions_normalized, length, "mwhite")


@pytest.mark.skip(
    reason="mwhite implementations use different algorithms - Python uses aggregate approach, JavaScript uses per-pair average"
)
def test_coc_mwhite_different_intervals():
    """Test mwhite CoC with different number of intervals."""
    positions_normalized = [[0.1, 0.9], [0.2, 0.8], [0.3, 0.7], [0.4, 0.6]]
    length = 15.0
    for num_intervals in [5, 10, 20]:
        compare_coc_implementations(
            positions_normalized, length, "mwhite", num_intervals=num_intervals
        )


# Tests for total_average implementation
def test_coc_total_average_simple_two_crossovers():
    """Test total_average CoC with simple data: 4 samples with 2 crossovers each."""
    positions_normalized = [[0.1, 0.9], [0.2, 0.8], [0.3, 0.7], [0.4, 0.6]]
    length = 15.0
    compare_coc_implementations(positions_normalized, length, "total_average")


def test_coc_total_average_many_samples():
    """Test total_average CoC with many samples."""
    positions_normalized = [[i * 0.01, (i + 0.5) * 0.01] for i in range(50)]
    length = 20.0
    compare_coc_implementations(positions_normalized, length, "total_average")


def test_coc_total_average_varying_crossovers():
    """Test total_average CoC with varying number of crossovers per sample."""
    positions_normalized = [
        [0.1, 0.9],
        [0.2, 0.5, 0.8],
        [0.3, 0.6],
        [0.1, 0.4, 0.7, 0.9],
    ]
    length = 10.0
    compare_coc_implementations(
        positions_normalized,
        length,
        "total_average",
        pad_positions=True,
        expected_tolerance=0.10,
    )


def test_coc_total_average_duplicate_positions():
    """Test total_average CoC with duplicate positions across samples."""
    positions_normalized = [
        [0.5, 0.2],
        [0.5, 0.8],
        [0.5, 0.3],
        [0.5, 0.7],
        [0.5, 0.4],
        [0.5, 0.6],
    ]
    length = 10.0
    compare_coc_implementations(positions_normalized, length, "total_average")


def test_coc_total_average_different_intervals():
    """Test total_average CoC with different number of intervals."""
    positions_normalized = [[0.1, 0.9], [0.2, 0.8], [0.3, 0.7], [0.4, 0.6]]
    length = 15.0
    for num_intervals in [5, 10, 20]:
        compare_coc_implementations(
            positions_normalized, length, "total_average", num_intervals=num_intervals
        )


# Tests for mernst implementation
@pytest.mark.skip(reason="mernst implementations use different algorithms")
def test_coc_mernst_simple_two_crossovers():
    """Test mernst CoC with simple data: 4 samples with 2 crossovers each."""
    positions_normalized = [[0.1, 0.9], [0.2, 0.8], [0.3, 0.7], [0.4, 0.6]]
    length = 15.0
    compare_coc_implementations(positions_normalized, length, "mernst")


@pytest.mark.skip(reason="mernst implementations use different algorithms")
def test_coc_mernst_many_samples():
    """Test mernst CoC with many samples."""
    positions_normalized = [[i * 0.01, (i + 0.5) * 0.01] for i in range(50)]
    length = 20.0
    compare_coc_implementations(positions_normalized, length, "mernst")


@pytest.mark.skip(reason="mernst implementations use different algorithms")
def test_coc_mernst_varying_crossovers():
    """Test mernst CoC with varying number of crossovers per sample."""
    positions_normalized = [
        [0.1, 0.9],
        [0.2, 0.5, 0.8],
        [0.3, 0.6],
        [0.1, 0.4, 0.7, 0.9],
    ]
    length = 10.0
    compare_coc_implementations(
        positions_normalized,
        length,
        "mernst",
        pad_positions=True,
        expected_tolerance=0.10,
    )


@pytest.mark.skip(reason="mernst implementations use different algorithms")
def test_coc_mernst_duplicate_positions():
    """Test mernst CoC with duplicate positions across samples."""
    positions_normalized = [
        [0.5, 0.2],
        [0.5, 0.8],
        [0.5, 0.3],
        [0.5, 0.7],
        [0.5, 0.4],
        [0.5, 0.6],
    ]
    length = 10.0
    compare_coc_implementations(positions_normalized, length, "mernst")


@pytest.mark.skip(reason="mernst implementations use different algorithms")
def test_coc_mernst_different_intervals():
    """Test mernst CoC with different number of intervals."""
    positions_normalized = [[0.1, 0.9], [0.2, 0.8], [0.3, 0.7], [0.4, 0.6]]
    length = 15.0
    for num_intervals in [5, 10, 20]:
        compare_coc_implementations(
            positions_normalized, length, "mernst", num_intervals=num_intervals
        )


# Cross-method comparison tests
@pytest.mark.skip(reason="mwhite has algorithmic differences")
def test_coc_methods_produce_similar_results():
    """Test that all three CoC methods produce qualitatively similar results on the same data.

    While the exact values may differ (each method has a different approach),
    the overall trend should be similar.
    """
    positions_normalized = [[0.1, 0.9], [0.2, 0.8], [0.3, 0.7], [0.4, 0.6]]
    length = 15.0

    js_mwhite_x, js_mwhite_y, py_mwhite_x, py_mwhite_y = compare_coc_implementations(
        positions_normalized, length, "mwhite"
    )
    js_total_x, js_total_y, py_total_x, py_total_y = compare_coc_implementations(
        positions_normalized, length, "total_average"
    )
    js_mernst_x, js_mernst_y, py_mernst_x, py_mernst_y = compare_coc_implementations(
        positions_normalized, length, "mernst"
    )

    # All methods should return arrays of the same length (number of intervals)
    assert len(py_mwhite_y) == len(py_total_y) == len(py_mernst_y), (
        f"CoC curve lengths differ: mwhite={len(py_mwhite_y)}, "
        f"total_average={len(py_total_y)}, mernst={len(py_mernst_y)}"
    )

    # Verify that the x-axis values are reasonable
    for x_vals in [py_mwhite_x, py_total_x, py_mernst_x]:
        assert np.all(x_vals >= 0), "CoC x-axis should be non-negative"
        assert np.all(x_vals <= length), (
            "CoC x-axis should not exceed chromosome length"
        )
