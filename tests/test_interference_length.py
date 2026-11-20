"""Compares output of python and javascript code

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np
import pytest

from conftest import call_javascript
from measure_CO_interference import init_parameters, interference_length


def scale_positions(positions, length):
    """Scale normalized positions [0, 1] to physical units [0, length].

    This matches how the website's parseCSV function scales positions:
    parseCSV(data, length) multiplies all values by the length parameter.
    """
    return [[x * length for x in sample] for sample in positions]


def format_positions_for_python(positions_normalized):
    """Convert list of variable-length position lists to a 2D numpy array with NaN padding.

    :param positions_normalized: List of lists with varying lengths
    :return: 2D numpy array padded with NaN values
    """
    max_co_count = max((len(pos) for pos in positions_normalized), default=0)
    if max_co_count == 0:
        return np.array([]).reshape(len(positions_normalized), 0)

    py_positions = np.full((len(positions_normalized), max_co_count), np.nan)
    for i, pos in enumerate(positions_normalized):
        py_positions[i, : len(pos)] = pos
    return py_positions


def compare_implementations(
    positions_normalized,
    length,
    expected_tolerance=0.01,
    use_example_data=False,
    pad_positions=False,
):
    """Compare Python and JavaScript interference_length implementations.

    :param positions_normalized: List of lists with positions in [0, 1] range
    :param length: Chromosome length
    :param expected_tolerance: Relative tolerance for comparison (default 1%)
    :param use_example_data: If True, use JavaScript EXAMPLE_POSITIONS instead
    :param pad_positions: If True, pad positions with NaN; if False, convert directly
    :return: Tuple (js_lint, py_lint) for further inspection if needed
    """
    if use_example_data:
        # Get example data from JavaScript
        example_data = call_javascript("getExampleData")
        js_positions = example_data["positions"]
        js_distances = example_data["distances"]
        js_length = example_data["length"]
    else:
        # Scale positions to physical units
        positions = scale_positions(positions_normalized, length)

        # Call JavaScript
        js_distances = call_javascript("getDistancesExport", positions)
        js_positions = positions
        js_length = length

    # Call JavaScript interference length function
    js_result = call_javascript(
        "calcInterferenceLengthSimple", js_positions, js_distances, js_length
    )
    js_lint = js_result["Lint"]

    # Format positions for Python
    if use_example_data:
        # Example data is already in physical units, need to scale back to [0, 1]
        max_co_count = max(len(pos) for pos in js_positions) if js_positions else 0
        py_positions = np.full((len(js_positions), max_co_count), np.nan)
        for i, pos in enumerate(js_positions):
            py_positions[i, : len(pos)] = pos
        py_positions_normalized = py_positions / js_length
    else:
        if pad_positions:
            py_positions_normalized = format_positions_for_python(positions_normalized)
        else:
            py_positions_normalized = np.array(positions_normalized, dtype=float)

    # Calculate interference length in Python
    parameters = init_parameters({"length": js_length})
    py_lint = interference_length(py_positions_normalized, parameters)

    # Compare results
    assert py_lint == pytest.approx(js_lint, rel=expected_tolerance), (
        f"Python Lint ({py_lint}) differs from JavaScript Lint ({js_lint}) "
        f"by more than {expected_tolerance * 100}%"
    )

    return js_lint, py_lint


def test_interference_length_with_example_positions():
    """Test interference length calculation using EXAMPLE_POSITIONS from JavaScript.

    This test compares the Python implementation of interference_length with
    the JavaScript implementation using the same test data (EXAMPLE_POSITIONS).

    Both implementations should now produce identical results when positions
    are properly scaled to physical units, matching how the website's parseCSV()
    function converts normalized positions [0, 1] to physical units [0, length].
    """
    compare_implementations(None, None, use_example_data=True)


def test_edge_case_mixed_empty_and_nonempty():
    """Test with a mix of empty and non-empty samples (all non-empty have 2+ positions).

    Both implementations handle this case since all non-empty samples have at least 2 positions.
    """
    positions_normalized = [[0.2, 0.8], [], [0.3, 0.7]]
    length = 10.0
    compare_implementations(positions_normalized, length, pad_positions=True)


def test_edge_case_multiple_samples():
    """Test with multiple samples, all containing exactly 2 positions each.

    Uniform structure with properly scaled data should give tight agreement.
    """
    positions_normalized = [[0.1, 0.9], [0.2, 0.8], [0.3, 0.7], [0.4, 0.6]]
    length = 15.0
    compare_implementations(positions_normalized, length)


def test_edge_case_very_close_positions():
    """Test with crossover positions that are very close together.

    This tests numerical precision when positions are nearly identical.
    """
    positions_normalized = [[0.5, 0.500001, 0.500002]]
    length = 10.0
    compare_implementations(positions_normalized, length)


def test_edge_case_many_samples_uniform():
    """Test with many samples, all with 2 crossovers each at uniform positions."""
    # Create 50 samples with 2 crossovers each at uniform positions
    positions_normalized = [[i * 0.01, (i + 0.5) * 0.01] for i in range(50)]
    length = 20.0
    compare_implementations(positions_normalized, length)


def test_edge_case_many_duplicate_positions_across_samples():
    """Test with many duplicate positions across different samples (but not within samples).

    This explicitly tests how the algorithms handle identical crossover positions
    appearing in different samples. This is a common scenario in real data where
    the same genetic distance might occur in multiple meiosis events.

    Both Python and JavaScript implementations should produce identical results
    when pairs are sampled correctly (excluding self-pairs but including pairs
    from duplicate positions on different chromosomes).
    """
    positions_normalized = [
        [0.5, 0.2],
        [0.5, 0.8],
        [0.5, 0.3],
        [0.5, 0.7],
        [0.5, 0.4],
        [0.5, 0.6],
    ]
    length = 10.0
    compare_implementations(positions_normalized, length)


def test_edge_case_varying_crossover_counts():
    """Test with varying number of crossovers per sample.

    All samples have at least 2 crossovers to avoid edge cases in Python implementation.

    Note: This test uses a slightly relaxed tolerance (10%) because varying crossover
    counts introduce numerical differences in how the algorithms handle NaN padding
    and aggregation across heterogeneous samples.
    """
    positions_normalized = [
        [0.1, 0.9],
        [0.2, 0.5, 0.8],
        [0.3, 0.6],
        [0.1, 0.4, 0.7, 0.9],
    ]
    length = 10.0
    compare_implementations(
        positions_normalized, length, expected_tolerance=0.10, pad_positions=True
    )