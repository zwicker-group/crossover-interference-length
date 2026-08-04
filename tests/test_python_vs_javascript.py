"""Compares output of python and javascript code

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np
import pytest
from jsrun import Runtime

from measure_CO_interference import init_parameters, interference_length


def test_Lint_example(js_code):
    """Test interference distance calculation for random data."""
    # run JS extracted from webpage
    with Runtime() as runtime:
        runtime.eval(js_code)
        CO_positions = runtime.eval("positions = parseCSV(EXAMPLE_POSITIONS, 1.)")
        Lint_js = runtime.eval("""
            const observed_distances = getDistances(positions);
            calcInterferenceLength(positions, observed_distances, 1., 1.)
        """)

    # convert CO positions to python format
    max_CO_count = max(len(pos) for pos in CO_positions)
    dists = np.full((len(CO_positions), max_CO_count), np.nan)
    for i, pos in enumerate(CO_positions):
        dists[i, : len(pos)] = pos

    # determine Lint in python
    parameters = init_parameters({})
    Lint_py = interference_length(dists, parameters)
    assert isinstance(Lint_py, float)

    # check that the interference length are the same (within errors)
    assert Lint_js["Lint"] == pytest.approx(Lint_py, rel=1e-2)
