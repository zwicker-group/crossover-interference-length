"""This file is used to configure the test environment when running py.test.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

PACKAGE_PATH = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=False, name="rng")
def init_random_number_generators():
    """Get a random number generator and set the seed of the random number generator.

    The function returns an instance of :func:`~numpy.random.default_rng()` and
    initializes the default generators of both :mod:`numpy` and :mod:`numba`.
    """
    return np.random.default_rng(0)


@pytest.fixture(scope="session")
def js_code():
    """Extract javascript code from webpage"""
    from bs4 import BeautifulSoup

    # Read HTML content from a file
    with open(
        PACKAGE_PATH / "measure_CO_interference.html", "r", encoding="utf-8"
    ) as file:
        html_cont = file.read()

    # Parse the HTML content
    soup = BeautifulSoup(html_cont, "html.parser")
    script = soup.find_all("script")[1].string

    # remove plotly loading from the first lines and the document setup in the last lines
    return "\n".join(script.split("\n")[6:-3])


@pytest.fixture(scope="session")
def js_file(js_code):
    with tempfile.NamedTemporaryFile(suffix=".js", mode="w") as jsfile:
        jsfile.write(js_code)
        jsfile.flush()
        yield jsfile.name
