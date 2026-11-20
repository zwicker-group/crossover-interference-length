"""This file is used to configure the test environment when running py.test.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

PACKAGE_PATH = Path(__file__).resolve().parents[1]
JS_RUNNER = Path(__file__).parent / "javascript" / "run_js.mjs"


@pytest.fixture(autouse=False, name="rng")
def init_random_number_generators():
    """Get a random number generator and set the seed of the random number generator.

    The function returns an instance of :func:`~numpy.random.default_rng()` and
    initializes the default generators of both :mod:`numpy` and :mod:`numba`.
    """
    return np.random.default_rng(0)


def get_js_code():
    """Extract javascript code from HTML file"""
    from bs4 import BeautifulSoup

    # Read HTML content from a file
    path = PACKAGE_PATH / "measure_CO_interference.html"
    with open(path, "r", encoding="utf-8") as file:
        html_cont = file.read()
    if not html_cont:
        msg = f"Could not read HTML file from `{path}`"
        raise RuntimeError(msg)

    # Parse the HTML content
    soup = BeautifulSoup(html_cont, "html.parser")

    lines = []
    for script in soup.find_all("script"):
        # Ignore imported/external JavaScript files.
        if script.get("src") is not None:
            continue

        # `script.string` can be None in some HTML documents, so use get_text().
        code = script.get_text()

        if code.strip():
            lines.extend(code.strip().split("\n"))

    # remove plotly loading from the first lines and the document setup in the last lines
    lines = lines[5:-2]

    if not lines:
        msg = f"Could not read script file content `{soup}`"
        raise RuntimeError(msg)

    # add module exports
    lines.extend(
        [
            "",
            "// ES module exports",
            "export { parseCSV, getDistances, calcInterferenceLength, getDnoInt, arrayStats, EXAMPLE_POSITIONS, EXAMPLE_LENGTH };",
        ]
    )

    return "\n".join(lines)


def pytest_configure(config):
    """Extract the javascript file from HTML and add ES module exports"""
    # write the extracted javascript file
    path = PACKAGE_PATH / "tests" / "javascript" / "extracted_from_html.mjs"
    with path.open("w") as jsfile:
        jsfile.write(get_js_code())


def call_javascript(function_name: str, *args):
    """Call an exported JavaScript function with JSON-compatible arguments."""
    if shutil.which("node") is None:
        pytest.skip("Node.js is required for JavaScript crossover tests")

    request = {
        "function": function_name,
        "args": args,
    }

    result = subprocess.run(
        ["node", str(JS_RUNNER)],
        input=json.dumps(request),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, (
        f"JavaScript function {function_name!r} failed.\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )

    return json.loads(result.stdout)