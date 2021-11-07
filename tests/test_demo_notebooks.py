'''
Test that demo notebook cells execute without errors
Using https://www.thedataincubator.com/blog/2016/06/09/testing-jupyter-notebooks/
'''
import os
from pathlib import Path
import subprocess
import tempfile

import pytest
import nbformat


# Assumes test are called from project root directory
notebook_dir = Path(os.getcwd()) / 'notebooks'


@pytest.mark.parametrize(
    'notebook_path',
    [
        notebook_dir / 'bayesian-network.ipynb',
        notebook_dir / 'conditional-probability-tables-with-constraints.ipynb'
    ]
)
def test_ipynb(notebook_path):
    nb, errors = _notebook_run(notebook_path)
    assert errors == []


def _notebook_run(path):
    """
    Execute a notebook via nbconvert and collect output.
    """
    with tempfile.NamedTemporaryFile(suffix=".ipynb") as fout:
        args = [
            "python", "-m", "nbconvert", "--to", "notebook", "--execute",
            "--ExecutePreprocessor.timeout=60",
            "--output", fout.name, path
        ]
        subprocess.check_call(args)

        fout.seek(0)
        nb = nbformat.read(fout, nbformat.current_nbformat)

    errors = [
        output for cell in nb.cells if "outputs" in cell
        for output in cell["outputs"] if output.output_type == "error"
    ]

    return nb, errors
