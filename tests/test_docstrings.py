import pytest
import sys
import os


sys.path.insert(0, "../acse_la")

from acse_la import gauss, matmul, zeromat

def test_gauss_docstring():
    assert bool(gauss.__doc__) == True

def test_matmul_docstring():
    assert bool(matmul.__doc__) == True

def test_zeromat_docstring():
    assert bool(zeromat.__doc__) == True