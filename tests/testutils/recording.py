#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"record outputs & test them later"
import shelve
from pathlib import Path
import pytest
import pandas as pd
import numpy as np
from numpy.testing import assert_allclose, assert_equal

def pytest_addoption(parser):
    "add an option to the pytest parser"
    parser.addoption(
        "--save-patterns",
        action  = "store_true",
        default = False,
        help    = "store new test data"
    )

def _record(request):
    try:
        from tests.testingcore import utpath
    except ImportError:
        from tests.testutils import utpath

    key     = f'{request.module.__name__}::{request.function.__name__}'
    dostore = request.config.getoption("--save-patterns", False)

    with shelve.open(str(Path(utpath())/"testdata.dump")) as stream:
        class _Store:
            _key: str

            def __getitem__(self, cur: str) -> '_Store':
                assert not hasattr(self, '_key')
                self._key = f"{key}::{cur}"
                return self

            if dostore:

                def __eq__(self, value):
                    stream[self._key] = value
                    del self._key
                    return True
                approx = __eq__
            else:

                def __eq__(self, value):
                    if isinstance(value, np.ndarray):
                        assert_equal(value, stream[self._key])
                    else:
                        assert value == stream[self._key]
                    del self._key
                    return True

                def approx(self, value, rtol = 1e-5, atol = 1e-5):
                    "compare approximatly"
                    kwa = {'rtol': rtol, 'atol': atol}

                    if isinstance(value, pd.DataFrame):
                        truth = stream[self._key]
                        assert truth.shape == value.shape
                        assert set(truth.columns) == set(value.columns)
                        for i, j in truth.items():
                            if any(j.dtype == k for k in ('bool', 'i4', 'i8')):
                                assert_equal(j.values, value[i].values)
                            elif any(j.dtype == k for k in ('f8', 'f4')):
                                assert_allclose(j.values, value[i].values, **kwa)
                            else:
                                assert list(j) == list(value[i])

                    else:
                        assert_allclose(value, stream[self._key], **kwa)
                    del self._key
                    return True

        yield _Store()

@pytest.fixture
def record(request):
    "Stores a test result and compares it to later calls to the test"
    yield from _record(request)
