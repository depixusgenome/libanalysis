#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" access to files """
from    importlib import import_module
from    pathlib   import Path
from    typing    import Union, Sequence, Dict, Any
import  os
import  sys
import  warnings
from    pytest     import mark, fixture, param
import  numpy as np

NO_DISPLAY      = not (sys.platform.startswith("win") or 'DISPLAY' in os.environ)
integrationmark = mark.integration  # pylint: disable=invalid-name
needsdisplay    = mark.needsdisplay # pylint: disable=invalid-name

@fixture(params = [param("", marks = needsdisplay)])
def bokehaction(monkeypatch):
    """
    Create a BokehAction fixture.
    BokehAction.view is the created view. Any of its protected attribute can
    be accessed directly, for example BokehAction.view._ctrl  can be accessed
    through BokehAction.ctrl.
    """
    with import_module("tests.testingcore.bokehtesting").BokehAction(monkeypatch) as act:
        yield act

warnings.filterwarnings('error', category = FutureWarning)
warnings.filterwarnings('error', category = DeprecationWarning)
warnings.filterwarnings('error', category = PendingDeprecationWarning)
warnings.filterwarnings('ignore', category = DeprecationWarning,
                        message  = '.*generator .* raised StopIteration.*')
warnings.filterwarnings('ignore', category = DeprecationWarning,
                        message  = '.*the imp module is deprecated.*')
warnings.filterwarnings('ignore', category = DeprecationWarning,
                        message  = ".*Using or importing the ABCs from 'collections'.*")
np.seterr(all='raise')

class ResourcePath:
    "get resources"
    def __init__(self, root = None, kwa = None):
        self.paths: Dict[str, Any] = {} if kwa is None else kwa
        if root is None:
            path = Path(__file__).parent.parent.parent
            if path.stem == 'build':
                path = path.parent
            self.root = str(path/"data")+"/"
        else:
            self.root = str(root)

    def __call__(self, name: Union[None, Sequence[str], str] = "") -> Union[str, Sequence[str]]:
        "returns the path to the data"
        if isinstance(name, (tuple, list)):
            return tuple(self(i) for i in name) # type: ignore
        directory = Path(self.root)
        if name is None:
            return str(directory)

        default = self.paths.get(str(name).lower().strip(), name)
        if callable(default):
            return default()

        def _test(i):
            val = directory/i
            if not val.exists():
                val = Path(i)
                if not val.exists():
                    raise KeyError("Check your file name!!! {}".format(val))
            return str(val.resolve())

        return (tuple(_test(i) for i in default) if isinstance(default, tuple) else
                _test(default))

def getmonkey():
    "for calling with pudb"
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category = DeprecationWarning)
        warnings.filterwarnings('ignore', category = PendingDeprecationWarning)
        import  pytest  # pylint: disable=unused-import,unused-variable
        from    _pytest.monkeypatch import MonkeyPatch
        warnings.warn("Unsafe call to MonkeyPatch. Use only for manual debugging")
        return MonkeyPatch()

class DummyPool:
    "DummyPool"
    nworkers = 2
    @staticmethod
    def map(*args):
        "DummyPool"
        return map(*args)
