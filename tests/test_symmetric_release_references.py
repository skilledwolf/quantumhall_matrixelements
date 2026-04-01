from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

pytestmark = pytest.mark.release


_MODULE_PATH = Path(__file__).resolve().parents[1] / "validation" / "symmetric_reference.py"
_SPEC = spec_from_file_location("symmetric_reference", _MODULE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Could not load validation helper from {_MODULE_PATH}")
_MODULE = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_symmetric_release_references_match_current_outputs():
    failures = [item for item in _MODULE.validate_reference_data() if not item["ok"]]
    assert not failures, failures
