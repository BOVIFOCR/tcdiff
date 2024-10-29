












import sys

from .dependency_versions_table import deps
from .utils.versions import require_version, require_version_core








pkgs_to_check_at_runtime = "python tqdm regex requests packaging filelock numpy tokenizers".split()
if sys.version_info < (3, 7):
    pkgs_to_check_at_runtime.append("dataclasses")
if sys.version_info < (3, 8):
    pkgs_to_check_at_runtime.append("importlib_metadata")

for pkg in pkgs_to_check_at_runtime:
    if pkg in deps:
        if pkg == "tokenizers":

            from .utils import is_tokenizers_available

            if not is_tokenizers_available():
                continue  # not required, check version only if installed

        require_version_core(deps[pkg])
    else:
        raise ValueError(f"can't find {pkg} in {deps.keys()}, check dependency_versions_table.py")


def dep_version_check(pkg, hint=None):
    require_version(deps[pkg], hint)
