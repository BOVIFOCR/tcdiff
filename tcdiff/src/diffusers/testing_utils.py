import os
import random
import re
import unittest
from distutils.util import strtobool
from pathlib import Path
from typing import Union

import torch

import PIL.Image
import PIL.ImageOps
import requests
from packaging import version


global_rng = random.Random()
torch_device = "cuda" if torch.cuda.is_available() else "cpu"
is_torch_higher_equal_than_1_12 = version.parse(version.parse(torch.__version__).base_version) >= version.parse("1.12")

if is_torch_higher_equal_than_1_12:
    torch_device = "mps" if torch.backends.mps.is_available() else torch_device


def parse_flag_from_env(key, default=False):
    try:
        value = os.environ[key]
    except KeyError:

        _value = default
    else:

        try:
            _value = strtobool(value)
        except ValueError:

            raise ValueError(f"If set, {key} must be yes or no.")
    return _value


_run_slow_tests = parse_flag_from_env("RUN_SLOW", default=False)


def floats_tensor(shape, scale=1.0, rng=None, name=None):
    """Creates a random float32 tensor"""
    if rng is None:
        rng = global_rng

    total_dims = 1
    for dim in shape:
        total_dims *= dim

    values = []
    for _ in range(total_dims):
        values.append(rng.random() * scale)

    return torch.tensor(data=values, dtype=torch.float).view(shape).contiguous()


def slow(test_case):
    """
    Decorator marking a test as slow.

    Slow tests are skipped by default. Set the RUN_SLOW environment variable to a truthy value to run them.

    """
    return unittest.skipUnless(_run_slow_tests, "test is slow")(test_case)


def load_image(image: Union[str, PIL.Image.Image]) -> PIL.Image.Image:
    """
    Args:
    Loads `image` to a PIL Image.
        image (`str` or `PIL.Image.Image`):
            The image to convert to the PIL Image format.
    Returns:
        `PIL.Image.Image`: A PIL Image.
    """
    if isinstance(image, str):
        if image.startswith("http://") or image.startswith("https://"):
            image = PIL.Image.open(requests.get(image, stream=True).raw)
        elif os.path.isfile(image):
            image = PIL.Image.open(image)
        else:
            raise ValueError(
                f"Incorrect path or url, URLs must start with `http://` or `https://`, and {image} is not a valid path"
            )
    elif isinstance(image, PIL.Image.Image):
        image = image
    else:
        raise ValueError(
            "Incorrect format used for image. Should be an url linking to an image, a local path, or a PIL image."
        )
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image





pytest_opt_registered = {}


def pytest_addoption_shared(parser):
    """
    This function is to be called from `conftest.py` via `pytest_addoption` wrapper that has to be defined there.

    It allows loading both `conftest.py` files at once without causing a failure due to adding the same `pytest`
    option.

    """
    option = "--make-reports"
    if option not in pytest_opt_registered:
        parser.addoption(
            option,
            action="store",
            default=False,
            help="generate report files. The value of this option is used as a prefix to report names",
        )
        pytest_opt_registered[option] = 1


def pytest_terminal_summary_main(tr, id):
    """
    Generate multiple reports at the end of test suite run - each report goes into a dedicated file in the current
    directory. The report files are prefixed with the test suite name.

    This function emulates --duration and -rA pytest arguments.

    This function is to be called from `conftest.py` via `pytest_terminal_summary` wrapper that has to be defined
    there.

    Args:
    - tr: `terminalreporter` passed from `conftest.py`
    - id: unique id like `tests` or `examples` that will be incorporated into the final reports filenames - this is
      needed as some jobs have multiple runs of pytest, so we can't have them overwrite each other.

    NB: this functions taps into a private _pytest API and while unlikely, it could break should
    pytest do internal changes - also it calls default internal methods of terminalreporter which
    can be hijacked by various `pytest-` plugins and interfere.

    """
    from _pytest.config import create_terminal_writer

    if not len(id):
        id = "tests"

    config = tr.config
    orig_writer = config.get_terminal_writer()
    orig_tbstyle = config.option.tbstyle
    orig_reportchars = tr.reportchars

    dir = "reports"
    Path(dir).mkdir(parents=True, exist_ok=True)
    report_files = {
        k: f"{dir}/{id}_{k}.txt"
        for k in [
            "durations",
            "errors",
            "failures_long",
            "failures_short",
            "failures_line",
            "passes",
            "stats",
            "summary_short",
            "warnings",
        ]
    }




    dlist = []
    for replist in tr.stats.values():
        for rep in replist:
            if hasattr(rep, "duration"):
                dlist.append(rep)
    if dlist:
        dlist.sort(key=lambda x: x.duration, reverse=True)
        with open(report_files["durations"], "w") as f:
            durations_min = 0.05  # sec
            f.write("slowest durations\n")
            for i, rep in enumerate(dlist):
                if rep.duration < durations_min:
                    f.write(f"{len(dlist)-i} durations < {durations_min} secs were omitted")
                    break
                f.write(f"{rep.duration:02.2f}s {rep.when:<8} {rep.nodeid}\n")

    def summary_failures_short(tr):

        reports = tr.getreports("failed")
        if not reports:
            return
        tr.write_sep("=", "FAILURES SHORT STACK")
        for rep in reports:
            msg = tr._getfailureheadline(rep)
            tr.write_sep("_", msg, red=True, bold=True)

            longrepr = re.sub(r".*_ _ _ (_ ){10,}_ _ ", "", rep.longreprtext, 0, re.M | re.S)
            tr._tw.line(longrepr)








    config.option.tbstyle = "auto"  # full tb
    with open(report_files["failures_long"], "w") as f:
        tr._tw = create_terminal_writer(config, f)
        tr.summary_failures()


    with open(report_files["failures_short"], "w") as f:
        tr._tw = create_terminal_writer(config, f)
        summary_failures_short(tr)

    config.option.tbstyle = "line"  # one line per error
    with open(report_files["failures_line"], "w") as f:
        tr._tw = create_terminal_writer(config, f)
        tr.summary_failures()

    with open(report_files["errors"], "w") as f:
        tr._tw = create_terminal_writer(config, f)
        tr.summary_errors()

    with open(report_files["warnings"], "w") as f:
        tr._tw = create_terminal_writer(config, f)
        tr.summary_warnings()  # normal warnings
        tr.summary_warnings()  # final warnings

    tr.reportchars = "wPpsxXEf"  # emulate -rA (used in summary_passes() and short_test_summary())
    with open(report_files["passes"], "w") as f:
        tr._tw = create_terminal_writer(config, f)
        tr.summary_passes()

    with open(report_files["summary_short"], "w") as f:
        tr._tw = create_terminal_writer(config, f)
        tr.short_test_summary()

    with open(report_files["stats"], "w") as f:
        tr._tw = create_terminal_writer(config, f)
        tr.summary_stats()


    tr._tw = orig_writer
    tr.reportchars = orig_reportchars
    config.option.tbstyle = orig_tbstyle
