"""


"""

__title__ = 'openeo'
__author__ = 'Jeroen Dries'


class BaseOpenEoException(Exception):
    pass


import importlib.metadata
import warnings

from openeo._version import __version__
from openeo.rest.connection import Connection, connect, session
from openeo.rest.datacube import UDF, DataCube
from openeo.rest.graph_building import collection_property
from openeo.rest.job import BatchJob, RESTJob
from openeo.rest.multiresult import MultiResult
from openeo.rest.vectorcube import VectorCube

# Distributions that all provide the same ``openeo`` import package.
_CLIENT_DISTRIBUTIONS = ("openeo", "openeo-python-client-dedl")


def client_version() -> str:
    for dist_name in _CLIENT_DISTRIBUTIONS:
        try:
            return importlib.metadata.version(dist_name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return __version__


def _check_distribution_conflict() -> None:
    """Warn if multiple openEO client distributions are installed.

    The upstream ``openeo`` distribution and the DEDL
    ``openeo-python-client-dedl`` distribution provide the same ``openeo``
    import package, so installing both leads to files being overwritten
    unpredictably and to ambiguous metadata. Install exactly one (uninstall the
    other first), or use separate virtual environments.
    """
    installed = []
    for dist_name in _CLIENT_DISTRIBUTIONS:
        try:
            importlib.metadata.version(dist_name)
            installed.append(dist_name)
        except importlib.metadata.PackageNotFoundError:
            pass
    if len(installed) > 1:
        warnings.warn(
            "Multiple openEO client distributions are installed ({names}); they provide the "
            "same 'openeo' package. Uninstall all but one, e.g. run 'pip uninstall openeo' "
            "before installing 'openeo-python-client-dedl' (or vice versa), or use a separate "
            "virtual environment.".format(names=", ".join(installed)),
            RuntimeWarning,
            stacklevel=2,
        )


_check_distribution_conflict()
