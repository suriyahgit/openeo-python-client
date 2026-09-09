import inspect
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Optional

import rioxarray
import xarray as xr
from openeo_pg_parser_networkx import ProcessRegistry
from openeo_pg_parser_networkx.process_registry import Process

_log = logging.getLogger(__name__)


def _register_processes_from_module(
    *,
    process_registry: ProcessRegistry,
    implementations_module,
    specs_module,
) -> None:
    """Register process implementations exported by a process package."""

    processes_from_module = [
        func
        for _, func in inspect.getmembers(
            implementations_module,
            inspect.isfunction,
        )
    ]

    for func in processes_from_module:
        spec = getattr(specs_module, func.__name__, None)
        if spec is None:
            continue
        process_registry[func.__name__] = Process(spec=spec, implementation=func)


def _register_load_collection(
    *, process_registry: ProcessRegistry, specs_module
) -> None:
    process_registry["load_collection"] = Process(
        spec=specs_module.load_collection,
        implementation=load_local_collection,
    )


def init_dask_process_registry() -> ProcessRegistry:
    """
    Build the historic local-processing registry backed by ``openeo_processes_dask``.

    Imports are intentionally lazy so ``openeo.local`` can still be imported in
    environments that provide another process package.
    """
    import openeo_processes_dask.process_implementations
    import openeo_processes_dask.specs
    from openeo_processes_dask.process_implementations.core import process
    from openeo_processes_dask.process_implementations.cubes import load

    process_registry = ProcessRegistry(wrap_funcs=[process])
    _register_processes_from_module(
        process_registry=process_registry,
        implementations_module=openeo_processes_dask.process_implementations,
        specs_module=openeo_processes_dask.specs,
    )

    # Resolve load_stac at call time so monkey-patches on the implementation
    # module are honored.
    if "load_stac" in process_registry:
        process_registry["load_stac"] = Process(
            spec=process_registry["load_stac"].spec,
            implementation=lambda *args, **kwargs: load.load_stac(*args, **kwargs),
        )
    _register_load_collection(
        process_registry=process_registry,
        specs_module=openeo_processes_dask.specs,
    )
    return process_registry


def init_dedl_process_registry(
    *,
    load_stac_implementation: Optional[Callable] = None,
) -> ProcessRegistry:
    """
    Build a local-processing registry backed by DEDL process implementations.

    Raster loading is delegated to ``openeo-processes-dedl-cube-load`` while
    process execution is delegated to ``openeo-processes-dedl-slim``.
    """
    import openeo_processes_dedl_slim.process_implementations
    import openeo_processes_dedl_slim.specs
    from openeo_processes_dedl_slim.process_implementations.core import process

    if load_stac_implementation is None:
        from openeo_processes_dedl_cube_load import load_stac as load_stac_implementation

    process_registry = ProcessRegistry(wrap_funcs=[process])
    _register_processes_from_module(
        process_registry=process_registry,
        implementations_module=openeo_processes_dedl_slim.process_implementations,
        specs_module=openeo_processes_dedl_slim.specs,
    )

    # The package-level import uses ``except ImportError as e`` after importing
    # math.e, which removes the package-level name in Python. Register it from
    # the source module explicitly.
    from openeo_processes_dedl_slim.process_implementations.math import e as _e

    process_registry["e"] = Process(
        spec=openeo_processes_dedl_slim.specs.e,
        implementation=_e,
    )
    process_registry["load_stac"] = Process(
        spec=openeo_processes_dedl_slim.specs.load_stac,
        implementation=lambda *args, **kwargs: load_stac_implementation(*args, **kwargs),
    )
    _register_load_collection(
        process_registry=process_registry,
        specs_module=openeo_processes_dedl_slim.specs,
    )
    return process_registry


def init_process_registry() -> ProcessRegistry:
    """Build the default local-processing registry."""
    return init_dask_process_registry()


_DEFAULT_PROCESS_REGISTRY = None


def get_process_registry() -> ProcessRegistry:
    """
    Get the default local-processing registry.

    ``openeo_processes_dask`` remains the preferred default for backwards
    compatibility. If it is not installed, fall back to DEDL when both DEDL
    packages are available.
    """
    global _DEFAULT_PROCESS_REGISTRY
    if _DEFAULT_PROCESS_REGISTRY is not None:
        return _DEFAULT_PROCESS_REGISTRY

    try:
        _DEFAULT_PROCESS_REGISTRY = init_dask_process_registry()
    except ImportError as dask_error:
        try:
            _DEFAULT_PROCESS_REGISTRY = init_dedl_process_registry()
        except ImportError as dedl_error:
            raise ImportError(
                "Local processing requires either openeo_processes_dask or "
                "openeo-processes-dedl-slim with openeo-processes-dedl-cube-load."
            ) from dedl_error
        _log.info(
            "Using DEDL local process registry because openeo_processes_dask "
            "is not available: %s",
            dask_error,
        )
    return _DEFAULT_PROCESS_REGISTRY


class _LazyProcessRegistry:
    """Compatibility proxy for code importing PROCESS_REGISTRY directly."""

    def _registry(self) -> ProcessRegistry:
        return get_process_registry()

    def __contains__(self, key):
        return key in self._registry()

    def __getitem__(self, key):
        return self._registry()[key]

    def __iter__(self):
        return iter(self._registry())

    def __len__(self):
        return len(self._registry())

    def keys(self):
        return self._registry().keys()

    def items(self):
        return self._registry().items()

    def values(self):
        return self._registry().values()

    def get(self, *args, **kwargs):
        return self._registry().get(*args, **kwargs)


PROCESS_REGISTRY = _LazyProcessRegistry()


def load_local_collection(*args, **kwargs):
    pretty_args = {k: repr(v)[:80] for k, v in kwargs.items()}
    _log.info("Running process load_collection")
    _log.debug(
            f"Running process load_collection with resolved parameters: {pretty_args}"
        )
    collection = Path(kwargs['id'])
    if '.zarr' in collection.suffixes:
        data = xr.open_dataset(kwargs['id'],chunks={},engine='zarr')
    elif '.nc' in collection.suffixes:
        data = xr.open_dataset(kwargs['id'],chunks={},decode_coords='all') # Add decode_coords='all' if the crs as a band gives some issues
        crs = None
        if 'crs' in data.coords:
            if 'spatial_ref' in data.crs.attrs:
                crs = data.crs.attrs['spatial_ref']
            elif 'crs_wkt' in data.crs.attrs:
                crs = data.crs.attrs['crs_wkt']
        if crs is not None:
            for var in data.data_vars:
                data[var].rio.write_crs(crs, inplace=True)
    elif '.tiff' in collection.suffixes or '.tif' in collection.suffixes:
        data = rioxarray.open_rasterio(kwargs['id'],chunks={},band_as_variable=True)
        for d in list(data.data_vars):
            descriptions = [v for k, v in data[d].attrs.items() if k.lower() == "description"]
            if descriptions:
                data = data.rename({d: descriptions[0]})
    return data
