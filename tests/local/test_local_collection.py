import numpy as np
import pandas as pd
import pytest
import xarray as xr

try:
    from openeo.local import LocalConnection
except ImportError:
    LocalConnection = None


@pytest.mark.skipif(
    not LocalConnection, reason="environment does not support localprocessing"
)
def test_local_collection_metadata(tmp_path_factory):
    sample_netcdf = create_local_data(tmp_path_factory,2,2,2,'netcdf')
    sample_geotiff = create_local_data(tmp_path_factory,2,2,2,'tiff')
    local_conn = LocalConnection(sample_netcdf.as_posix())
    assert len(local_conn.list_collections()) == 1
    local_conn = LocalConnection([sample_netcdf.as_posix(),sample_geotiff.as_posix()])
    assert len(local_conn.list_collections()) == 2


@pytest.mark.skipif(not LocalConnection, reason="environment does not support localprocessing")
def test_execute_uses_injected_process_registry():
    from openeo_pg_parser_networkx import ProcessRegistry
    from openeo_pg_parser_networkx.process_registry import Process

    sentinel = {"loaded": True}

    def load_stac(*args, **kwargs):
        assert args == ()
        kwargs.pop("named_parameters")
        assert kwargs == {"url": "https://example.test/catalog"}
        return sentinel

    registry = ProcessRegistry()
    registry["load_stac"] = Process(spec={"id": "load_stac"}, implementation=load_stac)
    local_conn = LocalConnection([], process_registry=registry)

    result = local_conn.execute(
        {
            "loadstac1": {
                "process_id": "load_stac",
                "arguments": {"url": "https://example.test/catalog"},
                "result": True,
            }
        }
    )

    assert result is sentinel


@pytest.mark.skipif(not LocalConnection, reason="environment does not support localprocessing")
def test_load_stac_dataset_metadata_keeps_healpix_spatial_dimension():
    def load_stac(*args, **kwargs):
        assert args == ()
        assert kwargs == {"url": "https://example.test/catalog"}
        return xr.Dataset(
            {"ch1": (["t", "healpix_index"], np.ones((1, 2)))},
            coords={
                "t": np.array(["2024-01-01"], dtype="datetime64[ns]"),
                "healpix_index": [1, 2],
            },
        )

    cube = LocalConnection([], load_stac_implementation=load_stac).load_stac("https://example.test/catalog")

    assert [d.name for d in cube.metadata.spatial_dimensions] == ["healpix_index"]
    assert cube.metadata.dimension_names() == ["healpix_index", "t", "bands"]


@pytest.mark.skipif(not LocalConnection, reason="environment does not support localprocessing")
def test_load_local_collection_returns_dataset(tmp_path_factory):
    """load_local_collection returns Dataset for NetCDF files."""
    from openeo.local.processing import load_local_collection

    sample_dir = create_local_data(tmp_path_factory, 2, 2, 2, "netcdf")
    result = load_local_collection(id=str(sample_dir / "sample_data.nc"))
    assert isinstance(result, xr.Dataset)
    assert "temperature" in result.data_vars
    assert "precipitation" in result.data_vars


@pytest.mark.skipif(not LocalConnection, reason="environment does not support localprocessing")
def test_load_local_collection_tiff_returns_dataset(tmp_path_factory):
    """load_local_collection returns Dataset for GeoTIFF files."""
    from openeo.local.processing import load_local_collection

    sample_dir = create_local_data(tmp_path_factory, 2, 2, 2, "tiff")
    result = load_local_collection(id=str(sample_dir / "sample_data.tiff"))
    assert isinstance(result, xr.Dataset)


@pytest.mark.skipif(not LocalConnection, reason="environment does not support localprocessing")
def test_execute_returns_dataset(tmp_path_factory):
    """LocalConnection.execute() returns a Dataset for a load_collection graph."""
    from openeo_pg_parser_networkx import ProcessRegistry
    from openeo_pg_parser_networkx.process_registry import Process

    from openeo.local.processing import load_local_collection

    sample_dir = create_local_data(tmp_path_factory, 2, 2, 2, "netcdf")
    registry = ProcessRegistry()
    registry["load_collection"] = Process(
        spec={"id": "load_collection"}, implementation=load_local_collection
    )
    local_conn = LocalConnection(sample_dir.as_posix(), process_registry=registry)
    flat_graph = {
        "loadcol": {
            "process_id": "load_collection",
            "arguments": {"id": str(sample_dir / "sample_data.nc")},
            "result": True,
        }
    }
    result = local_conn.execute(flat_graph)
    assert isinstance(result, xr.Dataset)
    assert "temperature" in result.data_vars or "precipitation" in result.data_vars


@pytest.mark.skipif(not LocalConnection, reason="environment does not support localprocessing")
def test_execute_scopes_callback_cache_by_dataset_variable():
    """Reducer callbacks must be evaluated independently for each data variable."""
    import inspect
    from functools import wraps

    from openeo_pg_parser_networkx import ProcessRegistry
    from openeo_pg_parser_networkx.pg_schema import ParameterReference
    from openeo_pg_parser_networkx.process_registry import Process

    def process(f):
        @wraps(f)
        def wrapper(
            *args,
            positional_parameters=None,
            named_parameters=None,
            **kwargs,
        ):
            args = list(args)
            positional_parameters = positional_parameters or {}
            named_parameters = dict(named_parameters or {})
            for arg_name, i in positional_parameters.items():
                named_parameters[arg_name] = args[i]

            resolved_args = []
            for arg in args:
                if isinstance(arg, ParameterReference):
                    resolved_args.append(named_parameters[arg.from_parameter])

            resolved_kwargs = {}
            for name, arg in kwargs.items():
                if isinstance(arg, ParameterReference):
                    resolved_kwargs[name] = named_parameters[arg.from_parameter]
                else:
                    resolved_kwargs[name] = arg

            for special_arg in ["axis", "keepdims"]:
                if special_arg not in inspect.signature(f).parameters:
                    resolved_kwargs.pop(special_arg, None)

            return f(*resolved_args, **resolved_kwargs)

        return wrapper

    def load_stac(*args, **kwargs):
        return xr.Dataset(
            {
                "a": (["t", "x"], np.array([[1.0, 2.0], [3.0, 4.0]])),
                "b": (["t", "x"], np.array([[0.0, 1.0], [1.0, 0.0]])),
            },
            coords={
                "t": np.array(["2026-01-01", "2026-01-02"], dtype="datetime64[ns]"),
                "x": [10, 20],
            },
        )

    def reduce_dimension(data, reducer, dimension):
        return data.reduce(
            reducer,
            dim=dimension,
            positional_parameters={"data": 0},
        )

    def mean(data, axis=None, keepdims=False):
        return np.nanmean(data, axis=axis, keepdims=keepdims)

    registry = ProcessRegistry(wrap_funcs=[process])
    registry["load_stac"] = Process(spec={"id": "load_stac"}, implementation=load_stac)
    registry["reduce_dimension"] = Process(
        spec={"id": "reduce_dimension"}, implementation=reduce_dimension
    )
    registry["mean"] = Process(spec={"id": "mean"}, implementation=mean)

    local_conn = LocalConnection([], process_registry=registry)
    result = local_conn.execute(
        {
            "loadstac1": {
                "process_id": "load_stac",
                "arguments": {"url": "https://example.test/catalog"},
            },
            "reducedimension1": {
                "process_id": "reduce_dimension",
                "arguments": {
                    "data": {"from_node": "loadstac1"},
                    "dimension": "t",
                    "reducer": {
                        "process_graph": {
                            "mean1": {
                                "process_id": "mean",
                                "arguments": {"data": {"from_parameter": "data"}},
                                "result": True,
                            }
                        }
                    },
                },
                "result": True,
            },
        }
    ).compute()

    np.testing.assert_allclose(result["a"].values, [2.0, 3.0])
    np.testing.assert_allclose(result["b"].values, [0.5, 0.5])

def create_local_data(tmp_path_factory,lat_size,lon_size,t_size,file_format):
    np.random.seed(0)
    lon = np.linspace(10.5,11.5,lon_size)
    lat = np.linspace(46.0,47.0,lat_size)
    time = pd.date_range('2014-09-06', periods=t_size)
    reference_time = pd.Timestamp('2014-09-05')

    if file_format.lower() in ['nc','netcdf']:
        temperature = 15 + 8 * np.random.randn(lat_size, lon_size, t_size)
        precipitation = 10 * np.random.rand(lat_size, lon_size, t_size)
        ds = xr.Dataset(
            data_vars=dict(
                temperature=(['x', 'y', 'time'], temperature),
                precipitation=(['x', 'y', 'time'], precipitation),
            ),
            coords=dict(
                lon=(['x'], lon),
                lat=(['y'], lat),
                time=time,
                reference_time=reference_time,
            ),
            attrs=dict(description='Weather related data.'),
        )
        d = tmp_path_factory.mktemp('sample_netcdf')
        ds.to_netcdf(d / 'sample_data.nc')
    elif file_format.lower() in ['tif','tiff','geotiff']:
        temperature = 15 + 8 * np.random.randn(lat_size, lon_size)
        precipitation = 10 * np.random.rand(lat_size, lon_size)
        ds = xr.Dataset(
            data_vars=dict(
                temperature=(['x', 'y'], temperature),
                precipitation=(['x', 'y'], precipitation),
            ),
            coords=dict(
                lon=(['x'], lon),
                lat=(['y'], lat),
            ),
            attrs=dict(description='Weather related data.'),
        )
        d = tmp_path_factory.mktemp('sample_geotiff')
        ds.to_array().transpose('variable', 'y', 'x').rio.to_raster(d / 'sample_data.tiff')

    return d
