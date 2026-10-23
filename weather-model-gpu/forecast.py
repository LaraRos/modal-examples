"""Run ECMWF's AIFS Single v2 weather model on a GPU with Modal.

Model card and weights   https://huggingface.co/ecmwf/aifs-single-2.0
Reference notebook       run_AIFS_v2.0.ipynb in the same repository

Papers, licence and attribution: see README.md.
"""

import modal
from pathlib import Path
import numpy as np

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.7.0", "torch-geometric==2.6.1")
    .pip_install(
        "anemoi-inference[huggingface]==0.8.3",
        "anemoi-models==0.9.3",
        "anemoi-utils==0.4.35.post3",
    )
    .pip_install(
        "earthkit-data<1.0.0",
        "earthkit-regrid==0.5.1",
        "ecmwf-opendata==0.3.29",
    )
    .pip_install("flash-attn @ https://github.com/cathalobrien/get-flash-attn/releases/download/v0.1-alpha/flash_attn-2.8.3+cu12torch2.7cxx11abiFALSE-cp312-cp312-linux_x86_64.whl", extra_options="--no-deps --force-reinstall")
)

with image.imports():
    import torch
    import flash_attn
    import datetime
    from collections import defaultdict
    import earthkit.data as ekd
    import earthkit.regrid as ekr

    from anemoi.inference.runners.simple import SimpleRunner
    from anemoi.inference.outputs.printer import print_state

    from ecmwf.opendata import Client as OpendataClient
    from huggingface_hub import hf_hub_download

app = modal.App(name="weather-model")

PLOT_DIR = Path(__file__).parent / "out"

# Input parameters, per the AIFS Single v2 model card. These are fixed by the
# checkpoint -- the model takes exactly these fields and nothing else.
PARAM_SFC = ["10u", "10v", "2d", "2t", "msl", "skt", "sp", "tcw", "lsm", "z", "slor", "sdor", "sd"]
PARAM_SOIL = ["vsw", "sot"]
PARAM_WAVE = ["wmb", "h1012", "h1214", "h1417", "h1721", "h2125", "h2530", "mwd", "cdww", "mwp", "swh"]
PARAM_PL = ["gh", "t", "u", "v", "q"]
LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50, 10]
SOIL_LEVELS = [1, 2]
SOURCE = "ecmwf" # aws is also available if ecmwf times out

@app.function(image=image, gpu="A100")
def forecast(lead_time=12):
    ekd.config.set({'cache-policy': 'user'})
    states = []
    DATE, fields = prepare_input()
    checkpoint = {"huggingface":"ecmwf/aifs-single-2.0"}
    runner = SimpleRunner(checkpoint)
    input_state = dict(date=DATE, fields=fields)
    for state in runner.run(input_state=input_state, lead_time=lead_time):
        # The runner yields the same dict every step and mutates it in place, so
        # appending it directly leaves every entry pointing at the final step --
        # you get N references to the last step, not N steps.
        states.append(
            {
                "date": state["date"],
                "latitudes": state["latitudes"],
                "longitudes": state["longitudes"],
                # Copy every field: which ones we want to plot is a decision for
                # the caller, not for this loop. latitudes/longitudes are constant
                # across steps so they can be shared rather than copied.
                "fields": {k: v.copy() for k, v in state["fields"].items()},
            }
        )
        print_state(state)
    return states

def get_open_data(param, date, source, levelist=[], **kwargs):
    fields = defaultdict(list)
    for d in [date - datetime.timedelta(hours=6), date]:
        data = ekd.from_source("ecmwf-open-data", date=d, param=param, levelist=levelist, source=source, **kwargs)
        for f in data: # type: ignore
            # Open data is between -180 and 180, we need to shift it to 0-360
            assert f.to_numpy().shape == (721,1440)
            values = np.roll(f.to_numpy(), -f.shape[1] // 2, axis=1)
            # Interpolate the data to from 0.25 to N320
            values = ekr.interpolate(values, {"grid": (0.25, 0.25)}, {"grid": "N320"})
            # Add the values to the list
            name = f"{f.metadata('param')}_{f.metadata('levelist')}" if levelist else f.metadata("param")
            fields[name].append(values)

    # Collapse the per-timestep lists into a single (2, n_points) array per
    # parameter. Without this the masking below indexes a list and throws.
    for name, values in fields.items():
        fields[name] = np.stack(values)

    return fields

def load_fields(date):
    """Fetch and assemble the input fields the checkpoint expects."""
    fields = {}

    # Surface
    fields.update(get_open_data(param=PARAM_SFC, date=date, source=SOURCE, levtype="sfc"))
    assert all(p in fields for p in PARAM_SFC), "Missing parameters: %s" % (set(PARAM_SFC) - set(fields.keys()))

    # Wave
    fields.update(get_open_data(param=PARAM_WAVE, date=date, source=SOURCE, stream="wave"))
    assert all(p in fields for p in PARAM_WAVE), "Missing parameters: %s" % (set(PARAM_WAVE) - set(fields.keys()))

    # Soil
    soil = get_open_data(param=PARAM_SOIL, date=date, source=SOURCE, levelist=SOIL_LEVELS)
    soil_names = [f"{p}_{lev}" for p in PARAM_SOIL for lev in SOIL_LEVELS]
    assert all(p in soil for p in soil_names), "Missing parameters: %s" % (set(soil_names) - set(soil.keys()))
    # Soil fields enter `fields` only under their model names via `mapping` below.

    # Pressure level
    fields.update(get_open_data(param=PARAM_PL, date=date, source=SOURCE, levelist=LEVELS))
    pressure_names = [f"{p}_{lev}" for p in PARAM_PL for lev in LEVELS]
    assert all(p in fields for p in pressure_names), "Missing parameters: %s" % (set(pressure_names) - set(fields.keys()))

    # Convert mwd to cos and sin
    mwd = fields.pop("mwd")
    mwd_rad = np.deg2rad(mwd)
    fields["cos_mwd"] = np.cos(mwd_rad)
    fields["sin_mwd"] = np.sin(mwd_rad) 

    mapping = {'sot_1': 'stl1', 'sot_2': 'stl2',
           'vsw_1': 'swvl1','vsw_2': 'swvl2'}
    for k,v in soil.items():
        fields[mapping[k]]=v

    fields.pop("q_10", None)  # Remove the 10hPa level for specific humidity, as it is not used in the model
    fields.pop("q_50", None)  # Remove the 50hPa level for specific humidity, as it is not used as a prognostic in the model

    # Land-sea mask ships in the HF model repo, not in our image.
    lsm_path = hf_hub_download("ecmwf/aifs-single-2.0", "lsm.grib")
    mask = np.equal(ekd.from_source("file", lsm_path)[0].to_numpy(flatten=True), 0)

    fields["sd"][:, mask] = np.nan
    fields["swvl1"][:, mask] = np.nan
    fields["swvl2"][:,mask] = np.nan

    # Transform GH to Z
    for level in LEVELS:
        gh = fields.pop(f"gh_{level}")
        fields[f"z_{level}"] = gh * 9.80665

    return fields

def prepare_input():
    DATE = OpendataClient(source=SOURCE).latest()
    print("Initial date is", DATE)

    fields = load_fields(DATE)
    print(f"Loaded {len(fields)} fields")

    return DATE, fields

def fix(lons):
    # Shift the longitudes from 0-360 to -180-180
    return np.where(lons > 180, lons - 360, lons)

def plot_states(states):
    # Imported here, not at module level: these run only on the local machine,
    # and the container image has no matplotlib/cartopy.
    import matplotlib
    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt
    import matplotlib.tri as tri
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    for state in states:

        latitudes = state["latitudes"]
        longitudes = state["longitudes"]
        values = state["fields"]["100u"]
        fig, ax = plt.subplots(figsize=(11, 6), subplot_kw={"projection": ccrs.PlateCarree()})
        ax.coastlines()
        ax.add_feature(cfeature.BORDERS, linestyle=":")

        triangulation = tri.Triangulation(fix(longitudes), latitudes)

        contour=ax.tricontourf(triangulation, values, levels=20, transform=ccrs.PlateCarree(), cmap="RdBu")
        cbar = fig.colorbar(contour, ax=ax, orientation="vertical", shrink=0.7, label="100u")

        plt.title("100m winds (100u) at {}".format(state["date"]))
        stamp = state["date"].strftime("%Y%m%dT%H%M")
        plt.savefig(PLOT_DIR / f"100u_{stamp}.png")
        plt.close()

@app.local_entrypoint()
def main():
    states = forecast.remote(lead_time=24)
    plot_states(states)