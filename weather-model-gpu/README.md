# AIFS Single v2 on Modal

Runs [ECMWF's AIFS Single v2][model] weather forecasting model on a GPU.
Based on the [inference notebook][notebook] from the model card.

The underlying architecture of AIFS Single v2[^model] is described in Lang et al.[^lang2024], with the v2 changes in Moldovan et al.[^update2025].

Model weights are CC BY 4.0, so attribution is a license condition.

Inference runs through ECMWF's [anemoi-inference][anemoi] framework, and the
initial conditions come from the free [ECMWF open data][opendata] feed.

## Run

You need Python 3.13, [uv](https://docs.astral.sh/uv/) and a [Modal](https://modal.com) account.

```bash
uv sync                        
uv run modal setup             # one-time: log in to Modal
uv run modal run 01-forecast/forecast.py   
```

## Attribution

The model weights, the reference notebook and the open data are all CC BY 4.0
from ECMWF. The maps this example produces are derived from them.

> This service is based on data and products of the European Centre for
> Medium-Range Weather Forecasts (ECMWF).
> Copyright © 2026 European Centre for Medium-Range Weather Forecasts (ECMWF).
> Source: www.ecmwf.int. This data is published under a Creative Commons
> Attribution 4.0 International (CC BY 4.0) licence. ECMWF does not accept any
> liability whatsoever for any error or omission in the data, their
> availability, or for any loss or damage arising from their use.

Modifications: the open data fields are regridded from 0.25° to N320 and used
as initial conditions for AIFS Single v2; the forecast output is plotted.

## Citation

```bibtex
@misc{ecmwf2026aifssingle2,
  title        = {{AIFS} Single v2},
  author       = {{ECMWF}},
  year         = {2026},
  howpublished = {Hugging Face model repository},
  url          = {https://huggingface.co/ecmwf/aifs-single-2.0},
  note         = {Model weights licensed CC BY 4.0}
}

@misc{lang2024aifsecmwfsdatadriven,
      title={AIFS -- ECMWF's data-driven forecasting system}, 
      author={Simon Lang and Mihai Alexe and Matthew Chantry and Jesper Dramsch and Florian Pinault and Baudouin Raoult and Mariana C. A. Clare and Christian Lessig and Michael Maier-Gerber and Linus Magnusson and Zied Ben Bouallègue and Ana Prieto Nemesio and Peter D. Dueben and Andrew Brown and Florian Pappenberger and Florence Rabier},
      year={2024},
      eprint={2406.01465},
      archivePrefix={arXiv},
      primaryClass={physics.ao-ph},
      url={https://arxiv.org/abs/2406.01465}, 
}

@misc{moldovan2025updateecmwfsmachinelearnedweather,
      title={An update to ECMWF's machine-learned weather forecast model AIFS}, 
      author={Gabriel Moldovan and Ewan Pinnington and Ana Prieto Nemesio and Simon Lang and Zied Ben Bouallègue and Jesper Dramsch and Mihai Alexe and Mario Santa Cruz and Sara Hahner and Harrison Cook and Helen Theissen and Mariana Clare and Cathal O'Brien and Jan Polster and Linus Magnusson and Gert Mertes and Florian Pinault and Baudouin Raoult and Patricia de Rosnay and Richard Forbes and Matthew Chantry},
      year={2025},
      eprint={2509.18994},
      archivePrefix={arXiv},
      primaryClass={physics.ao-ph},
      url={https://arxiv.org/abs/2509.18994}, 
}
```

[model]: https://huggingface.co/ecmwf/aifs-single-2.0
[notebook]: https://huggingface.co/ecmwf/aifs-single-2.0/blob/main/run_AIFS_v2.0.ipynb
[anemoi]: https://github.com/ecmwf/anemoi-inference
[opendata]: https://www.ecmwf.int/en/forecasts/datasets/open-data

[^model]: ECMWF. *AIFS Single v2*. Hugging Face model repository, 2026.
    <https://huggingface.co/ecmwf/aifs-single-2.0>. Weights licensed CC BY 4.0.

[^lang2024]: Lang, S., Alexe, M., Chantry, M., Dramsch, J., Pinault, F.,
    Raoult, B., Clare, M. C. A., Lessig, C., Maier-Gerber, M., Magnusson, L.,
    et al. *AIFS — ECMWF's data-driven forecasting system*. arXiv:2406.01465,
    2024. <https://arxiv.org/abs/2406.01465>

[^update2025]: *An update to ECMWF's machine-learned weather forecast model
    AIFS*. arXiv:2509.18994, 2025. <https://arxiv.org/abs/2509.18994>
