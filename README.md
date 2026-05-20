# Breda-Charlton-FixationGrower

Code and data for the figures in:

> Breda JR\*, Charlton JA\*, Willock JM, Kopec CD, Brody CD (2026).  
> **FixGrower: An efficient and robust curriculum for shaping fixation behavior in rodents.**  
> *eLife* — [bioRxiv preprint](https://www.biorxiv.org/content/10.1101/2025.09.12.675850v1.full.pdf)

---

## Setup

**1. Create and activate the conda environment (Python 3.10)**

```bash
conda env create -f environment.yml
conda activate fixation-grower
```

**2. Register the Jupyter kernel**

```bash
python -m ipykernel install --user --name fixation-grower --display-name "fixation-grower"
```

**3. Data**

Processed data files (`trials_df.parquet`, `days_df.parquet`, `poke_df.parquet`) belong in `data/`.  
This directory is not committed to the repository. See the [figures folder](https://drive.google.com/drive/u/0/folders/1tAd1ds7th0tjRLQgb6MhBt-k-P8ZPf5v) for the associated publication figures.

---

## Running a figure

Open any notebook in `notebooks/` with Jupyter, select the **fixation-grower** kernel, and run all cells top-to-bottom. Figures are saved to `figures/`.

```bash
jupyter notebook notebooks/fig01.ipynb
```

---

## Repository layout

```
Breda-Charlton-FixationGrower/
├── environment.yml                  # conda environment (Python 3.10)
├── pyproject.toml                   # installable package definition
├── data/                            # parquet data files (not committed)
├── figures/                         # figure outputs (not committed)
│   ├── figNN_FULL.png               # main-figure composite (reference; added manually)
│   ├── suppSN_FULL.png              # supplement composite (e.g. suppS3_FULL.png)
│   └── figNN<letter>_….png          # panel PNGs reproduced by notebooks
├── notebooks/
│   ├── fig01.ipynb
│   └── fig02.ipynb
└── src/fixation_grower/
    ├── config.py      # cohort definitions, colors, stage constants
    ├── io.py          # data loaders (load_trials_df, load_poke_df, …)
    ├── transforms.py  # feature engineering (days relative to stage, …)
    ├── plotting.py    # shared aesthetics and save_figure()
    └── stats.py       # Legacy vs FixGrower statistical comparisons
```

---

## Notebook conventions

Each notebook follows a standard structure to maximize readability:

1. **Figure overview (markdown)** — `# Figure N`, embedded `![Full Figure](../figures/figNN_FULL.png)`, then `## Code For …` with goal, panel table, and list of panel outputs the notebook generates
2. **Imports cell** — package imports only, no logic
3. **Shared settings (optional)** — constants reused across panels in the same notebook
4. **Panel sections** — one `## Panel X` markdown header per panel, followed by its code
5. **Cleared outputs** — notebooks are committed without cell outputs

The `figNN_FULL.png` file is the full published figure for context; notebooks reproduce individual panels only.

**Code placement rule:** if logic is reused across figures or supplements it lives in `src/fixation_grower/`; if it is panel-specific constants and styling it stays inline in the notebook. No `def` statements in notebooks unless the function is fewer than 5 lines and used only once.

**Terminology:** curriculum arms are called **Legacy** and **FixGrower** throughout — never V1/V2.

**Data access:** always use the package loaders (`load_trials_df()`, etc.) rather than bare `pd.read_parquet()` calls in notebooks.
