# Athlete Lifelogging Pipeline

Main notebook: `athlete_pipeline.ipynb`

This repository contains an end-to-end athlete pipeline for:
- multi-source data loading and merging,
- nutrition estimation from food images,
- feature engineering,
- modeling (injury classification + performance regression),
- dataset export.

## 1) Project Structure

```text
.
├── README.md
├── requirements.txt
├── athlete_pipeline.ipynb
└── src/
    └── athlete_pipeline/
        ├── __init__.py
        ├── config/
        │   └── constants.py
        ├── utils/
        │   ├── io.py
        │   ├── parsing.py
        │   └── imaging.py
        ├── features/
        │   └── engineering.py
        ├── models/
        │   ├── injury.py
        │   └── performance.py
```

## 2) Steps To Execute (Environment Setup)

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
jupyter lab
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
jupyter lab
```

Then open `athlete_pipeline.ipynb` and run cells in order.

## 3) Data Setup

In the notebook, change `DATA_ROOT` to your local dataset path, for example:

```python
DATA_ROOT = Path(r"C:\path\to\your\data")
```

Expected top-level layout inside `DATA_ROOT`:

```text
DATA_ROOT/
├── participant-overview.xlsx
├── p01/
│   ├── fitbit/          (calories.json, steps.json, sleep.json, ...)
│   ├── pmsys/           (injury.csv, srpe.csv, wellness.csv)
│   ├── googledocs/      (reporting.csv)
│   └── food-images/     (*.jpg, *.png, *.heic)
├── p03/
└── p05/
```

## 4) Constants Used And Outputs

### Constants

Defined in `src/athlete_pipeline/config/constants.py`:
- `IMAGE_EXT = {".jpg", ".jpeg"}`
- `CACHE_FILE = Path("food_cache_cv.json")`
- `DEFAULT_PORTION = 200`
- `FALLBACK = (350, 10.0, 40.0, 15.0, 3.0)`
- `NUTRITION`: food -> `(kcal, protein_g, carbs_g, fat_g, fiber_g)` per 100g
- `PORTIONS`: default serving size overrides for selected foods

### Outputs

Expected generated artifacts:
- `test_dataset.csv` (final engineered dataset export)
- `food_cache_cv.json` (image classification cache, if food image step is run)
