# CSA-20-SPCK

## Quick start

1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate    # or `.venv\Scripts\activate` on Windows
pip install -r requirements.txt
```

2. Run the app:

```bash
streamlit run app.py
```

Files:
- `app.py` — main page
- `pages/1_Data_Analysis.py` — data exploration
- `pages/2_Input_Record.py` — append rows to CSV
- `pages/3_Prediction.py` — simple training / evaluation / predict
- `datamodules/DataModule.py` — minimal helper to read/visualize CSV
- `config.py` — path constants