# CS-4775---Final-Project
Implementing various model-based and deep-learning based models to benchmark single cell RNA sequencing.

## How to run
1) Set up a virtual environment locally:

```
python3 -m venv .venv
```

Activate on Mac: ```source .venv/bin/activate```

Activate on Windows: ```.venv/Scripts/activate```

2) Install requirements:
```
pip install -r requirements.txt
```

3) Download datasets:
```
python3 download_paul15.py
python3 download_pbmc3k.py
```

4) All implementation and visualization files for our methods (SIMLR, DESC, scGNN) are located in their respective folders.

4.1. SIMLR visualization:
```
python3 simlr/SIMLR_visualize.py
```

4.2. DESC visualization:

4.3. scGNN visualization: