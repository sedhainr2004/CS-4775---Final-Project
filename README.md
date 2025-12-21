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
Figures are found in simlr/figures.

4.2. DESC visualization:
```
python3 desc/train_desc.py --config desc/config.yaml
```
Figures are found in desc/outputs. To modify the input data, change the file path in desc/config.yaml (e.g. data/paul15.h5ad --> data/pbmc3k.h5ad)

4.3. scGNN visualization (run pipeline first):

Run the scGNN pipeline to generate artifacts/results (paul15 example):
```
python3 scgnn/preprocess.py --config configs/paul15.yaml
python3 scgnn/build_graph.py --config configs/paul15.yaml
python3 scgnn/split_data.py --config configs/paul15.yaml
python3 scgnn/train_scgnn.py --config configs/scgnn_paul15.yaml
```

Embeddings plot:
```
python3 scgnn/plot_embeddings.py --config configs/scgnn_paul15.yaml --results_dir results/paul15_scgnn_v2 --output_path reports/paul15_umap.png --method umap --labels both
```

Training loss plot:
```
python3 scgnn/plot_training_loss.py --metrics_path results/paul15_scgnn_v2/metrics.json --output_path reports/paul15_loss.png
```

Clustering metrics plot:
```
python3 scgnn/plot_clustering_metrics.py --metrics_path results/paul15_scgnn_v2/metrics.json --output_path reports/paul15_cluster_metrics.png
```

Note: If you `cd scgnn`, drop the `scgnn/` prefix in the script paths.
