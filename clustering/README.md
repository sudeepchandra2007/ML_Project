# Air-quality clustering

This project clusters the 12 Beijing monitoring stations in the PRSA archive by their pollution and weather profiles. The default station-level representation uses mean, median, standard deviation, minimum, and maximum for numeric variables plus wind-direction shares. Features are median-imputed, standardized, reduced with PCA, and evaluated across K-Means, Ward hierarchical clustering, and Gaussian mixture models.

## Run

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Run the station analysis:

```powershell
python cluster_air_quality.py
```

Results are written to `clustering_output/`, including cluster assignments, model comparison metrics, cluster profiles, a plot, and a serialized preprocessing/model pipeline.

For hourly observation clusters instead, run:

```powershell
python cluster_air_quality.py --mode hourly
```

Hourly mode is more computationally expensive and is best used for discovering recurring conditions, while station mode is the appropriate default for comparing monitoring locations.