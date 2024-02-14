# Conformal Data Cleaning

This repository contains source code for the experiments conducted in the AISTATS 2024 paper `From Data Imputation to Data Cleaning - Automated Cleaning of Tabular Data Improves Downstream Predictive Performance`.


## Run Experiments

First of all, use [`load_corrupt_and_test_datasets.ipynb`](./notebooks/load_corrupt_and_test_datasets.ipynb) to download and corrupt the datasets and setup the expected structure of the [`data`](./data/) directory. 

[`run_experiment.py`](./scripts/run_experiment.py) implements a simple CLI script (`run-experiment`), which allows to easily run experiments.

**Conformal Data Cleaning:**
```bash
run-experiment \
	--task_id \
	"42493" \
	--error_fractions \
	"0.01" \
	"0.05" \
	"0.1" \
	"0.3" \
	"0.5" \
	--num_repetitions \
	"3" \
	--results_path \
	"/conformal-data-cleaning/results/final-experiments" \
	--models_path \
	"/conformal-data-cleaning/models/final-experiments" \
	--how_many_hpo_trials \
	"50" \
	experiment \
	--confidence_level \
	"0.999"
```

**ML Baseline:**
```bash
run-experiment \
	--task_id \
	"42493" \
	--error_fractions \
	"0.01" \
	"0.05" \
	"0.1" \
	"0.3" \
	"0.5" \
	--num_repetitions \
	"3" \
	--results_path \
	"/conformal-data-cleaning/results/final-experiments" \
	--models_path \
	"/conformal-data-cleaning/models/final-experiments" \
	--how_many_hpo_trials \
	"50" \
	baseline \
	--method \
	"AutoGluon" \
	--method_hyperparameter \
	"0.999"
```

**PyOD Baseline (not included in the paper):**
```bash
run-experiment \
	--task_id \
	"42493" \
	--error_fractions \
	"0.01" \
	"0.05" \
	"0.1" \
	"0.3" \
	"0.5" \
	--num_repetitions \
	"3" \
	--results_path \
	"/conformal-data-cleaning/results/final-experiments" \
	--models_path \
	"/conformal-data-cleaning/models/final-experiments" \
	--how_many_hpo_trials \
	"50" \
	baseline \
	--method \
	"PyodECOD" \
	--method_hyperparameter \
	"0.3"
```

For Garf, please use [main.py](./garf/main.py).
```bash
python \
	main.py \
	--task_id \
	"42493" \
	--error_fractions \
	"0.01" \
	"0.05" \
	"0.1" \
	"0.3" \
	"0.5" \
	--num_repetitions \
	"3" \
	--results_path \
	"/conformal-data-cleaning/results/final-experiments" \
	--models_path \
	"/conformal-data-cleaning/models/final-experiments"
```


## Run our Experimental Setup

We ran our experiments on Kubernetes using Helm. Please checkout the [helm charts](./infrastructure/helm/) and change the `image` and `imagePullSecrets` settings in the `values.yaml` files accordingly to your setup.
Therefore, some read-write-many volumes are necessary to store the experiment results. Please checkout the [`infrastructure/k8s`](./infrastructure/k8s/) directory (and don't forget to setup the data directory as describe above).

Using `make docker` builds and pushes the necessary docker images and `make helm-install` uses [`deploy_experiments.py`](./scripts/deploy_experiments.py) to start our experimental setup.


## Evaluation

[`notebooks/evaluation`](./notebooks/evaluation/) contains notebooks we use for evaluating the results and [`5_plotting.ipynb`](./notebooks/evaluation/5_plotting.ipynb) outputs the plots shown in the paper.