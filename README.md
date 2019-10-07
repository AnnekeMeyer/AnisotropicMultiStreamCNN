# AnisotropicMultiStreamCNN

Source code of the paper "Anisotropic 3D Multi-Stream CNN for Accurate Prostate Segmentation from Multi-Planar MRI".

## Setup
```
pip install -r requirements.txt
```

## Inference


## Training

## Hyperparameter optimization

The hyperparameter optimization (HPO) uses the HpBandSter package, which combined Hyperband with Bayesian optimization.
The subsections below show how a HPO job can be started in a distributed system with n>=1 workers.

### Preparere configspace file
This python file describes the hyperparameter search space. The file must define a `get_configspace()` function
returning a `ConfigSpace` object. Take a look at `example_configspace.py`.

### Start hpo_server
This script is repsonsible for sampling configurations and scheduling jobs to hpo_workers. The output directory is used
to save trained models (for efficiency, as some of them will be reused in the course of the optimization) and to log
sampled configs. At the end of the optimization a file with the best configuration is written.

```
python hpo_server.py example_configspace.py output --iterations 3 --workers 2 --min-budget=30
--max-budget=270
```

### Start hpo_worker(s)
This script starts a worker that evaluates sampled configs received from the server.
```
python hpo_worker.py output
```