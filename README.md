# AnisotropicMultiStreamCNN
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Source code of the paper "Anisotropic 3D Multi-Stream CNN for Accurate Prostate Segmentation from Multi-Planar MRI".

![Network](Network.PNG)

## Setup
```
pip install -r requirements.txt
```

## Inference

### Usage
```
Usage: inference.py [-h] {single,dual,triple} model_dir image_dir output

Inference.

positional arguments:
  {single,dual,triple}  Path to the config space definition.
  model_dir             Model directory.
  image_dir             Input image directory.
  output                Output filename

optional arguments:
  -h, --help            show this help message and exit
```

### Example
```
The inputs to the algorithm are the three orthogonal volumes, which should be registered to each other (example data is provided in the ‘data’ directory). Data used in this research was obtained from the ProstateX Challenge [1-3]. The output of the algorithm is the high resolution segmentation

python inference.py dual models/dual data/ProstateX-0176 output.nrrd
```

## Preprocessing for Training 
The preprocessing (cropping, intensity normalization) is done offline. The script generateTrainingData.py runs the preprocessing.
We assume the following data structure for the preprocessing. Deviations from this structure need adaption in the sourcecode:

- imgs
	- Case1
		- tra.nrrd
		- sag.nrrd
		- cor.nrrd
	- Case2
		- tra.nrrd
		- sag.nrrd
		- cor.nrrd
	- Case3
		- ....
		- ...
- GT
	- Case1
		- prostate_smooth_label.nrrd
	- Case2
		- prostate_smooth_label.nrrd
	- ...


According to the available cases in the GT input directory, folds are generated. For each fold, the training files are saved as filenames in a npy array (only strings, no images). 
The validation data is saved in image arrays, as it is not needed for further augmentation and can be simply fed into the training procedure.


### Usage 
```
usage: generateTrainingData.py [-h]
                               input_dir_imgs input_dir_GT output_dir nr_folds

generate training data (preprocessing).

positional arguments:
  input_dir_imgs  Directory with original input images (cor, sag, tra).
  input_dir_GT    Directory with ground truth data.
  output_dir      Output directory for preprocessed images.
  nr_folds        Number of folds to be created.

optional arguments:
  -h, --help      show this help message and exit
```

### Example
```
python generateTrainingData.py /data/data_multiplane/data_original /data/data_multiplane/data_GT/ /data/data_multiplane/ 4

```

### Augmentation
Data augmentation is done online except for elastic deformation, as the processing time for elastic deformation is too high for online augmentation.
Thus, the preprocessed images can be deformed elastically with the script elastic_deformation.py.

```
usage: elastic_deformation.py [-h] [-n NUM_ITERATIONS] [-i INPUT_DIR]
                              [-cp CONTROL_POINTS] [-s SIGMA]

Create additional images by elastic deformation.

optional arguments:
  -h, --help            show this help message and exit
  -n NUM_ITERATIONS, --num-iterations NUM_ITERATIONS
                        How many times each of input datasets should be
                        deformed.
  -i INPUT_DIR, --input-dir INPUT_DIR
                        Input data directory.
  -cp CONTROL_POINTS, --control-points CONTROL_POINTS
  -s SIGMA, --sigma SIGMA
```

## Training

### Usage
```
usage: train.py [-h] [-m {single,dual,triple}]
                [--data-dir DATA_DIR] [-lr LEARNING_RATE] [-bs BATCH_SIZE]
                [-e EPOCHS] [--early-stop] [--lr-decay] 
                data_dir train_list val_tra val_cor val_sag val_GT output

Start training.

positional arguments:
  output                output directory
  train_list            Name of train list (npy array)
  val_tra               Name of tra imgs validation array
  val_cor               Name of cor imgs validation array
  val_sag               Name of sag imgs validation array
  val_GT                Name of GT validation array
  output                output directory


optional arguments:
  -h, --help            show this help message and exit
  -m {single,dual,triple}, --model-type {single,dual,triple} (default triple)
  -lr LEARNING_RATE, --learning-rate LEARNING_RATE
                        learning rate (default 0)
  -bs BATCH_SIZE, --batch-size BATCH_SIZE
                        batch size (default 1)
  -e EPOCHS, --epochs EPOCHS
                        epoch count (default 100)
  --early-stop          use early stop
  --lr-decay            use learning rate decay
```

### Example
```
python train.py '/data/data_multiplane' 'train_fold1.npy' 'fold1_val_imgs_tra.npy' 'fold1_val_imgs_cor.npy' 'fold1_val_imgs_sag.npy' 'fold1_val_GT.npy'  --model-type triple output_dir

```

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

## References
[1] G. Litjens, O. Debats, J. Barentsz, N. Karssemeijer, and H. Huisman. "ProstateX Challenge data", The Cancer Imaging Archive (2017). https://doi.org/10.7937/K9TCIA.2017.MURS5CL

[2] G. Litjens, O. Debats, J. Barentsz, N. Karssemeijer and H. Huisman. "Computer-aided detection of prostate cancer in MRI", IEEE Transactions on Medical Imaging 2014;33:1083-1092.

[3] Clark K, Vendt B, Smith K, Freymann J, Kirby J, Koppel P, Moore S, Phillips S, Maffitt D, Pringle M, Tarbox L, Prior F. The Cancer Imaging Archive (TCIA): Maintaining and Operating a Public Information Repository, Journal of Digital Imaging, Volume 26, Number 6, December, 2013, pp 1045-1057.
