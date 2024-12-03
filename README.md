# ViTs-BrainMet-Failure


This software is licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) license for non-commercial use. For commercial use, please contact Jan Peeken, Technical University of Munich at [jan.peeken@tum.de](mailto:jan.peeken@tum.de) to obtain a commercial license.



## Usage

Required packages and versions are listed in `requirements.txt`

We advise you to build PyTorch with CUDA 12.1 yourself from [pytorch.org](https://pytorch.org/get-started/previous-versions/). 

The dataset structure should be like:

```
parent_dir/
├── train_test_features.csv
├── Training_Set
│   ├── <center 1>
│   │   ├── <patient 1>
│   │   ├── <patient 2>
│   │   |   ├── *_t1c.nii.gz
│   │   |   ├── *_fla.nii.gz
│   │   |   ├── *_label.nii.gz
│   ├── <center 2>
├── Test_Set
│   ├── <center 1>
│   │   ├── <patient 1>
│   │   ├── <patient 2>
│   │   |   ├── 
│   ├── <center 2>

```

`train_test_features.csv` should contain the ground truth and preprocessed (normalized, dummy-encoded) tabular features. Patients should be named like `<patient>_<center>` in the **ID** column.


Please have a look at `config/base_survival.yaml` and `config/vit.yaml` to see how to modify config files. Then the code is run simply by the command:

```
python main.py -cn vit.yaml
```

Training and testing are governed by the same script, just adjust `general.train` and `general.test` flags in the configs. If both are true, first a complete training will be executed and then the checkpoints will be used for testing.

We are using hydra, so configs can be overwritten on the command line like:

```
python main.py -cn vit.yaml general.train=False
```