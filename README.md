# MetaMTReg
Official code for "Improving Few-Shot Learning through Multi-task Representation Learning Theory" ECCV 2022.

![alt text](./images/assumptions.png "Explaining important assumptions.")

## How to run

### Install required packages

You can run `pip install -r requirements.txt` to install required packages or `conda env create -f environment.yml` to create a new environment with the required packages installed.

### Train MAML:

Train MAML with the script `train_maml.sh`. Add arguments `--s_ratio` *and* `--s_norm` to train with the regularization.

### Train ProtoNet:

Train ProtoNet with the script `train_proto.sh`. Add arguments `--norm` to train with normalized prototypes.

### Evaluate:

Evaluate MAML or ProtoNet with the script `eval.sh`.  
For Cross-dataset evaluation, change the argument `--dataset`. However, you will need to download and create your own Dataset class. In the folder `datasets` you can find the code for CropDisease, but you need to download the dataset manually.