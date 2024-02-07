# ConUDiff: Diffusion Model with Contrastive Pretraining and Uncertain Region Optimization for Segmentation of Left Ventricle from Echocardiography

# Requirements

The project has been trained and tested in Ubuntu 20.04 using python 3.7.
* torch
* torchvision
* tensorboardX
* medpy
* SimpleITK
* PyYAML
* pillow
* colorlog

## Data

The EchoNet-Dynamic dataset is available for research purpose at https://echonet.github.io/dynamic/. And the EchoNet-Pediatric dataset is available for research purpose at https://echonet.github.io/pediatric/.


## Usage

We set the flags as follows:
```
MODEL_FLAGS="--image_size 64 --num_channels 64 --class_cond False --num_res_blocks 1 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16"
DIFFUSION_FLAGS="--diffusion_steps 300 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False"
TRAIN_FLAGS="--lr 1e-4 --batch_size 28"
```
To train the segmentation model, run

```
python scripts/segmentation_train.py --data_dir ./data_path $TRAIN_FLAGS $MODEL_FLAGS $DIFFUSION_FLAGS

```
The model will be saved in the *results/weights/weight* folder.
For sampling an ensemble of 1 segmentation masks with the ConUDiff approach, run:

```
python scripts/segmentation_sample.py  --data_dir ./data_path  --model_path ./results/weights/xxx.pt --num_ensemble=1 $MODEL_FLAGS $DIFFUSION_FLAGS
```


