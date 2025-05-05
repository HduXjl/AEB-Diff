# AEB-Diff: An Adaptive Expert Blending Diffusion Framework for Uncertainty-Aware Medical Image Segmentation

## Data

We evaluated our method on the [LIDC dataset](https://www.cancerimagingarchive.net/collection/lidc-idri/).
For our dataloader, the expert annotations as well as the original images need to be stored in the following structure:

```
data
└───training
│   └───0
│       │   image_0.jpg
│       │   label0_.jpg
│       │   label1_.jpg
│       │   label2_.jpg
│       │   label3_.jpg
│   └───1
│       │  ...
└───testing
│   └───3
│       │   image_3.jpg
│       │   label0_.jpg
│       │   label1_.jpg
│       │   label2_.jpg
│       │   label3_.jpg
│   └───4
│       │  ...

```
An example can be seen in folder *data*.

## Usage

We set the flags as follows:

```

MODEL_FLAGS="--image_size 128 --num_channels 128 --class_cond False --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False"
TRAIN_FLAGS="--lr 1e-4 --batch_size 20"

```
To train the ambiguous segmentation model, run

```
python scripts/segmentation_train.py --data_dir /data1/xjlDatasets/LIDC/manifest-1600709154662/data_AMISDM_shang/training $TRAIN_FLAGS $MODEL_FLAGS $DIFFUSION_FLAGS
```
The model will be saved in the *results* folder.
For sampling an ensemble of 4 segmentation masks with the DDPM approach, run:

```
python scripts/segmentation_sample.py --data_dir /data1/xjlDatasets/LIDC/manifest-1600709154662/data_AMISDM/testing  --model_path /data/jupyter/xjl/AMISDM_Beta/results_train/emasavedmodel_0.9999_175000.pt --num_ensemble=4 $MODEL_FLAGS $DIFFUSION_FLAGS
```
The generated segmentation masks will be stored in the *results* folder. A visualization can be done using [Visdom](https://github.com/fossasia/visdom). If you encounter high frequency noise, you can use noise filters such as [median blur](https://www.tutorialspoint.com/opencv/opencv_median_blur.htm) in post-processing step.

## Reference Codes

1. [Ambiguous Medical Image Segmentation using Diffusion Models](https://github.com/aimansnigdha/Ambiguous-Medical-Image-Segmentation-using-Diffusion-Models).
2. [Diffusion Models for Implicit Image Segmentation Ensembles](https://github.com/JuliaWolleb/Diffusion-based-Segmentation).
