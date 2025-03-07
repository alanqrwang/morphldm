# MorphLDM
MorphLDM is a 3D brain MRI generation method based on state-of-the-art latent diffusion models (LDMs), called MorphLDM, that generates novel images by applying synthesized deformation fields to a learned template.

![arch](figures/arch.png)
![denoising](figures/denoising.png)

## Dependencies
Our code builds directly on [MONAI](https://github.com/Project-MONAI/MONAI/tree/dev) and [GenerativeModels](https://github.com/Project-MONAI/GenerativeModels) repositories.
Make sure they are installed and included in your PYTHONPATH.

## Training on your own data
To train on your own data, edit the `get_data()` function in both `train_autoencoder.py` and `train_diffusion.py` to return your `train_loader` and `val_loader`.
The code expects each mini-batch to be in the form of a dictionary with keys `image`, `age`, and `sex`.
You can edit this to include your own conditions

`config.json` contains the hyperparameters for training the models.
`environment_config.json` contains the paths to the data, output directory, and logging information.

### Train Autoencoder
`python train_autoencoder.py -c config.json -e environment_config.json`

### Train Diffusion UNet
`python train_diffusion.py -c config.json -e environment_config.json`

## Citation
```
@misc{wang2025generatingnovelbrainmorphology,
      title={Generating Novel Brain Morphology by Deforming Learned Templates}, 
      author={Alan Q. Wang and Fangrui Huang and Bailey Trang and Wei Peng and Mohammad Abbasi and Kilian Pohl and Mert Sabuncu and Ehsan Adeli},
      year={2025},
      eprint={2503.03778},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2503.03778}, 
}
```
