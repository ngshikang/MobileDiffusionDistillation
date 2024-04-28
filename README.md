## Installation
Login to NSCC and submit the following command.

```bash
conda create -n mdd python=3.8
conda activate mdd
git clone https://github.com/ngshikang/MobileDiffusionDistillation.git
cd MobileDiffusionDistillation
pip install -r requirements.txt
```

## Dataset
You may refer to [Nota AI's repo for a preprocessed dataset of LAION](https://github.com/Nota-NetsPresso/BK-SDM?tab=readme-ov-file#single-gpu-training-for-bk-sdm-base-small-tiny), or use your own dataset of image-text pairs.

## Training
Login to NSCC and submit the job using the following command.
```
export HF_HOME="/scratch/.cache/huggingface"
export HF_DATASETS_CACHE="/scratch/.cache/huggingface/datasets"
export TRANSFORMERS_CACHE="/scratch/.cache/huggingface/models"
wandb login
```
Log into your weights and biases (wandb) account to access the live training logs.
Then, submit the job to begin the training.
```
qsub scripts/distill_absolutereality.pbs
```
Retrieve the final model from the *results* folder after training has ended. You may monitor the progress on wandb to check when it has ended.

## License
This project, along with its weights, is subject to the [CreativeML Open RAIL-M license](LICENSE), which aims to mitigate any potential negative effects arising from the use of highly advanced machine learning systems. [A summary of this license](https://huggingface.co/blog/stable_diffusion#license) is as follows.
```
1. You can't use the model to deliberately produce nor share illegal or harmful outputs or content,
2. We claim no rights on the outputs you generate, you are free to use them and are accountable for their use which should not go against the provisions set in the license, and
3. You may re-distribute the weights and use the model commercially and/or as a service. If you do, please be aware you have to include the same use restrictions as the ones in the license and share a copy of the CreativeML OpenRAIL-M to all your users.
```

## Acknowledgments
- [National Supercomputing Centre (NSCC) Singapore](https://www.nscc.sg/) for their GPU resources.
- [CompVis](https://github.com/CompVis/latent-diffusion), [Runway](https://runwayml.com/), [CivitAI, Lykon](https://civitai.com/user/Lykon) and [Stability AI](https://stability.ai/) for the Stable Diffusion models.
- [LAION](https://laion.ai/), [Diffusers](https://github.com/huggingface/diffusers), [PEFT](https://github.com/huggingface/peft), [Core ML Stable Diffusion](https://github.com/apple/ml-stable-diffusion), and [Nota AI](https://github.com/Nota-NetsPresso/BK-SDM) for their contributions.
