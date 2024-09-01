
<p align="center">
  This is the official implementation of the paper
</p>
<p align="center" style="font-size: 24px; font-weight: bold;">
Score Distillation via Reparametrized DDIM
</p>



<p align="center">
  <a href="https://arxiv.org/abs/2405.15891">
    <img src="https://img.shields.io/badge/arXiv-2405.15891-b31b1b.svg?logo=arXiv">
  </a>
  <a href="https://lukoianov.com/sdi">
    <img src="https://img.shields.io/badge/SDI-Project%20Page-b78601.svg">
  </a>
</p>

<p align="center">
  <img alt="sample generation" src="https://lukoianov.com/static/media/a_photograph_of_a_ninja.279adaff.gif" width="50%">
<!-- <img alt="sample generation" src="https://lukoianov.com/static/media/A_DSLR_photo_of_a_freshly_baked_round_loaf_of_sourdough_bread.8bfaaad1.gif" width="70%">
<br/> -->
</p>


<p align="center">
    <a class="active text-decoration-none" href="https://lukoianov.com">Artem Lukoianov</a><sup> 1</sup>,  &nbsp;
    <a class="active text-decoration-none" href="https://scholar.google.com/citations?user=aP0OakUAAAAJ&amp;hl=en">Haitz Sáez de Ocáriz Borde</a><sup> 2</sup>, &nbsp;
    <a class="active text-decoration-none" href="https://kgreenewald.github.io">Kristjan Greenewald</a><sup> 3</sup>, &nbsp;
    <a class="active text-decoration-none" href="https://scholar.google.com.br/citations?user=ow3r9ogAAAAJ&amp;hl=en">Vitor Campagnolo Guizilini</a><sup> 4</sup>, &nbsp;
    <a class="active text-decoration-none" href="https://scholar.google.ch/citations?user=oLi7xJ0AAAAJ&amp;hl=en">Timur Bagautdinov</a><sup> 5</sup>, &nbsp;
    <a class="active text-decoration-none" href="https://www.vincentsitzmann.com">Vincent Sitzmann</a><sup> 1</sup>, &nbsp;
    <a class="active text-decoration-none" href="https://people.csail.mit.edu/jsolomon/">Justin Solomon</a><sup> 1</sup>
</p>
<p align="center">
  <span class="author-block"><sup>1 </sup>Massachusetts Institute of Technology,</span>&nbsp;
  <span class="author-block"><sup>2 </sup>University of Oxford,</span>&nbsp;
  <span class="author-block"><sup>3 </sup>MIT-IBM Watson AI Lab, IBM Research,</span>&nbsp;
  <span class="author-block"><sup>4 </sup>Toyota Research Institute,</span>&nbsp;
  <span class="author-block"><sup>5 </sup>Meta Reality Labs Research</span>
</p>


<p align="center">
  For any questions please shoot an email to <a href="mailto:arteml@mit.edu">arteml@mit.edu</a>
</p>

## Prerequisites
For this project we recommend using a UNIX server with CUDA support and a GPU with at least 40GB of VRAM.
In the case if the amount of available VRAM is limited, we recommend reducing the rendering resolution by adding the following argument to the running command:

```sh
data.width=128 data.height=128
```

Please note that this will reduce the quality of the generated shapes.

## Installation

This project is based on [Threestudio](https://github.com/threestudio-project/threestudio).
Below is an example of the installation used by the authors for Ubuntu 22.04 and CUDA 12.3:

```sh
conda create -n threestudio-sdi python=3.9
conda activate threestudio-sdi

# Consult https://pytorch.org/get-started/locally/ for the latest PyTorch installation instructions
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

pip install ninja
pip install -r requirements.txt
```

<span style="color: orange;">⚠️ Newer versions of diffusers can break the generation, please make sure you are using `diffusers==0.19.3`.</span>


For additional options please address the official installation instructions of Threestudio [here](https://github.com/threestudio-project/threestudio?tab=readme-ov-file#installation) to install threestudio.

## Running generation
The proccess of generating a shape is similar to the one described in the [threestudio](https://github.com/threestudio-project/threestudio?tab=readme-ov-file#quickstart) documentation.
Make sure you are using the SDI config file, like below.
Here are a few examples with different prompts:

```sh
python launch.py --config configs/sdi.yaml --train --gpu 0 system.prompt_processor.prompt="pumpkin head zombie, skinny, highly detailed, photorealistic"

python launch.py --config configs/sdi.yaml --train --gpu 1 system.prompt_processor.prompt="a photograph of a ninja"

python launch.py --config configs/sdi.yaml --train --gpu 2 system.prompt_processor.prompt="a zoomed out DSLR photo of a hamburger"

python launch.py --config configs/sdi.yaml --train --gpu 3 system.prompt_processor.prompt="bagel filled with cream cheese and lox"
```

The results will be saved to `outputs/score-distillation-via-inversion/`.

### Export Meshes

To export the scene to texture meshes, use the `--export` option. Threestudio currently supports exporting to obj+mtl, or obj with vertex colors:

```sh
# this uses default mesh-exporter configurations which exports obj+mtl
python launch.py --config path/to/trial/dir/configs/parsed.yaml --export --gpu 0 resume=path/to/trial/dir/ckpts/last.ckpt system.exporter_type=mesh-exporter
# specify system.exporter.fmt=obj to get obj with vertex colors
# you may also add system.exporter.save_uv=false to accelerate the process, suitable for a quick peek of the result
python launch.py --config path/to/trial/dir/configs/parsed.yaml --export --gpu 0 resume=path/to/trial/dir/ckpts/last.ckpt system.exporter_type=mesh-exporter system.exporter.fmt=obj
# for NeRF-based methods (DreamFusion, Magic3D coarse, Latent-NeRF, SJC)
# you may need to adjust the isosurface threshold (25 by default) to get satisfying outputs
# decrease the threshold if the extracted model is incomplete, increase if it is extruded
python launch.py --config path/to/trial/dir/configs/parsed.yaml --export --gpu 0 resume=path/to/trial/dir/ckpts/last.ckpt system.exporter_type=mesh-exporter system.geometry.isosurface_threshold=10.
# use marching cubes of higher resolutions to get more detailed models
python launch.py --config path/to/trial/dir/configs/parsed.yaml --export --gpu 0 resume=path/to/trial/dir/ckpts/last.ckpt system.exporter_type=mesh-exporter system.geometry.isosurface_method=mc-cpu system.geometry.isosurface_resolution=256
```

For all the options you can specify when exporting, see [the documentation](https://github.com/threestudio-project/threestudio/blob/main/DOCUMENTATION.md#exporters).

See [here](https://github.com/threestudio-project/threestudio#supported-models) for example running commands of all our supported models. Please refer to [here](https://github.com/threestudio-project/threestudio#tips-on-improving-quality) for tips on getting higher-quality results, and [here](https://github.com/threestudio-project/threestudio#vram-optimization) for reducing VRAM usage.

### Ablations
There are 5 main parameters in `system.guidance` reproduce the ablation results:
```yaml
enable_sdi: true # if true - the noise is obtained by running DDIM inversion procvess, if false - noise is sampled randomly as in SDS
inversion_guidance_scale: -7.5 # guidance scale for DDIM inversion process
inversion_n_steps: 10 # number of steps in the inversion process
inversion_eta: 0.3 # random noise added to in the end of the inversion process
t_anneal: true # if true - timestep t is annealed from 0.98 to 0.2 instead of sampled from U[0.2, 0.98] like in SDS
```

## Citing

If you find our project useful, please consider citing it:

```
@misc{lukoianov2024score,
    title={Score Distillation via Reparametrized DDIM}, 
    author={Artem Lukoianov and Haitz Sáez de Ocáriz Borde and Kristjan Greenewald and Vitor Campagnolo Guizilini and Timur Bagautdinov and Vincent Sitzmann and Justin Solomon},
    year={2024},
    eprint={2405.15891},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
