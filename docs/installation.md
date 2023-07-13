# Installation

## Prerequisite

- NVIDIA GPU with at least 6GB VRAM. The more memory you have, the more methods and higher resolutions you can try.
- [NVIDIA Driver](https://www.nvidia.com/Download/index.aspx) whose version is higher than the [Minimum Required Driver Version](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html) of CUDA Toolkit you want to use.

## Install CUDA Toolkit

You can skip this step if you have installed sufficiently new version or you use Docker.

Install [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive).

- Example for Ubuntu 22.04:
  - Run [command for CUDA 11.8 Ubuntu 22.04](https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local)
- Example for Ubuntu on WSL2:
  - `sudo apt-key del 7fa2af80`
  - Run [command for CUDA 11.8 WSL-Ubuntu](https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local)

## Git Clone

```bash
git clone https://github.com/threestudio-project/threestudio.git
cd threestudio/
```

## Install threestudio via Docker

1. [Install Docker Engine](https://docs.docker.com/engine/install/).
   This document assumes you [install Docker Engine on Ubuntu](https://docs.docker.com/engine/install/ubuntu/).
2. [Create `docker` group](https://docs.docker.com/engine/install/linux-postinstall/).
   Otherwise, you need to type `sudo docker` instead of `docker`.
3. [Install NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#setting-up-nvidia-container-toolkit).
4. If you use WSL2, [enable systemd](https://learn.microsoft.com/en-us/windows/wsl/wsl-config#systemd-support).
5. Edit [Dockerfile](../docker/Dockerfile) for your GPU to speed-up build.
   The default Dockerfile takes into account many types of GPUs.
6. Run Docker via `docker compose`.

```bash
cd docker/
docker compose build  # build Docker image
docker compose up -d  # create and start a container in background
docker compose exec threestudio bash  # run bash in the container

# Enjoy threestudio!

exit  # or Ctrl+D
docker compose stop  # stop the container
docker compose start  # start the container
docker compose down  # stop and remove the container
```

Note: The current Dockerfile will cause errors when using the OpenGL-based rasterizer of nvdiffrast.
You can use the CUDA-based rasterizer by adding commands or editing configs.

- `system.renderer.context_type=cuda` for training
- `system.exporter.context_type=cuda` for exporting meshes

[This comment by the nvdiffrast author](https://github.com/NVlabs/nvdiffrast/issues/94#issuecomment-1288566038) could be a guide to resolve this limitation.
