from setuptools import find_packages, setup

setup(
    name="threestudio",
    version='"0.2.3"',  # the current version of your package
    packages=find_packages(),  # automatically discover all packages and subpackages
    url="https://github.com/threestudio-project/threestudio",  # replace with the URL of your project
    author="Yuan-Chen Guo and Ruizhi Shao and Ying-Tian Liu and Christian Laforte and Vikram Voleti and Guan Luo and Chia-Hao Chen and Zi-Xin Zou and Chen Wang and Yan-Pei Cao and Song-Hai Zhang",  # replace with your name
    author_email="shaorz20@mails.tsinghua.edu.cn",  # replace with your email
    description="threestudio is a unified framework for 3D content creation from text prompts, single images, and few-shot images, by lifting 2D text-to-image generation models.",  # replace with a brief description of your project
    install_requires=[
        # list of packages your project depends on
        # you can specify versions as well, e.g. 'numpy>=1.15.1'
    ],
    classifiers=[
        # classifiers help users find your project by categorizing it
        # for a list of valid classifiers, see https://pypi.org/classifiers/
        "License :: Apache-2.0",
        "Programming Language :: Python :: 3",
    ],
)
