# Multi-discount Q-learning with ACME Framework

## Overview

This codebase is built from a fork from deepmind/acme frame work. https://github.com/deepmind/acme

Following are the key files we develop for Multi-discount Q-learning. Please refer to following files for code for Multi-discount Q-learning

1. [run_dqn.py](examples/gym/run_dqn.py]) : Main file for single RL agent
2. [service_manager](external_interface/service_manager) : Online I-AIM run
3. [vehicle_env_mp.py](external_env/vehicle_controller/vehicle_env_mp.py) : Environment for Trajectory + Cruise Control
4. [agent.py](acme/agents/tf/MOdqn/agent.py) : Multi-discount Q-learning Agent (Actor + learner)
5. [learning.py](acme/agents/tf/MOdqn/learning.py) : Multi-discount Q-learning learner
6. [transition.py](acme/adders/reverb/transition.py) [MoNStepTransitionAdder] : Creates sample for experience replay by multiplying the reward vector and discount vector and the reward dependent discount function. 
7. [actors.py](acme/agents/tf/actors.py) [GurobiLpActor] : Linear programming formulation

transition.py : This file contains how to convert the normal Q-learning equation into Multi-discount Q-learning according the equations given in the Section 5 in the manuscript.

### SMARTS

Note that the SMARTS simulator should be connected to python interface via zeromq message interface. The messages are transfered as JSON strings. The connection initiated from vehicle_env_mp.py file. 
Any environment can be connected to run_dqn.py to test against any other environment. 
SMARTS source code is not included in the source code.

### Run

Without connecting SMARTS you can run a simple simulation without a front vehicle for trajectory control task
```bash
 python run_dqn.py --controller RL --num_actions 3
```

With simulator;
```bash
 python run_dqn.py --controller RL --num_actions 3 --use_smarts --front_vehicle
```

### Dependencies

channels:
  - gurobi
  - https://conda.anaconda.org/gurobi
  - defaults
dependencies:
  - _libgcc_mutex=0.1=main
  - blas=1.0=mkl
  - ca-certificates=2020.12.8=h06a4308_0
  - certifi=2020.12.5=py37h06a4308_0
  - cycler=0.10.0=py37_0
  - dbus=1.13.16=hb2f20db_0
  - expat=2.2.9=he6710b0_2
  - fontconfig=2.13.0=h9420a91_0
  - freetype=2.10.2=h5ab3b9f_0
  - glib=2.65.0=h3eb4bd4_0
  - gst-plugins-base=1.14.0=hbbd80ab_1
  - gstreamer=1.14.0=hb31296c_0
  - gurobi=9.1.0=py37_0
  - icu=58.2=he6710b0_3
  - intel-openmp=2020.2=254
  - jpeg=9b=h024ee3a_2
  - kiwisolver=1.2.0=py37hfd86e86_0
  - lcms2=2.11=h396b838_0
  - ld_impl_linux-64=2.33.1=h53a641e_7
  - libedit=3.1.20191231=h14c3975_1
  - libffi=3.3=he6710b0_2
  - libgcc-ng=9.1.0=hdf63c60_0
  - libgfortran-ng=7.3.0=hdf63c60_0
  - libpng=1.6.37=hbc83047_0
  - libstdcxx-ng=9.1.0=hdf63c60_0
  - libtiff=4.1.0=h2733197_1
  - libuuid=1.0.3=h1bed415_2
  - libxcb=1.14=h7b6447c_0
  - libxml2=2.9.10=he19cac6_1
  - lz4-c=1.9.2=he6710b0_1
  - matplotlib=3.3.1=0
  - matplotlib-base=3.3.1=py37h817c723_0
  - mkl=2020.2=256
  - mkl-service=2.3.0=py37he904b0f_0
  - mkl_fft=1.1.0=py37h23d657b_0
  - mkl_random=1.1.1=py37h0573a6f_0
  - ncurses=6.2=he6710b0_1
  - numpy=1.19.1=py37hbc911f0_0
  - numpy-base=1.19.1=py37hfa32c7d_0
  - olefile=0.46=py_0
  - openssl=1.1.1i=h27cfd23_0
  - pandas=1.1.1=py37he6710b0_0
  - pcre=8.44=he6710b0_0
  - pillow=7.2.0=py37hb39fc2d_0
  - pip=20.2.2=py37_0
  - pyparsing=2.4.7=py_0
  - pyqt=5.9.2=py37h05f1152_2
  - python=3.7.9=h7579374_0
  - python-dateutil=2.8.1=py_0
  - pytz=2020.1=py_0
  - qt=5.9.7=h5867ecd_1
  - readline=8.0=h7b6447c_0
  - seaborn=0.10.1=py_0
  - setuptools=49.6.0=py37_0
  - sip=4.19.8=py37hf484d3e_0
  - six=1.15.0=py_0
  - sqlite=3.33.0=h62c20be_0
  - tk=8.6.10=hbc83047_0
  - tornado=6.0.4=py37h7b6447c_1
  - wheel=0.35.1=py_0
  - xz=5.2.5=h7b6447c_0
  - zlib=1.2.11=h7b6447c_3
  - zstd=1.4.5=h9ceee32_0
  - pip:
    - absl-py==0.10.0
    - astunparse==1.6.3
    - atari-py==0.2.6
    - bsuite==0.3.2
    - cachetools==4.1.1
    - chardet==3.0.4
    - chex==0.0.2
    - cloudpickle==1.3.0
    - dataclasses==0.6
    - decorator==4.4.2
    - descartes==1.1.0
    - dm-control==0.0.322773188
    - dm-env==1.2
    - dm-haiku==0.0.2
    - dm-reverb-nightly==0.1.0.dev20200708
    - dm-sonnet==2.0.0
    - dm-tree==0.1.5
    - frozendict==1.2
    - future==0.18.2
    - gast==0.3.3
    - glfw==1.12.0
    - google-auth==1.21.0
    - google-auth-oauthlib==0.4.1
    - google-pasta==0.2.0
    - grpcio==1.31.0
    - gym==0.17.2
    - h5py==2.10.0
    - idna==2.10
    - imageio==2.9.0
    - importlib-metadata==1.7.0
    - jax==0.1.75
    - jaxlib==0.1.52
    - keras-preprocessing==1.1.2
    - labmaze==1.0.3
    - lxml==4.5.2
    - markdown==3.2.2
    - mizani==0.7.1
    - networkx==2.5
    - oauthlib==3.1.0
    - opencv-python==4.4.0.44
    - opt-einsum==3.3.0
    - palettable==3.3.0
    - patsy==0.5.1
    - plotnine==0.7.1
    - portpicker==1.3.1
    - protobuf==3.13.0
    - pyasn1==0.4.8
    - pyasn1-modules==0.2.8
    - pyglet==1.5.0
    - pyopengl==3.1.5
    - pywavelets==1.1.1
    - pyzmq==19.0.2
    - requests==2.24.0
    - requests-oauthlib==1.3.0
    - rlax==0.0.2
    - rsa==4.6
    - scikit-image==0.17.2
    - scipy==1.4.1
    - statsmodels==0.12.0
    - tabulate==0.8.7
    - tb-nightly==2.3.0a20200722
    - tensorboard-plugin-wit==1.7.0
    - termcolor==1.1.0
    - tf-estimator-nightly==2.4.0.dev2020090201
    - tf-nightly==2.4.0.dev20200708
    - tfp-nightly==0.12.0.dev20200717
    - tifffile==2020.8.25
    - toolz==0.10.0
    - tqdm==4.48.2
    - trfl==1.1.0
    - urllib3==1.25.10
    - werkzeug==1.0.1
    - wrapt==1.12.1
    - zipp==3.1.0
    - zmq==0.0.0

