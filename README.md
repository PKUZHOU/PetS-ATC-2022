# Artifact of PetS (USENIX ATC'22)


PetS is a unified inference serving framework for parameter-efficient transformers. This prototype is developed based on [TurboTransformers](https://github.com/Tencent/TurboTransformers).

## Overview
[1. Project Structure](#1-project-structure)  
[2. Getting Started](#2-getting-started) (5 human-minutes + 20 compute-minutes)  
[3. Build Stuff](#3-build-stuff) (2 human-minutes + 5 compute-minutes)  
[4. Run Experiments](#4-run-experiments) (10 human-minutes + 60 compute-minutes)  
[5. Validate Results](#5-validate-results) (15 human-minutes + 1 compute-minutes)  
[6. How to reuse beyond paper](#6-how-to-reuse-beyond-paper) 


## 1. Project Structure

First of all, let's take a glance at the file structure of PetS. We only list the main components here:

```
├── 3rd                     # the 3rd-party dependencies
├── build.sh                # click-to-run build script (used in Step 2.)
├── docker_scripts          # docker scripts to set up the environments for PetS/SeqS/ParS (used in Step 2.)
├── pets_deps               # pre-downloaded 3rd-party dependencies
├── research                # scripts to reproduce the experimental results (used in Step 4.)
│   ├── download_bert.sh    # download the bert model, which is required by ParS 
│   ├── eval_batching_strategies.sh
│   ├── eval_breakdown.sh
│   ├── eval_multi_stream.sh
│   ├── eval_throughput_pars.sh
│   ├── eval_throughput_pets.sh
│   ├── eval_throughput_seqs.sh
│   ├── run_pets_main_results.sh 
│   ├── plot_results.sh     # plot the results (used in Step 5.)
│   ├── python_scripts      # python scripts for experiments, the batch-scheduling algorithm is implemented here
│   ├── exp_results         # the experimental results will be logged here
│   └── reproduced_figures  # the reproduced figures will be stored here
├── pull_docker_images.sh   # pull all the prepared docker images
├── tools                   # docker files & model converting scripts
└── turbo_transformers      # contains all the source files
    ├── core                # the PET manager, sparse tensor algebra, memory management, etc.
    ├── layers              # PET operators, c++ backend of the shared bert model, etc.
    ├── loaders 
    └── python              # the python interfaces and transformer model discriptions 
```

Something you might be interested in:
* We implement four PET algorithms in this file. `turbo_transformers/python/turbo_transformers/layers/modeling_pets.py` 
* We implement the PET manager in this file: `turbo_transformers/core/pet_manager.h`
* We implement the PET operators in this file: `turbo_transformers/layers/shadow_op.cpp`
* In this file, we call the C++ backend to register PETs and infer the shared model. `turbo_transformers/python/turbo_transformers/layers/modeling_shared_bert.py` 
* We mainly modify the `bert_attention.cpp`,`bert_intermediate.cpp`,`bert_output.cpp` and `multi_head_attention.cpp` in the `turbo_transformers/layers` folder to call the shared/PET operators. 



## 2. Getting Started


#### Hardware requirements

We have evaluated PetS on three hardware platforms, namely TX2, 1080Ti, and V100, in the paper. This artifact mainly targets the V100 and 1080Ti GPUs (To run it on TX2,  additional modifications are required). Many other GPU platforms (e.g., 2080Ti, P100, K80, etc.) may also be compatible with this artifact. However, the Ampere architecture (e.g., A100, A6000) is not currently supported by the sputnik library.  

For reference, we list our system configurations here:

The 1080Ti machine: 
> * OS: Ubuntu 18.04
>  * GPU: 1080Ti-11GB
>  * CPU: Intel Xeon E5-2690
>  * DRAM: 32GB 



The V100 machine:
> * OS: Ubuntu 18.04
> * GPU: V100-32GB
> * CPU: Intel Xeon Golden 5220
> * DRAM: 256GB 


Note that at least 20GB of free storage space is required to hold the docker images. For downloading the docker images, an internet connection is also required. Please make sure that your machine can access hub.docker.com
#### Setup the environment with nvidia-docker 


PetS and the SeqS (sequential serving) and ParS (parallel serving) baselines have complicated dependencies, so it is strongly recommended to use the `nvidia-docker` to set up the running environments. We have prepared the docker images for PetS and the SeqS/ParS baselines. To set up the docker environment, you should:

* **Install the nvidia-docker**

  If your machine has not installed the `nvidia-docker`, please install it following the official instructions: https://github.com/NVIDIA/nvidia-docker


  Please make sure that the nvidia-docker works properly after the installation.  It is recommended to add your user to the docker group to avoid switching to sudo each time: https://docs.docker.com/engine/install/linux-postinstall/

* **Pull the docker images**

  In the PetS folder, run the following script to pull all the required images from docker-hub:
  ```
  ~> bash pull_docker_images.sh
  ```

  The downloading time depends on your network bandwidth. After that, check the downloaded images using the following command:
    ```
    ~> docker images
    ```
  If everything is OK, there will be three new images: 
  ```
  thufeifeibear/turbo_transformers_gpu_dev 
  pkuzhou/pets_gpu_dev
  pkuzhou/pars_gpu_dev
  ```

## 3. Build Stuff (< 10 minutes)

We have built and installed SeqS/ParS in their docker images, which can be directly used to run the [experiments](#4-run-experiments). For PetS, we have to build it manually in the docker environment:

#### Start the container

To build PetS, please start the PetS container:

```
bash docker_scripts/run_pets_in_docker.sh
```

Then a container named `pets_dev` will be started, and you are now in the container.

In case you have closed the interactive terminal,  you can  **re-enter the container** using the following commands:

```
docker start pets_dev  # if it has been stopped 
docker exec -it pets_dev /bin/bash
```

#### Build PetS in the container environment

In the container, run the following commands to build the project:

```
~> cd /workspace
~> bash build.sh
```

The build usually takes less than 5 minutes to finish. If you see `Successfully installed turbo-transformers-0.6.0`, then PetS has been successfully installed in the container. Note that if you delete the container, PetS should be built again after executing the `run_pets_in_docker.sh` script.


## 4. Run Experiments (< 60 minutes)
 
We have provided many click-to-run scripts in the `research` folder for conducting the majority of experiments mentioned in the paper, including the fixed-length throughput (Fig. 7, Fig. 8), total supported tasks (Table 5.), execution time breakdown (Fig. 9.), PET operators scheduling (Fig. 11) and batch scheduling (Fig. 12).  We use the `bert-base` model in default. 
#### PetS main results all-in-one (~30 minutes):

To save the task-loading time, we provide an all-in-one script to load up to 128 tasks and run fixed-length throughput, PET operators scheduling, and batch scheduling all in once:

```
(in the pets_dev container)
~> cd /workspace/research
~> bash run_pets_main_results.sh
```

This script will write the QPS results to the `research/exp_results/pets/` folder. In [Step 5](#5-validate-results) we will interpret the results. If you encounter any errors when running this script, please refer to [Known issues](#known-issues).

#### Fixed-length throughput: (~15 minutes)

* PetS:  **(optional)**

  If you have not run the all-in-one script, you can run this script to evaluate the fixed-length throughput of PetS alone:

  ```
  (in the pets_dev container)
  ~> cd /workspace/research
  ~> bash eval_throughput_pets.sh
  ```

* SeqS 

  To evaluate the throughput of the SeqS baseline (Fig. 8), you have to switch to the SeqS environment. (Please exit the PetS docker first, or run SeqS docker in another terminal)

  ```
  (in the host environment)
  ~> bash docker_scripts/run_seqs_in_docker.sh
  ```

  This script will run the SeqS docker and execute the `research/eval_throughput_seqs.sh` automatically. 


* ParS 

  Finally, to evaluate the ParS throughput, you should switch to the ParS container:

  ```
  (in the host environmenet)
  bash docker_scripts/run_pars_in_docker.sh
  ```
  This script will also enter the ParS container and test the throughput automatically. 

The QPS results will be written to `research/exp_results/{pets/pars/seqs}/bert_base/serving_throughput_{PetS/ParS/SeqS}.log`

#### Different batching strategies (optional)

If you did not run the all-in-one script, you can still run the following script to evaluate the batch scheduling of PetS alone: 

```
(in the pets_dev container)
~> cd /workspace/research
~> bash eval_batching_strategies.sh
```

The results will be generated in `research/exp_results/pets/bert_base/compare_batching_PETS.log`


#### Multi-stream PET operator scheduling (optional)

If you did not run the all-in-one script, you can still run the following script to evaluate the PET operator scheduling: 

```
(in the pets_dev container)
~> cd /workspace/research
~> bash eval_multi_stream.sh
```

The results will be written to `research/exp_results/pets/bert_base/multi_stream_PETS.log`

#### Execution Time Breakdown (<5 minutes)

To profile the execution time of PetS (Fig. 9), please run the following commands in the PetS container:

```
(in the pets_dev container)
~> cd /workspace/research
~> bash eval_breakdown.sh
```

You will see the proportion of PET operators execution time under two configs: {bs=1,seq_len=64} and {bs=2,seq_len=32}. According to Figure-9 in the paper, you will find that the PET operators only take a small portion of the total execution time.

#### The serving capacity (optional)

Finally, you can run the following script to perform a stress test to find out how many PET tasks can be loaded to the GPU for serving (Table-5). This script will first load 128 tasks and increase the number of tasks by a step of 8, until encountering the out-of-memory error.

```
(in the pets_dev container)
~> cd /workspace/research
~> bash eval_capacity_pets.sh
```

Please note that this experiment usually takes a long time to finish (> 1 hour). ***You can skip this experiment if you don't want to waste such a long time.*** 


#### Run other configurations

To explore more configurations beyond the paper, you can modify the parameters in `research/python_scripts/pet_experiments.py`. For instance, you can add a new (batch_size, seq_len) configuration in line 292. 

To run other bert models, i.e. distil-bert and bert-large, please change the model configuration in the shell scripts. (For ParS, only bert-based model is supported in the script.)

#### Known issues:

In rare cases, PetS may encounter a `CUDA illegal memory access` error, especially when the engine is called for many times (e.g., running the main results all-in-one script).  We infer that this is caused by a [unfixed bug](https://github.com/Tencent/TurboTransformers/issues/191) of TurboTransformers. If you are faced with this issue, please run each task separately using the provided scripts. 


## 5. Validate Results

Based on the logged results, we can validate the  performance by reproducing (or partially reproducing) the figures in the paper.  Specifically, we provide a click-to-run script to plot the results. 



  ```
  (in the pets_dev container)
  ~> cd /worksapce/research
  ~> bash plot_results.sh
  ```

The figures, including fig_7, fig_8, fig_11  and fig_12  will be saved to the `research/reproduced_figures` folder. 


* **fig_7.jpg:** The plotted fig_7.jpg reproduces one of the subgraphs in Figure 7. You will see that when serving multiple tasks (16-64), PetS will show about 1.5x to 1.8x (depends on your GPU platform) higher QPS against the single-task baseline, which will confirm the claims in Section 6.2.2.   
* **fig_8.jpg:** fig_8.jpg reproduces one of the subgraphs in Figure 8. You will see that both PetS and ParS outperform SeqS in serving throughput. And the PetS's throughput becomes higher as the number of concurrent tasks increases from 2 to 16, which proves PetS's good scalability (see Section 6.2.3)
* **fig_11.jpg:** fig_11.jpg reproduces all the results of Figure 11. You will see that under some configurations (long sequences, few tasks), the performance increases with the total number of streams, while for the other configurations (short sequences, many tasks), the performance is the highest with a single stream. Note that, when running bert-base on V100, the PET operator scheduling will be less useful than on 1080ti (used in Figure 7.) This is also in line with the analysis in Section 6.3.3: On more powerful GPUs, the execution time of PET operators is too short for the GPU's scheduler to overlap concurrent streams, making the multi-stream scheduling less effective.  
* **fig_12.jpg:** fig_12.jpg reproduces Figure 12 in the paper. You will see that when the std values are set from 1-4, the CB strategy always achieves the highest QPS. For std=8, the alpha-only strategy will beat the CB strategy. For all the configurations, batch-scheduling performs better than the fixed-batching strategy. Note that since we use a 1080ti performance model for scheduling, the batch-scheduling performance may not be optimal. If you are interested, you can generate the performance model by running the `research/python_scripts/perf_model/generate_alpha_table.sh` and `research/python_scripts/perf_model/generate_beta_table.sh` for your GPUs (this will take a long time).


Note that, due to the difference in system configurations, it is possible that the reproduced results are slightly different from that in the paper. For reference, we also provide the V100 and 1080Ti results in folder `research/reproduced_figures/for_reference/` 

## 6. How to reuse beyond paper

#### Support New PET algorithms

A PET algorithm can work with PetS as long as it meets two requirements: 

* Its PET operations are **separable** (with necessary equivalent-transformations) from the shared operations.  

* The separated  PET operations are **light-weighted**. 

To support a new algorithm, we should first identify its PET operations. Then the related functions should be extended accordingly. For example, a [LoRA](https://arxiv.org/abs/2106.09685) algorithm has a formula: Y_t = X_t*W + s*X_tW_{down}W_{up}. Its PET operation is  a scaled dense MVM: s*X_tW_{down}W_{up}, which should be implemented as a new PET operator. 


After knowing the PET operations, three steps are required to add the new PET algorithm to PetS:

* **Step 1.**  Register a new PET type and implement the PET operations using Pytorch  APIs in  `python/turbo_transformers/layers/modeling_pets.py`.

* **Step 2.** Deal with the PET parameters loading. Add new loading functions in  `modeling_shared_bert.py` and `pet_manager.h`, respectively.

* **Step 3.**  Implement the PET operator in `shadow_op.cpp/shadow_op.h` 

* **Step 4.**  If the new PET operators should be called at  places that are different from the four PETs in the paper, you should also modify the bert layers backends, e.g., `bert_output.cpp`. 

