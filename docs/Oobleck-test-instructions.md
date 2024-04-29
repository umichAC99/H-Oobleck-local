# Oobleck Testing Instructions

## Setup the environment
You can either build the dockerfile or directly pull the image from the dockerhub. The docker image is available at `pullze/ditto_devel`.

For example, to pull the image with CUDA feature enabled, run:

``` bash
docker pull pullze/ditto_devel:mlir
```

Note that this may take several minites and make sure you have at least 35GB space avaliable on your device, and installed `nvidia-container-toolkit` to enable the gpu access in the container.

### Run the docker container and mount your current directory
Make sure your current path is `H-Oobleck-local`, this step will bind the code directory to `/workspace` in the container.

```bash
docker run -it -v $(pwd):/workspace --gpus=all pullze/ditto_devel:mlir
```

### Active the conda environment within docker container
```bash
conda activate oobleck
```

### Build Oobleck
```
# do this outside of docker
git submodule init
git submodule update
# do this in docker
cd /workspace
pip install .
```

## To run Profiler
To run the profiler, make sure you are in the docker container with `---gpus=all` flag passed
```bash
pytest tests/planning/test_profiler.py -s
```
This command will fail if CUDA is not avaliabe. By default it will test `gpt2` model, make sure you have enough memory and CUDA memory.

The result of profiler can be fount at `/tmp/oobleck/profiles/`. We also provided an example result got on our machine for `gpt2-xl`.

## To run planning 
To run planning, make sure you are in the `ditto_devel` container that you previous build Oobleck.

Remove `-s` if you don't want to see the output. This will try to do the node folding and planning algorithm (both DP and brute-force) for `gpt2-xl` model.

This may take several HOURS depends on you machine configuration.
```bash
pytest tests/planning/test_pipeline_template.py -s
```

## To run XLA cost modeling (Separate Container & Repo Needed)
First you need to generate a bunch of HLO code from a BERT model. And then Next a script is run to do some simple text manipulation of these files to prepare them. Then you load this HLO code into an internal HLO representation within XLA, and use the XLA cost optimizer to generate timings. Finally, a python script is run to convert these times into json format for Oobleck to digest. 

This part (except for the first step, generating HLO from BERT) needs a separate Docker container and an additional repo to run on. Make sure you follow instructions carefully to prepare the environment.

Ditto XLA repo: [Link](https://github.com/umichAC99/ditto_xla)

```bash
# in the parent folde of H-Oobleck-local
git clone git@github.com:umichAC99/xla.git
cd xla
```

```
docker pull psenta/ditto_xla:latest
docker run -it psenta/ditto_xla:latest
```

All of the following commands must be run within the separate docker container, and in the path specified for each step.

### To Generate HLO from BERT
This is the only step in this chain that must be run within the Oobleck repository
instead of XLA. To generate STABLEHLO representation for BERT model, simply run: 

```bash
mkdir results
pytest tests/planning/test_hlo_string.py -s
```
Output: a directory full of mlir files containing HLO code for each layer of the model.

### Prepare the HLO for reading
Copy the STABLEHLO result to `xla` folder:
```bash
cp -r /workspace/H-Oobleck-local/results /workspace/xla/service/gpu/model/results
```
Run preparation script:
```bash
cd /workspace/xla/service/gpu/model/results/
./convertScript.sh
```
Output: a directory full of mlir files, edited in place, prepared to be loaded.

### Load HLO into XLA and spit out layer times
```bash
cd /workspace
bazel run //xla/service/gpu/model:gpu_cost_model_stats_collection_test
```
Output: `layer_times.txt` a file with the times expected for each layer on each device.

### To format the times for Oobleck consumption
```bash
cd /workspace/xla/service/gpu/model/results/
python3 json_generator.py
```
Ouptut: json files readable for Oobleck. In a parent directory `Oobleck` with a child directory for each gpu type, and a json file within each gpu type directory.

