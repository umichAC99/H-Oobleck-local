# Oobleck Testing Instructions

### Setup the environment
You can either build the dockerfile or directly pull the image from the dockerhub. The docker image is available at `lukezhuz/h-oobleck`.
```
docker pull pullze/ditto_devel:mlir_cpu
```

### Run the docker container and mount your repo
```
sudo docker run -it -v/home/lukezhuz/H-Oobleck:/home/H-Oobleck pullze/ditto_devel:mlir_cpu
```

### Active the conda environment within docker container
```
conda activate oobleck
```

### Build Oobleck
```
# do this outside of docker
git submodule init
git submodule update
# do this in docker
cd /home/H-Oobleck/
pip install .
```

### Run our planning algorithm
Remove `-s` if you don't want to see the output
```
pytest tests/planning/test_pipeline_template.py -s
```
