# Oobleck Testing Instructions

### Setup the environment
You can either build the dockerfile or directly pull the image from the dockerhub. The docker image is available at `lukezhuz/h-oobleck`.
```
docker pull lukezhuz/h-oobleck:latest
```

### Run the docker container and mount your repo
```
sudo docker run -it -v/home/lukezhuz/H-Oobleck:/home/H-Oobleck lukezhuz/h-oobleck:latest
```

### Active the conda environment within docker container
```
conda activate oobleck
```

### Build Oobleck
```
cd /home/H-Oobleck/
pip install .
```

### Run the test
Remove `-s` if you don't want to see the output
```
pytest tests/planning/test_pipeline_template.py -s
```
