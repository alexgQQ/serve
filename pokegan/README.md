# Pix2Pix Inference Server

This is a implementation of [torchserve](../README.md) to handle my model inference.

## Model Archiving and Handler

This is an implementation of torchserve's custom services outlined [here](../docs/custom_service.md).
This contains the code for the custom data handler and the means to archive the model,

## Build

Some of the custom packages for this handler need aditional installs on the base image.
Build the docker image from [here](../docker/README.md) to use these handlers.

```sh
# sudo is needed for docker on my machine :(
cd ../docker
sudo ./build_image.sh -t us.gcr.io/pixelpopart-1764327015/pytorch/torchserve:latest-cpu

# If we need to push to GCP
sudo docker push us.gcr.io/pixelpopart-1764327015/pytorch/torchserve:latest-cpu
```

## Run

Run a local instance for testing or archiving.
Specific files need to be mounted to run properly. The mount directory 
needs to contain the model archive, torchserve config, audio and video assets and the GCP service account creds.

```sh
sudo docker run --rm -it -p 8080:8080 -p 8081:8081 -p 8082:8082 -p 7070:7070 -p 7071:7071 \
                --name serve -e TEMP=/tmp -e UPLOAD_BUCKET=pixelpopart-test \
                -e AUDIO_LOCATION=/home/model-server/data/assets/sound/sound.mp3 \
                -e VIDEO_TEMPLATE_LOCATION=/home/model-server/data/assets/video/base.mp4 \
                -e GOOGLE_APPLICATION_CREDENTIALS=/home/model-server/config/service-account.json \
                --workdir /home/model-server/src -d \
                -v ${PWD}/pokegan/data:/home/model-server/data \
                -v ${PWD}/pokegan/config:/home/model-server/config \
                -v ${PWD}/pokegan/src:/home/model-server/src \
                -v ${PWD}/pokegan/checkpoints:/home/model-server/checkpoints \
                us.gcr.io/pixelpopart-1764327015/pytorch/torchserve:latest-cpu
```

## Archive Model and Handler

This package has to be [archived](../model-archiver/README.md) on order to be used by torchserve. This prepares the proper .mar file for usage.
Torchserve will package the model and pickle the python modules.
Model entrypoint should be `model-file`, handler entrypoint should be `handler`, model checkpoint should be `serialized-file`, any additional 
files needed, such as for an import, should be provided with `extra-files` and custim python reqs should be `requirements-file`.

```sh
sudo docker exec -it serve bash

# Within container
torch-model-archiver --model-name pokegan --version 1.0 \
                     --model-file model.py \
                     --serialized-file /home/model-server/checkpoints/latest_net_G-0-1-0.pth \
                     --export-path /home/model-server/tmp \
                     --handler pokegan_handler.py \
                     --extra-files localutils.py \
                     --requirements-file requirements.txt

# Exit container
sudo docker cp serve:/home/model-server/tmp/pokegan.mar pokegan/data/model-store
```


TODOS:
* Random CORS errors I get from the client application. Something with cloudflare hosting, actual server is responding fine.
