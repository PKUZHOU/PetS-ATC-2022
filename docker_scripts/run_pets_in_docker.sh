nvidia-docker run --gpus 1  -it   -v $PWD:/workspace -v /etc/passwd:/etc/passwd --name=pets_dev pkuzhou/pets_gpu_dev:latest
