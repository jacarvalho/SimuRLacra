#!/bin/sh
docker build . -t pyrado
wget -nc https://raw.githubusercontent.com/mviereck/x11docker/master/x11docker
wget -nc https://raw.githubusercontent.com/jfrazelle/dotfiles/master/etc/docker/seccomp/chrome.json
sudo chmod +x x11docker
echo "./x11docker --user=RETAIN -- --gpus all --security-opt seccomp=$(pwd)/chrome.json -- pyrado xterm" > run_docker.sh
sudo chmod +x run_docker.sh
echo "Execute run_docker.sh to launch a terminal in the docker container"
