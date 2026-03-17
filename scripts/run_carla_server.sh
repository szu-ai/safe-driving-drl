#!/usr/bin/env bash
set -e
./CarlaUE4.sh -opengl -quality-level=Low -windowed -ResX=800 -ResY=600 -carla-rpc-port=2200 -nosound
