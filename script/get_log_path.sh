#!/usr/bin/env bash

CONTAINER=$1

docker inspect $CONTAINER | grep LogPath | cut -d:
