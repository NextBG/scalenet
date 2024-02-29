#!/bin/bash

# Destination directory
DEST_DIR="/home/caoruixiang/datasets_mnt/scalenet"

# Create the destination directory if it does not exist
mkdir -p $DEST_DIR

# Remove all file in partitions directory
rm -rf $DEST_DIR/partitions/*
echo "Remove all file in partitions directory"

# Rsync from the remote pc
ROUTE_1="caoruixiang@192.168.123.68"
ROUTE_2="caoruixiang@192.168.88.57"

echo "Rsync from $ROUTE_1"
rsync -av --ignore-existing $ROUTE_1:/home/caoruixiang/vnav_data_collection/nvme_mnt/* $DEST_DIR/trajectories/

echo "Done"
