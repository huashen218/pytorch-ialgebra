#!/usr/bin/env bash
# Download COCO data for interpretation
mkdir -p ../coco
(
  cd ../coco
  wget --continue http://images.cocodataset.org/zips/val2014.zip
  wget --continue http://images.cocodataset.org/annotations/annotations_trainval2014.zip
  unzip val2014.zip
  unzip annotations_trainval2014.zip
)