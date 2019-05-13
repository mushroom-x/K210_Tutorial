#!/bin/bash
echo "uasge: ./tflite2kmodel.sh xxx.tflite"
name=`echo $1 | cut -d '.' -f 1`
name=$name.kmodel
./ncc/ncc -i tflite -o k210model --dataset images $1 ./$name
