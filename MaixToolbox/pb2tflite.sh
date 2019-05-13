#/bin/bash
echo "This script help you generate cmd to convert *.pb to *.tflite"
echo "Please put your pb into workspace dir"
echo "1. pb file name: (don't include workspace)"
read pb_name
echo "2. input_arrays name:"
read in_name
tflite_name=`echo $pb_name | cut -d '.' -f 1`
tflite_name=$tflite_name.tflite
echo "3. output_arrays name:"
read out_name
echo "4. input width:"
read in_w
echo "5. input height:"
read in_h
echo "6. input channel:"
read in_ch

echo "-----------------------------------"
echo "The command you need is:"
echo "-----------------------------------"
echo toco --graph_def_file=workspace/$pb_name --input_format=TENSORFLOW_GRAPHDEF --output_format=TFLITE --output_file=workspace/$tflite_name --inference_type=FLOAT --input_type=FLOAT --input_arrays=$in_name  --output_arrays=$out_name --input_shapes=1,$in_w,$in_h,$in_ch
`toco --graph_def_file=workspace/$pb_name --input_format=TENSORFLOW_GRAPHDEF --output_format=TFLITE --output_file=workspace/$tflite_name --inference_type=FLOAT --input_type=FLOAT --input_arrays=$in_name  --output_arrays=$out_name --input_shapes=1,$in_w,$in_h,$in_ch`
