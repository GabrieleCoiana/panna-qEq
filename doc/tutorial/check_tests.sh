#!/bin/bash

#Run Gvect calculator
echo -n "Running Gvector calculator...  "
python3 ../../src/panna/gvect_calculator.py --config input_files/gvect_sample.ini > Gvect.out 2> Gvect.err
if [ $? -eq 0 ]; then
    echo OK
    rm Gvect.out Gvect.err gvect_already_computed.dat
else
    echo FAIL, check the output/error files.
    exit 1
fi

#Run TFR calculator
echo -n "Running TFR packer...  "
python3 ../../src/panna/tfr_packer.py --config ./input_files/tfr_sample.ini > tfr.out 2> tfr.err
if [ $? -eq 0 ]; then
    echo OK
    rm tfr.out tfr.err
    rm -rf gvectors tfr
else
    echo FAIL, check the output/error files.
    exit 1
fi

#Run Train1
echo -n "Running training...  "
python3 ../../src/panna/train.py --config input_files/train1.ini > train1.out 2> train1.err
if [ $? -eq 0 ]; then
    echo OK
    rm train1.out train1.err
else
    echo FAIL, check the output/error files.
    exit 1
fi

#Run Validation1
echo -n "Running validation...  "
python3 ../../src/panna/evaluate.py --config input_files/validation1.ini > val1.out 2> val1.err
if [ $? -eq 0 ]; then
    echo OK
    rm -rf tutorial_validate val1.out val1.err
else
    echo FAIL, check the output/error files.
    exit 1
fi

#Run Extract_weights
echo -n "Running weight extraction...  "
python3 ../../src/panna/extract_weights.py --config input_files/extract_weights.ini > extw.out 2> extw.err
if [ $? -eq 0 ]; then
    echo OK
    rm -rf tutorial_train extw.out extw.err
else
    echo FAIL, check the output/error files.
    exit 1
fi

#Run Train2
echo -n "Running RE-training...  "
python3 ../../src/panna/train.py --config input_files/train2.ini > train2.out 2> train2.err
if [ $? -eq 0 ]; then
    echo OK
    rm train2.out train2.err tf.log
    rm -rf tutorial_train_2 saved_weights
else
    echo FAIL, check the output/error files.
    exit 1
fi

echo "The whole tutorial can be run successfully!"

