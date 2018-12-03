#!/bin/bash
source activate test_pytorch 
python test.py --model-path models/bak_model/deepspeech_51.pth --test-manifest ./val_manifest.csv  --cuda
