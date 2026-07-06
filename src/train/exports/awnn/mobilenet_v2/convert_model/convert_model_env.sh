#!/bin/bash

mkdir -p ../model

ln -s ../../scripts_model_convert/pegasus_import.sh pegasus_import.sh

ln -s ../../scripts_model_convert/pegasus_inference.sh pegasus_inference.sh

ln -s ../../scripts_model_convert/pegasus_quantize.sh pegasus_quantize.sh

ln -s ../../scripts_model_convert/pegasus_export_ovx_nbg.sh pegasus_export_ovx_nbg.sh