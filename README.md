## ONNX helper scripts

This repo contains small helper scripts for working with [ONNX](https://onnx.ai) files

### Installation

Setup the environment for utility scripts, by creating a virtual environment and installing dependencies
Note: Tested using `python3.8`. Run the following commands:

```bash
git clone https:/github.com/rahul-tuli/onnx-helpers
cd onnx-helpers
python3.8 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Scripts

- [Sample Out Generator](./sample_out_generator.py) A script to generate 
sample outputs from sample inputs and a given 
ONNX file, supports both DeepSparse and ONNX Runtime Engine. Inspired from an original version by 
[Ben Fineran](https://github.com/bfineran). Find more about the usage by `python3 sample_out_generator.py --help`

Examples:

```bash
❯ python sample_out_generator.py --help
usage: Generate sample outputs and from inputs [-h] --model-path MODEL_PATH --sample-inputs SAMPLE_INPUTS [--save-dir SAVE_DIR]
                                               [--engine {ort,deepsparse}]

optional arguments:
  -h, --help            show this help message and exit
  --model-path MODEL_PATH
                        ONNX model filepath
  --sample-inputs SAMPLE_INPUTS
                        Directory containing sample inputs
  --save-dir SAVE_DIR   Directory to save the sample outs, copy inputs and model. Defaults to model directory
  --engine {ort,deepsparse}
                        Engine to use for generating outs
```

```bash

python sample_out_generator --model-path ONNX_FILE \
--sample-inputs INPUTS_DIRECTORY \
--save-dir SAVE_DIR \
--engine deepsparse
```

