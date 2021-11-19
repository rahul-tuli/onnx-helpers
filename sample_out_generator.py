import argparse
import os
import numpy
import onnx
import onnxruntime as ort
from deepsparse import compile_model
from sparseml.onnx.utils import override_model_batch_size
from sparseml.utils import tensor_export, create_dirs

from abc import ABCMeta, abstractmethod

"""
Utility Script for generating Sample inputs from sample outputs
"""


class Engine(ABCMeta):
    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass


class ORTEngine(Engine):
    def __init__(self, filepath):
        self.sess = ort.InferenceSession(filepath)
        # extract input/output tensor names
        model = onnx.load(filepath)
        self.input_names = [i.name for i in model.graph.input]
        self.outputs_names = [o.name for o in model.graph.output]

    def __call__(self, inputs):
        input_dict = dict(zip(self.input_names, inputs))
        outputs = self.sess.run(self.outputs_names, input_dict)
        outputs = [x.squeeze() for x in outputs]
        return outputs

class DeepSparseEngine(Engine):
    def __init__(self, filepath):
        self.sess = compile_model(filepath, batch_size=1, num_cores=None)

    def __call__(self, inputs):
        if not isinstance(inputs, list):
            inputs = [inputs]
        outputs = self.sess.mapped_run(inputs)
        outputs = [x.squeeze() for x in outputs.values()]
        return outputs

def cleanup_unused_initializers(model):
    all_names = set()
    for tens in model.graph.input:
        all_names.add(tens.name)
    for tens in model.graph.output:
        all_names.add(tens.name)
    for node in model.graph.node:
        for inp in node.input:
            all_names.add(inp)
        for out in node.output:
            all_names.add(out)
    to_del = []
    for init in model.graph.initializer:
        if init.name not in all_names:
            to_del.append(init)
    [model.graph.initializer.remove(init) for init in to_del]


def export_samples(model_path, data_path, save_dir, engine):
    # load first 100 data files
    files = os.listdir(data_path)
    files = list(sorted(files))[:100]
    files = [os.path.join(data_path, file) for file in files]

    # load model at batch size 1
    model = onnx.load(model_path)
    override_model_batch_size(model, 1)

    # clean model and re-save
    # cleanup_unused_initializers(model)  # uncomment if needed
    onnx.save(model, os.path.join(save_dir, model_path.split("/")[-1]))

    if engine == 'ort':
        engine = ORTEngine(filepath=model_path)
    else:
        engine = DeepSparseEngine(filepath=model_path)

    for idx, file in enumerate(files):
        data = numpy.load(file)
        data = [x.reshape(1, *x.shape) for x in data.values()]
        outputs = engine(data)

        tensor_export(
            numpy.load(file).values(),
            os.path.join(save_dir, "_sample-inputs"),
            "inp-{:04d}".format(idx),
        )
        tensor_export(
            outputs,
            os.path.join(save_dir, "_sample-outputs"),
            "out-{:04d}".format(idx),
        )


def parse_args():
    parser = argparse.ArgumentParser("Generate sample outputs and from inputs")
    parser.add_argument("--model-path", required=True, type=str, help="ONNX model filepath")
    parser.add_argument("--sample-inputs", required=True, type=str, help="Directory containing sample inputs")
    parser.add_argument("--save-dir", default=None, type=str,
                        help="Directory to save the sample outs, copy inputs and model. Defaults to model directory")
    parser.add_argument("--engine", default='ort', type=str, choices=['ort', 'deepsparse'],
                        help="Engine to use for generating outs")
    return parser.parse_args()


if __name__ == "__main__":
    config = parse_args()

    if config.save_dir:
        create_dirs(config.save_dir)
    else:
        config.save_dir = '/'.join(config.model_path.split("/")[:-1])
    export_samples(config.model_path, config.sample_inputs, config.save_dir, config.engine)
