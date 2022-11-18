"""export checkpoint file into air, onnx, mindir models"""
import argparse
import numpy as np

import mindspore.common.dtype as ms
from mindspore import Tensor, load_checkpoint, load_param_into_net, export, context

from src.oneStep import OneStepCell
from src.model import wide_resnet50_2

parser = argparse.ArgumentParser(description='export')

parser.add_argument('--device_id', type=int, default=0, help='Device id')
parser.add_argument('--ckpt_file', type=str, required=True, help='Checkpoint file path')
parser.add_argument('--file_name', type=str, default='PathCore', help='output file name')
parser.add_argument('--file_format', type=str, choices=['AIR', 'ONNX', 'MINDIR'], default='MINDIR', help='file format')
parser.add_argument('--device_target', type=str, choices=['Ascend', 'GPU', 'CPU'], default='Ascend',
                    help='device target')

args = parser.parse_args()

if __name__ == '__main__':
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)

    if args.device_target == "Ascend":
        context.set_context(device_id=args.device_id)

    assert args.ckpt_file is not None, "args.ckpt_file is None."

    # network
    network = wide_resnet50_2()
    param_dict = load_checkpoint(args.ckpt_file)
    load_param_into_net(network, param_dict)

    for p in network.trainable_params():
        p.requires_grad = False

    model = OneStepCell(network)

    input_arr = Tensor(np.ones([1, 3, 224, 224]), ms.float32)
    export(model, input_arr, file_name=args.file_name, file_format=args.file_format)
