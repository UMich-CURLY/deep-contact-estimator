# This is a file that converts torch model to TensorRT model

import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import tensorrt as trt
# to load the input we need the following imports:
import sys
sys.path.append("..")
import torch
import argparse
from utils.data_handler import *
import yaml
print(torch.__version__)

# logger to capture errors, warnings, and other information during the build and inference phases
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine(ONNX_FILE_PATH):
    # initialize TensorRT engine and parse ONNX model
    EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(EXPLICIT_BATCH)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # Parse ONNX
    with open(ONNX_FILE_PATH, 'rb') as model:
        print('Beginning ONNX file parsing')
        parser.parse(model.read())
    print('Completed parsing of ONNX file')

    # allow TensorRT to use up to 1GB of GPU memory for tactic selection
    builder.max_workspace_size = 1 << 30
    # we have only one input in batch
    builder.max_batch_size = 1
    # use FP16 mode if possible
    if builder.platform_has_fast_fp16:
        builder.fp16_mode = True
    
    # TensorRT fails to recogonize output layer:
    last_layer = network.get_layer(network.num_layers - 1)
    # Checkif last layer recognizes it's output
    if not last_layer.get_output(0):
        # If not, then mark the output using TensorRT API
        network.mark_output(last_layer.get_output(0))
    

    # Generate TensorRT engine optimized for the target platform:
    print("Building an engine...")
    engine = builder.build_cuda_engine(network)
    context = engine.create_execution_context()
    print("Completed creating Engine")

    return engine, context


def main():
    print("In main")
    # initialize TensorRT engine and parse ONNX model
    ONNX_FILE_PATH = '/home/tingjun/Desktop/Cheetah_code/deep-contact-estimator/results/0412_1dcnn_64_128_no_tao_GRF.onnx'
    engine, context = build_engine(ONNX_FILE_PATH)

    # Get sizes of input and output and allocate memory required for
    # for input data and for output data
    for binding in engine:
        if engine.binding_is_input(binding): # One input only
            input_shape = engine.get_binding_shape(binding)
            input_size = trt.volume(input_shape) * engine.max_batch_size \
                         * np.dtype(np.float32).itemsize # in bytes
            device_input = cuda.mem_alloc(input_size)

        else: # and one output
            output_shape = engine.get_binding_shape(binding)
            # create page-locked memory buffers (i.e. won't be swapped to disk)
            host_output = cuda.pagelocked_empty(trt.volume(output_shape) \
                          * engine.max_batch_size, dtype=np.float32)
            device_output = cuda.mem_alloc(host_output.nbytes)

    # Create a stream in which to copy inputs/outputs and run inference.
    stream = cuda.Stream()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser(description='Test the contcat network')
    parser.add_argument('--config_name', type=str,
                        default=os.path.dirname(os.path.abspath(__file__)) + '/../config/test_params.yaml')
    args = parser.parse_args()
    config = yaml.load(open(args.config_name))

    # get host_input:
    test_data = contact_dataset(data_path=config['data_folder']+"test_lcm.npy",\
                            label_path=config['data_folder']+"test_label_lcm.npy",\
                            window_size=config['window_size'], device=device)
    test_dataloader = DataLoader(dataset=test_data, batch_size=1)
    for i in test_dataloader:
        print(i['data'].shape)
        input = i['data']
    
    print("Insert input")
    host_input = np.array(input.cpu().numpy(), dtype=np.float32, order='C')    
    cuda.memcpy_htod_async(device_input, host_input, stream)
    # run inference
    print("Flag1")
    context.execute_async(bindings=[int(device_input), int(device_output)], stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(host_output, device_output, stream)
    stream.synchronize()

    # postprocess results
    print("output: ", host_output)
    # postprocess(output_data)

if __name__ == '__main__':
    main()