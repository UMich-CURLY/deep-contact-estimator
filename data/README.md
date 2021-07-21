**This foulder contains data for the use of generating TensorRT engine:**

1. test_label_lcm.npy & test_lcm.npy: 
    Two numpy data that will be used in generating an ONNX model from a PyTorch model.

2. input_matrix.bin:
    A bin file that contains one batch of input data (size = 150 x 54). This will be used
    in generating a TensorRT engine from the ONNX model we generated before;