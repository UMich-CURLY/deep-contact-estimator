## This foulder contains data for the use of generating TensorRT engine

You may download the following datas through google drive link: 
https://drive.google.com/drive/folders/1DCCEQy2dJmEbbV-EThlwFpr4WurTzrs4?usp=sharing 


1. **test_label_lcm.npy & test_lcm.npy**: 
    
    Two numpy data that will be used in generating an ONNX model from a PyTorch model.

2. **input_matrix.bin**:
    
    A bin file that contains one batch of input data (size = 75 x 54). This will be used
    in generating and serializing a TensorRT engine from the ONNX model in **/weights** folder.