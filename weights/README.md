## This folder contains ONNX model(s)

You may generate an ONNX model by using **torch_to_onnx.py** in **utils/** folder. Then double check if the model is correctly safed in this **weights/** folder. If not, you may need to find and put the model in this folder. The model in this folder will then be used to generate a corresponding Nvidia TensorRT engine. You can also change the name of the saved model.
