## This folder contains ONNX model(s)

You may generate an ONNX model by using ['torch_to_onnx.py'] in ['utils/'] folder. Then you need to double check if the model is correctly saved in this ['weights/'] folder. If not, you may need to find and put the model in this folder. The model here will then be used to generate a corresponding Nvidia TensorRT engine. You can also change the name of the saved model.
