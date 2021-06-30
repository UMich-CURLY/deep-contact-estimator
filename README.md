Deep Contact Estimator <a name="TOP"></a>
===================
The deep contact estimator takes in proprioceptive measurements from a quadruped robot and estimates the current contact state of the robot.

## Contact Data Sets
* We created contact data sets using an MIT mini cheetah robot.
* The contact data sets can be downloaded [here](https://drive.google.com/drive/folders/1-6Su1HfE2KC1vMg4nkzsFy0X-OSzNMCS?usp=sharing).

## Dependency
* Python3
* PyTorch
* SciPy
* Tensorboard
* scikit-learn
* Lightweight Communications and Marshalling (LCM)

## Docker
* We also provide [docker](https://docs.docker.com/get-started/) files for cuda 10.1 and 11.1 in [`docker/`](https://github.com/UMich-CURLY/deep-contact-estimator/tree/master/docker).
* Detailed tutorial on how to build the docker container can be found in the README in each docker folder.

## Process Training Data
1. The network takes numpy data as input. To generate training data, first download the contact data sets from [here](https://drive.google.com/drive/folders/1-6Su1HfE2KC1vMg4nkzsFy0X-OSzNMCS?usp=sharing).
2. Collect all the `.mat` file from each terrain into a folder. (You can also reserve some sequences for testing only.)
3. Change `mat_folder` in `config/mat2numpy_config.yaml` to the above folder.
4. Change `save_path` to a desired path.
5. Change `mode` to `train` and adjust `train_ratio` and `val_ratio` to desired values.
6. Run `python3 utils/mat2numpy.py`. The program should automatically concatenate all data and separate it into train, validation, and test in numpy.

## Process Test Sequence
* If you would like to generate a complete test sequence without splitting into train, validation and test sets, all you need to do is to change `mode` to `inference` and repeat the above process. 
* However, instead of putting all training data into the `mat` folder, you should only put the reserved test sequence in the folder.

## Train the Network
1. To train the network, first you need to modify the params in `config/network_params.yaml'.
2. Run `python3 src/train.py`.
3. The log will be saved as [Tensorboard](https://pytorch.org/docs/stable/tensorboard.html) format in `log_writer_path` you defined.

## Pretrained Model
* If you just want to evaluate the result, we also provide pretrained models, which can be found [here](https://drive.google.com/drive/folders/1JGw1BZRxDjMim04J-BR-NzckcrHKS7hK?usp=sharing).

## Test the Network
1. Modify `config/test_params.yaml`.
2. Run `python3 src/test.py`.
3. `test.py` will compute the accuracy, precision, and jaccard index for the test sets.

## Inference a Complete Sequence
1. Generate a complete test sequence following the steps in [Process Test Sequence](#process-test-sequence).
2. Modify `config/inference_one_seq_params.yaml`.
3. Set `calculate_accuracy` to `True` if you wish to also compute the accuracy of the sequence. (Note: This requires ground truth labels and will slow down the inference process.)
4. Set `save_mat` to `True` if you want the result to be generated in `.mat` file for MATLAB.
5. Set `save_lcm` to `True` if you wish to generate results as a LCM log.
6. Run `python3 src/inference_one_seq.py`.
7. The saved LCM log can be used in [cheetah_inekf_ros](https://github.com/UMich-CURLY/cheetah_inekf_ros) for mini cheetah state estimation.

## Citation
This work was submitted to a conference and is under review. 

