# EduMeet

In this work, we are presenting a new approach to solve the problem of user engagement recognition by using the concept of transfer learning, multi-task classification and depthwise separable convolutions.     

We used a dataset titled "Dataset for Affective States in E-Environments" ([DAiSEE](https://iith.ac.in/~daisee-dataset/)) for training the model.  
For transfer learning we used [Xception](https://arxiv.org/pdf/1610.02357.pdf) (a model which was trained on the large ImageNet dataset) as our base model which uses the concept of depthwise separable convolutions.

## Implementation details
We first removed the top classification layers of Xception and then added our classification head. We added this classification head in two ways, 
1. Added on top of the global average pooling layer directly.
2. Added two fully connected (FC) layers between the global average pooling layer and the classification head.   

Adding 2 fully connected layers improved our accuracy.  
We then fine-tuned our model by retraining the exit flow of Xception but it did not improve our accuracy by a good margin.  

To train our model on all classes simultaneously we used the concept of multi-task classification and hence added four classification layers on top of fully connected layers directly. (four output classes are Boredom, Engagement, Confusion, Frustration.)  

We extracted the frames from DAiSEE videos at 0.7 fps. which gave us less number of images for training but for testing we extracted the frames with the default frame rate of FFmpeg (a tool which is used to extract the frames from videos) which gave us a large number of images, even after testing on this large number of images our experimental result proves that training with less number of images did not affect the accuracy of our model.

With this modification on the pre-trained Xception model we have slightly outperformed the frame-level classification benchmark for some of the affective states. This benchmark is given in the research paper titled [DAiSEE: Towards User Engagement Recognition in the Wild](https://arxiv.org/pdf/1609.01885.pdf).

## Usage

Download DAiSEE dataset from their website: 
https://iith.ac.in/~daisee-dataset/  
This dataset contains 9068 video snippets and 2,723,882 frames.  
We provide simple python scripts to train and evaluate the model.  

To run the scripts using GPU follow the instructions to install tensorflow with GPU support here:  
https://www.tensorflow.org/install/docker  

After that run this command in the diectory where the given Dockerfile is located.
    
    docker build -t tfgpu:latest .
    docker run --gpus all -u 1000 --rm --name tf -it -v </full-path/to/user-engagement-recognition/>:/tf tfgpu:latest bash
    
Now refer the following commands to run remaining scripts.

#### Extract frames from DAiSEE videos
    
    python extract_frames.py -i <videos directory> -o <directory to store extracted frames>

#### Save file path and labels as NumPy arrays
This script takes input as a frame directory which contains all the extracted frames and label directory which should contain TrainLabels.csv, TestLabels.csv, and ValidationLabels.csv (CSV files are available inside the downloaded DAiSEE dataset folder.)   
This NumPy arrays will be used in the data input pipeline.
    
    python save_fiilepath_label -i <frames dir>  -l <label dir> -o <directory to store filepath label>

#### Train model  
This script requires pre-trained weights for Xception which can be downloaded from here:   
[Weights for pre-trained Xception model on ImageNet](https://storage.googleapis.com/tensorflow/keras-applications/xception/xception_weights_tf_dim_ordering_tf_kernels_notop.h5)  
    
    python train.py -i <weights dir> -n <filepath label dir> -o <directory to store trained model and logs>
    
#### Evaluate model  
    
    python evaluate.py -i <trained model dir> -n <filepath label dir> -o <direcoty to store results>
    
## Results
Top-1 accuracy for each class is given below.    
The accuracy we are showing here was measured when we extracted the frames with the default frame rate, which gave us a large number of images for testing as compared to training and when we added two fully connected layers before the classification head.  

|               | Accuracy  |
|---------------|:---------:|
| Boredom | 44.66% | 
| Engagement | 45.17% |
| Confusion | 67.22% |
| Frustration | 44.55% |

For reference we have also provided all the [screenshots](screenshots/) of script execution and experimental results.
