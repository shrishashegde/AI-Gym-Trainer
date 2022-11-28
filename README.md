# AI-Gym-Trainer

An assistant to help you with the gym excersises

## List of team members
* Shrisha Shridhar Hegde
  * shrisha.hegde@sjsu.edu
  * 015955797
  * MS AI
* Suruchi Sharma
  * suruchi.sharma@sjsu.edu
  * 015900794
  * MS AI
* Ankur
  * ankur@sjsu.edu
  * 015266316
  * MS AI
* Kishore
  * place holder
  * place holder
  * place holder

## Team Coordinator
Shrisha Shridhar Hegde

## Execution Summary
Clone the repositry and set up the environment:
```
git clone https://github.com/shrishashegde/AI-Gym-Trainer.git
conda create ai_gym_trainer
pip install -r requirements.txt
```
To run the mediapipe model:

excercise_type can be push-up, pull-up or squat. video_src_file is the file that is fed to mediapipe. video_output_file is the destination file where output is stored

```
cd AI-Gym-Trainer
python src\main.py -t <excercise_type> -vs <video_src_file> -vo <video_output_file>
```
## Execution Summary - Yolo
1) Clone the repository 
2) cd AI-Gym-Trainer

```
python src\yolo_main.py -t <excercise_type> -vs <video_src_file(yolo_annotated_video)> -vo <video_output_file>
```
   * To annotate using yolo code:
     1) Go to the https://drive.google.com/drive/search?q=owner:shrisha.hegde%40sjsu.edu 
         -- CMPE 258 
            -- Yolo v7 folder 
              --unzip yolo_v7.zip 
     2) Install the required packages 
     3) Execute
        ```
        python yolo_v7_pose.py 
        ```
        Edit the path of the input video and output video in the code 


  * Video Alignment Code
    1) Expert Video path = v1_path_expert
    2) User Video path = v2_path_user
    ```
    cd AI-Gym-Trainer
    python src/video_sim.py -v1 <v1_path_expert> -v2 <v2_path_user>
    ```     
    Aligned Video will be stored in the root folder.
    Similarity score will be printed on the screen.

## Abstract
Objective:
The objective of this project is to assist people in measuring the quality of their exercises and aid in their health journey. We provide the effective number of repetitions and the quality of the exercises based on trainers ground truth videos.

Technical Challenges:
There have been many challenges while formulating, developing and implementing the ideas and models. Some of the major ones are listed below
1. Data Preparation. We have to annotate the data before using the data for anything meaningful. The annotation were done using deep neural network mediapipe, though we considered other options like Yolo as well.
2. Video Alignment. It is necessary to align two videos before comparing them. The video alignment is done using LAMV: Learning to Align and Match Videos with Kernelized Temporal Layers algorithm.

Methodology:
There are two different use cases of your AI-trainer.
1. User can use the solution to help count the number of effective repetition of a given exercise.
The solution use mediapipe for the annotations of the important thirty three points on the user's body. Then critical angles are utilized to compute the effective repetitions and repnet is utilized to compute total number of repetitions. 
Another solution is to utilize yolov7 for annotations of the important seventeen point's on user's body and then rest of the procedure is same as above.

2. User can use reference video of an expert, to calculate the degree of similarity between the two.
Utilizing the two video's frame and features to extract temporal match kernel features of same size, we can calculate similarity scores between the two feature maps of the video to find the best frame alignment and video similarity score.

Software Tools and Hardware:
Due to the sheer volume of the data, we have utilized HPC resources to store, process and develop the solution in conjunction with drives and local compute resources.
All of the development is done in Python and Jupyter notebooks.

Results:
Our results include output video files from our solutions.
One of the solution gives the true repetition count. Another use case provides the similarity score between two videos.

Experience:
It has been a steep learning curve for all of us.
Extensive literatue review gives us a mature view of the state of the art results in pose detection algorithms.
We have experimented with different approaches to tackle this problem.
The implementation of the solution in a group environment helped us to contribute in a collaborative manner.

## Work Contributions
| Team member                 | Responsibility           | Contributions  |
| --------------------------- |:------------------------:| :--------------|
| Shrisha Shridhar Hegde      | Dataset gathering, Critical angle extraction, mediapipe pose extraction, wrapper for extracting angles, code improvement by using OOPS, report and presentation | 1) Coding of pose, effective rep count, wrapper for extracting angles. About 30% of the entire project <br /> 2) Testing <br /> 3) PPT <br /> 4) Coordinator|
| Suruchi Sharma              |Annotating the videos using yolov7 network , counting total number of repetitions in videos  using repnet,some part of presentation  |  |
| Ankur                       |  |  |
| Kishore Kumaar              |  |  |

## References
1. Dwibedi, Debidatta, et al. "Counting out time: Class agnostic video repetition counting in the wild." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2020.
2. Wang, Chien-Yao, Alexey Bochkovskiy, and Hong-Yuan Mark Liao. "YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors." arXiv preprint arXiv:2207.02696 (2022).
3. Yang, Zetong, et al. "3d-man: 3d multi-frame attention network for object detection." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021.
4. Vaswani, Ashish, et al. "Attention is all you need." Advances in neural information processing systems 30 (2017).
5. https://arxiv.org/pdf/1902.09868.pdf
6. https://sites.google.com/view/repnet
7. https://colab.research.google.com/github/google-research/google-research/blob/master/repnet/repnet_colab.ipynb
8. https://github.com/WongKinYiu/yolov7
9. https://github.com/facebookresearch/videoalignment
10. https://research.facebook.com/publications/lamv-learning-to-align-and-match-videos-with-kernelized-temporal-layers/
