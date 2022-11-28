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
Clone the repositry:
```
git clone https://github.com/shrishashegde/AI-Gym-Trainer.git
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
         

## Abstract
place holder

## Work Contributions
| Team member                 | Responsibility           | Contributions  |
| --------------------------- |:------------------------:| --------------:|
| Shrisha Shridhar Hegde      |  |  |
| Suruchi Sharma              |  |  |
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
