from typing import NamedTuple
import torch
from models import TMK_Poullot

import cv2
import argparse

import matplotlib.pyplot as plt

class Args(NamedTuple):
    m: int = 16

ap = argparse.ArgumentParser()
ap.add_argument("-v1",
                "--video_1",
                type=str,
                help='Video 1 Path: Expert Video',
                required=True)

ap.add_argument("-v2",
                "--video_2",
                type=str,
                help='Video 2 Path: User Video',
                required= True)
args = vars(ap.parse_args())


model = TMK_Poullot(Args())

def data_prepare(video_path, dim=(120,120)):
    """
    """
    vid = cv2.VideoCapture(video_path)

    video_fps = vid.get(cv2.CAP_PROP_FPS),
    total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))

    print(f"Frame Per second: {video_fps } \nTotal Frames: {total_frames} \n Height: {height} \nWidth: {width}")
    
    res = torch.zeros(1, total_frames, dim[0]*dim[1])
    f = 0
    ret = True
    while ret:
        ret, frame = vid.read()
        if frame is not None:
            frame  = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
            frame = torch.Tensor.float(torch.from_numpy(frame))
            frame2 = torch.mean(frame, dim=2)
            frame_flat = torch.flatten(frame2)
            res[0][f] = frame_flat
            f += 1
    
    return res, total_frames

def Score_(a, b):
    """
    """
    f1, t1, fr1 = a
    f2, t2, fr2 = b
    tmk_fv_a = model.single_fv(f1, t1)
    tmk_fv_b = model.single_fv(f2, t2)
    
    # fr_max = max(fr1, fr2)
    fr_max = fr2
    offsets = torch.arange(-fr_max, fr_max).view(1, -1).float()
    scores = model.score_pair(tmk_fv_a, tmk_fv_b, offsets)

    return scores, torch.max(scores, 1)[0], torch.max(scores, 1)[1]

if __name__ == "__main__":
    frame_features_a, n_frames_a = data_prepare(args['video_1'])
    timestamps_a = torch.arange(n_frames_a).float().reshape(1,n_frames_a)

    frame_features_b, n_frames_b = data_prepare(args['video_2'])
    timestamps_b = torch.arange(n_frames_b).float().reshape(1,n_frames_b)

    s11, s12, s13 = Score_((frame_features_a, timestamps_a, n_frames_a), (frame_features_b, timestamps_b, n_frames_b))
    frame_s = torch.argmax(s11[0][:10])

    print(f"Max Similarity Score: {s12} at frame {s13}")
    
    cap = cv2.VideoCapture(args["video_2"])
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    cap.set(1, torch.IntTensor.item(frame_s))
    ret = True
    result = cv2.VideoWriter('result_align.avi',cv2.VideoWriter_fourcc(*'mp4v'), video_fps, (height, width))
    while ret:
        ret, frame = cap.read()
        if ret == True:
            result.write(frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break   
    
    cap.release()
    result.release()
        
    # Closes all the frames
    cv2.destroyAllWindows()