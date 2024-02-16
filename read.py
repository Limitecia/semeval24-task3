from concurrent.futures import ProcessPoolExecutor
from torchvision.io import read_video
from transformers import AutoImageProcessor
from torch.nn.functional import interpolate
import torch, os, cv2, time, shutil
from tqdm import tqdm
from typing import List 

COMPRESS_RATIO = 0.5

processor = AutoImageProcessor.from_pretrained('facebook/deit-tiny-distilled-patch16-224')


def process(path: str):
    frames, *_ = read_video(path, pts_unit='sec')
    # frames = interpolate(frames.permute(1, 2, 3, 0), (3, int(frames.shape[0]*COMPRESS_RATIO))).permute(3, 0, 1, 2)
    frames = processor(frames, return_tensors='pt').pixel_values
    torch.save(frames, path.replace('.mp4', '.fr'))
    

def read_cv2(path: str):
    video = cv2.VideoCapture(path)
    res, frame = video.read()
    frames = [frame]
    while res: 
        frames.append(frame)
        res, frame = video.read()
    return frames 

def read_images(path: str):
    folder = path.replace('.mp4', '')
    frames = [cv2.imread(f'{folder}/{file}') for file in os.listdir(folder)]
    return frames 


        
# if __name__ == '__main__':
#     num_workers = 20
#     folder = 'dataset/video/'
#     paths = [f'{folder}/{file}' for file in os.listdir(folder) if file.endswith('.mp4') and not os.path.exists(folder + file.replace('.mp4', '.fr'))]
#     with ProcessPoolExecutor(max_workers=num_workers) as pool:
#         results = list(tqdm(pool.map(process, paths), desc='video', total=len(paths)))


if __name__ == '__main__':
    folder = 'dataset/video/'
    paths = [f'{folder}/{file}' for file in os.listdir(folder) if file.endswith('.mp4') and not os.path.exists(folder + file.replace('.mp4', '.fr'))]
    paths = paths[:20]
    start = time.time()
    [read_cv2(path) for path in paths]
    end = time.time()
    print('Cv2 reading:', end-start)
    start = time.time()
    [read_images(path) for path in paths]
    end = time.time()
    print('Image reading:', end-start)

