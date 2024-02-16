import cv2, os, shutil
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

save = '/media/ana/Seagate Portable Drive/video_dataset/'

def read_video(path: str):
    save_folder = save + path.split('/')[-1].replace('.mp4', '')
    video = cv2.VideoCapture(path)
    res, frame = video.read()
    if os.path.exists(save_folder):
        shutil.rmtree(save_folder)
    os.makedirs(save_folder)
    i = 0
    while res:
        cv2.imwrite(f'{save_folder}/{i:05d}.jpg', frame)
        res, frame = video.read()
        i += 1
    
    
if __name__ == '__main__':
    folder = 'dataset/video/'
    paths = [folder + file for file in os.listdir(folder) if file.endswith('.mp4') and not os.path.exists(folder + file.replace('.mp4', ''))]
    with ProcessPoolExecutor(max_workers=20) as pool:
        list(tqdm(pool.map(read_video, paths), total=len(paths), desc='video dataset'))
    
        
