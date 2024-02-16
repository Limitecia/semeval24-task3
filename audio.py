import torchaudio
from torchvision.io import read_video
from concurrent.futures import ProcessPoolExecutor
import os, torch 
from tqdm import tqdm 

def load_audio(path: str):
    _, audio, _ = read_video(path, pts_unit='sec')
    torch.save(audio, path.replace('.mp4', '.wav'))

        
        
    
if __name__ == '__main__':
    num_workers = 20
    folder = 'dataset/video/'
    paths = [f'{folder}/{file}' for file in os.listdir(folder) if 
             file.endswith('.mp4') and not os.path.exists(folder + file.replace('.mp4', '.wav'))]
    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        results = list(tqdm(pool.map(load_audio, paths), desc='audio', total=len(paths)))


