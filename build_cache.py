import os
import io
import lmdb
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool

DATASET_NAME = "scalenet"

IN_TRAJS_DIR = f"/home/caoruixiang/datasets_mnt/{DATASET_NAME}/trajectories"
OUT_SIZE = (640, 360)
OUT_PATH = f'/home/caoruixiang/datasets_mnt/{DATASET_NAME}/cache_{OUT_SIZE[0]}x{OUT_SIZE[1]}.lmdb'

def process_image(args):
    traj_name, image_full_name = args
    image_path = os.path.join(IN_TRAJS_DIR, traj_name, "frames", image_full_name)

    with open(image_path, 'rb') as f:
        image_bytes = f.read()

    image = Image.open(io.BytesIO(image_bytes))
    image = image.resize(OUT_SIZE)
    with io.BytesIO() as output:
        image.save(output, format="JPEG")
        output_bytes = output.getvalue()

    # img_idx = image_full_name.split('.')[0]
    img_idx = int(image_full_name.split('.')[0])
    key = f"{traj_name}_{img_idx:06d}".encode()

    return key, output_bytes

def process_trajectory(traj_name):
    image_files = sorted(os.listdir(os.path.join(IN_TRAJS_DIR, traj_name, "frames")))
    with Pool() as pool:
        results = pool.map(process_image, [(traj_name, image_file) for image_file in image_files])
    return results

if not os.path.exists(OUT_PATH):
    with lmdb.open(OUT_PATH, map_size=2**40) as env:
        with env.begin(write=True) as txn:
            for traj_name in tqdm(sorted(os.listdir(IN_TRAJS_DIR))):
                results = process_trajectory(traj_name)
                for key, value in results:
                    txn.put(key, value)
    
    print("Done!")
else:
    print("Cache already exists!")
