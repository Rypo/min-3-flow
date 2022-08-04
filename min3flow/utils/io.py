import os
import requests

def download(url: str, fpath: str, chunk_size=1024):
    # ref: https://gist.github.com/yanqd0/c13ed29e29432e3cf3e7c38467f42f51
    from tqdm.auto import tqdm
    resp = requests.get(url, allow_redirects=True, stream=True)
    total = int(resp.headers.get('content-length', 0))
    with open(fpath, 'wb') as file, tqdm(desc=fpath,total=total,unit='iB',unit_scale=True,unit_divisor=1024) as pbar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            pbar.update(size)



def download_weights(weight_fpath, weight_url, force_download=False):
    #model_path = os.path.join(weights_root, weight_name)
    if (not os.path.exists(weight_fpath)) or force_download:
        os.makedirs(os.path.dirname(weight_fpath), exist_ok=True)
        
        #url = weight_base_url.format(os.path.basename(model_path))
        download(weight_url, weight_fpath)
    return weight_fpath