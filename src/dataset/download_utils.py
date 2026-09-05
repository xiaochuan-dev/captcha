import zipfile
import os
from huggingface_hub import hf_hub_download

def download_zip():
    zip_path = hf_hub_download(
    repo_id="freexiaochuan/captcha",
    filename="data.zip",
    repo_type="dataset"
)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(".")
        
    print("解压完成！图片在 . 目录下")

def download_file(filename):

    if not os.path.exists(f'./{filename}'):
        hf_hub_download(
            repo_id="freexiaochuan/captcha",
            filename=filename,
            repo_type="dataset",
            local_dir="./"
        )

        if os.path.exists(f'./{filename}'):
            print(f"{filename} 文件下载成功")
        else:
            print(f"{filename} 文件下载失败")