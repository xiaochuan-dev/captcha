import os
import zipfile
import json
from paddleocr import PaddleOCR
from huggingface_hub import hf_hub_download

zip_path = hf_hub_download(
    repo_id="freexiaochuan/captcha",
    filename="data.zip",
    repo_type="dataset"
)

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(".")
    
print("解压完成！图片在 . 目录下")

ocrv6 = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    enable_mkldnn=False,
    device="gpu",
    lang='en',
)

ocrv5 = PaddleOCR(
    ocr_version="PP-OCRv5",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    enable_mkldnn=False,
    device="gpu",
    lang='en', 
)

def get_label(start, end):
    with open('./data.json', 'r', encoding='utf-8') as f:
        data = json.load(f) 

    image_paths = [f"./data/captcha_{i:05d}.jpg" for i in range(start, end)]

    result1 = ocrv5.predict(image_paths)
    result2 = ocrv6.predict(image_paths)

    l = len(result1)

    for i in range(l):
        text1 = result1[i]["rec_texts"][0]
        text2 = result2[i]["rec_texts"][0]

        input_path = result1[i]["input_path"]

        if text1 is None:
            text1 = ""
        if text2 is None:
            text2 = ""

        data[input_path] = {
            "t1": text1,
            "t2": text2
        }
    with open('./data.json', 'w+', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"{start}-{end} done")

if __name__ == '__main__':

    i = 0

    while i < 80000:
        get_label(i, i + 1000)
        i += 1000