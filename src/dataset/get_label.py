import os
from paddleocr import PaddleOCR

save_file = './results/label.txt'

ocrv6 = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    enable_mkldnn=False, 
)

ocrv5 = PaddleOCR(
    ocr_version="PP-OCRv5",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    enable_mkldnn=False, 
)

def get_label_v5():
    with open(save_file, 'w+', encoding='utf-8') as f:

        image_paths = [f"./data/captcha_{i:05d}.jpg" for i in range(10)]  # 0到10，共11张

        result1 = ocrv5.predict(image_paths)
        result2 = ocrv6.predict(image_paths)

        for res in result1:
            text = res["rec_texts"][0]
            f.write(text)
            f.write('\n')
        

if __name__ == '__main__':
    os.makedirs('results')
    get_label_v5()