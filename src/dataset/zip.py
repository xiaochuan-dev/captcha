from paddleocr import PaddleOCR

ocr = PaddleOCR(
    ocr_version="PP-OCRv5",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    enable_mkldnn=False, 
)


image_paths = [f"./data/captcha_{i:05d}.jpg" for i in range(11)]  # 0到10，共11张

result = ocr.predict(image_paths)
for res in result:
    text = res["rec_texts"][0]
    print()