import requests
import os
import threading

save_dir = './data'
lock = threading.Lock()

def download_single(i):
    """单个下载任务"""
    url = 'http://zhjw.scu.edu.cn/img/captcha.jpg'
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/152.0.0.0 Safari/537.36 Edg/152.0.0.0'}
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200 and 'image' in response.headers.get('Content-Type', ''):
            filename = os.path.join(save_dir, f"captcha_{i:05d}.jpg")
            with open(filename, 'wb') as f:
                f.write(response.content)
            with lock:
                print(f"第 {i} 张下载成功")
        else:
            with lock:
                print(f"第 {i} 张请求异常，状态码: {response.status_code}")
    except Exception as e:
        with lock:
            print(f"第 {i} 张下载失败: {e}")

def download(nums=80000, thread_num=8):
    threads = []
    
    for i in range(13864, nums):
        t = threading.Thread(target=download_single, args=(i,))
        t.start()
        threads.append(t)
        
        if len(threads) >= thread_num:
            for t in threads:
                t.join()
            threads = []
    
    for t in threads:
        t.join()
    
    print("所有下载任务完成！")

if __name__ == '__main__':
    os.makedirs(save_dir, exist_ok=True)
    download(nums=80000, thread_num=8)