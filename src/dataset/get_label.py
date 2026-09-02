import os
import argparse
import multiprocessing as mp
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from paddleocr import PaddleOCR

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, required=True, help='起始文件编号')
    parser.add_argument('--end', type=int, required=True, help='结束文件编号（不包含）')
    parser.add_argument('--num_workers', type=int, default=4, help='并行进程数')
    parser.add_argument('--data_dir', default='./data', help='图片目录')
    parser.add_argument('--output_dir', default='./results', help='输出目录')
    parser.add_argument('--batch_size', type=int, default=50, help='每批处理的图片数')
    parser.add_argument('--ocr_version', default='v5', choices=['v5', 'v6'], help='使用哪个OCR版本')
    return parser.parse_args()

def get_file_paths(data_dir, start, end):
    """获取 captcha_00000.jpg 格式的图片路径"""
    all_paths = []
    missing_count = 0
    
    for i in range(start, end):
        filename = f"captcha_{i:05d}.jpg"
        full_path = Path(data_dir) / filename
        
        if full_path.exists():
            all_paths.append(str(full_path))
        else:
            missing_count += 1
    
    if missing_count > 0:
        print(f"⚠️  范围内有 {missing_count} 个文件不存在（已跳过）")
    
    return all_paths

def init_ocr(ocr_version):
    """初始化 PaddleOCR（每个进程独立初始化）"""
    if ocr_version == 'v5':
        return PaddleOCR(
            ocr_version="PP-OCRv5",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            enable_mkldnn=False,
            use_gpu=False,  # GitHub Actions 无 GPU，用 CPU
        )
    else:
        return PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            enable_mkldnn=False,
            use_gpu=False,
        )

def process_batch(args):
    """处理一批图片"""
    batch_id, image_paths, output_dir, ocr_version = args
    
    # 每个进程独立初始化 OCR
    ocr = init_ocr(ocr_version)
    
    results = []
    
    # PaddleOCR 支持批量预测，直接传入列表
    try:
        # 批量预测
        batch_results = ocr.predict(image_paths)
        
        # 提取识别结果
        for idx, img_path in enumerate(image_paths):
            basename = os.path.basename(img_path)
            file_id = basename.replace('captcha_', '').replace('.jpg', '')
            
            try:
                # 提取文本
                res = batch_results[idx]
                if res and len(res) > 0 and 'rec_texts' in res:
                    text = res['rec_texts'][0] if res['rec_texts'] else ''
                else:
                    text = ''
                
                results.append({
                    'id': file_id,
                    'filename': basename,
                    'text': text,
                    'status': 'success'
                })
            except Exception as e:
                results.append({
                    'id': file_id,
                    'filename': basename,
                    'text': '',
                    'error': str(e),
                    'status': 'error'
                })
    except Exception as e:
        # 如果整批失败，记录所有图片为错误
        for img_path in image_paths:
            basename = os.path.basename(img_path)
            file_id = basename.replace('captcha_', '').replace('.jpg', '')
            results.append({
                'id': file_id,
                'filename': basename,
                'text': '',
                'error': str(e),
                'status': 'error'
            })
    
    # 保存批次结果
    import pandas as pd
    batch_df = pd.DataFrame(results)
    batch_file = os.path.join(output_dir, f'batch_{batch_id:04d}.csv')
    batch_df.to_csv(batch_file, index=False, encoding='utf-8')
    
    return batch_file, len(results)

def merge_results(output_dir, output_file='label.txt'):
    """合并所有批次结果到 label.txt"""
    import pandas as pd
    import glob
    
    all_files = glob.glob(os.path.join(output_dir, 'batch_*.csv'))
    if not all_files:
        print("没有找到批次结果文件")
        return
    
    print(f"合并 {len(all_files)} 个批次文件...")
    
    all_dfs = []
    for f in all_files:
        df = pd.read_csv(f, encoding='utf-8')
        all_dfs.append(df)
    
    final_df = pd.concat(all_dfs, ignore_index=True)
    
    # 按 id 排序（数字排序）
    final_df['id_num'] = final_df['id'].astype(int)
    final_df = final_df.sort_values('id_num').drop('id_num', axis=1)
    
    # 写入 label.txt（每行一个文本）
    with open(os.path.join(output_dir, output_file), 'w', encoding='utf-8') as f:
        for _, row in final_df.iterrows():
            text = row['text'] if pd.notna(row['text']) else ''
            f.write(text + '\n')
    
    # 同时保存完整 CSV
    final_df.to_csv(os.path.join(output_dir, 'all_results.csv'), index=False, encoding='utf-8')
    
    print(f"✅ 合并完成！共 {len(final_df)} 条记录")
    print(f"📁 结果保存至: {os.path.join(output_dir, output_file)}")
    print(f"📁 完整CSV: {os.path.join(output_dir, 'all_results.csv')}")
    
    # 统计
    success_count = final_df[final_df['status'] == 'success'].shape[0]
    error_count = final_df[final_df['status'] == 'error'].shape[0]
    print(f"📊 成功: {success_count}, 失败: {error_count}")

def main():
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 获取指定范围的图片
    print(f"正在扫描编号 {args.start} 到 {args.end} 的图片...")
    print(f"文件格式：captcha_00000.jpg")
    print(f"使用 OCR 版本：{args.ocr_version}")
    
    all_images = get_file_paths(args.data_dir, args.start, args.end)
    total_images = len(all_images)
    
    print(f"找到 {total_images} 张图片")
    
    if total_images == 0:
        print("错误：没有找到任何图片")
        return
    
    # 分批
    batch_size = args.batch_size
    batches = []
    for i in range(0, total_images, batch_size):
        batch_id = i // batch_size
        batch_images = all_images[i:i+batch_size]
        batches.append((batch_id, batch_images, args.output_dir, args.ocr_version))
    
    print(f"分为 {len(batches)} 批，每批 {batch_size} 张")
    print(f"使用 {args.num_workers} 个进程并行")
    print("=" * 50)
    
    # 并行执行
    start_time = time.time()
    processed = 0
    
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(process_batch, batch): batch[0] 
                   for batch in batches}
        
        for future in as_completed(futures):
            batch_id = futures[future]
            try:
                batch_file, count = future.result()
                processed += count
                print(f"✅ 批次 {batch_id} 完成，进度 {processed}/{total_images}")
            except Exception as e:
                print(f"❌ 批次 {batch_id} 失败: {e}")
    
    elapsed = time.time() - start_time
    print("=" * 50)
    print(f"🎉 全部完成！共处理 {processed} 张图片")
    print(f"⏱️  耗时: {elapsed:.2f} 秒")
    print(f"📊 速度: {processed/elapsed:.1f} 张/秒")
    
    # 合并结果
    print("\n正在合并结果...")
    merge_results(args.output_dir)

if __name__ == '__main__':
    # 多进程必须用 spawn 启动方式
    mp.set_start_method('spawn', force=True)
    main()