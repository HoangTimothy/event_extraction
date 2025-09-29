#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script để tạo file JSONL hoàn toàn sạch và tương thích với Colab
"""

import json
import re

def create_clean_jsonl(input_file, output_file):
    """
    Tạo file JSONL hoàn toàn sạch, mỗi dòng 1 JSON object
    """
    print(f"Đang tạo file JSONL sạch từ {input_file}...")
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        line_count = 0
        valid_count = 0
        error_count = 0
        
        for line_num, line in enumerate(infile, 1):
            line_count += 1
            
            # Xóa khoảng trắng đầu/cuối
            line = line.strip()
            
            # Bỏ qua dòng rỗng
            if not line:
                continue
            
            try:
                # Parse JSON
                record = json.loads(line)
                
                # Đảm bảo có đầy đủ các trường cần thiết
                if not all(key in record for key in ['doc_id', 'sent_id', 'sentence']):
                    print(f"Dòng {line_num}: Thiếu trường bắt buộc")
                    error_count += 1
                    continue
                
                # Ghi lại JSON object trên 1 dòng duy nhất
                json_line = json.dumps(record, ensure_ascii=False, separators=(',', ':'))
                outfile.write(json_line + '\n')
                valid_count += 1
                
            except json.JSONDecodeError as e:
                print(f"Dòng {line_num}: JSON không hợp lệ - {e}")
                print(f"Nội dung: {repr(line[:200])}")
                error_count += 1
                continue
            except Exception as e:
                print(f"Dòng {line_num}: Lỗi khác - {e}")
                error_count += 1
                continue
    
    print(f"\n=== KẾT QUẢ ===")
    print(f"Tổng số dòng đọc: {line_count}")
    print(f"Số dòng hợp lệ: {valid_count}")
    print(f"Số dòng lỗi: {error_count}")
    print(f"File output: {output_file}")

def test_jsonl_reading(filename):
    """
    Test việc đọc file JSONL như trên Colab
    """
    print(f"\n=== TEST ĐỌC FILE {filename} ===")
    
    with open(filename, "r", encoding="utf-8") as infile:
        for lineno, line in enumerate(infile, 1):
            # Xóa khoảng trắng đầu/cuối + bỏ dòng rỗng
            line = line.strip()
            if not line:
                continue
            try:
                # Parse JSON của từng dòng
                record = json.loads(line)
                print(f"Dòng {lineno}: ✅ OK - {record.get('doc_id', 'N/A')}")
                
                # Chỉ test 5 dòng đầu
                if lineno >= 5:
                    break
                    
            except Exception as e:
                # In số dòng + nội dung thực tế (repr để thấy ký tự ẩn)
                print(f"Lỗi ở dòng {lineno}: {repr(line[:200])}")
                print(f"Error: {e}")
                break

if __name__ == "__main__":
    input_file = "sentences_cleaned.jsonl"
    output_file = "sentences_colab_ready.jsonl"
    
    # Tạo file JSONL sạch
    create_clean_jsonl(input_file, output_file)
    
    # Test việc đọc file
    test_jsonl_reading(output_file)
    
    print("\n✅ Hoàn thành! File sẵn sàng cho Colab.")


