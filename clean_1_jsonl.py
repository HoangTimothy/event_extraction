#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script để làm sạch file JSONL, loại bỏ các dữ liệu CSV lẫn vào
"""

import json
import re
import sys

def clean_sentence_text(text):
    """
    Làm sạch text trong trường sentence, loại bỏ dữ liệu CSV lẫn vào
    """
    if not text:
        return text
    
    # Loại bỏ các pattern CSV thường gặpg
    # Pattern 1: ",1/3/2018 7:56," và các dữ liệu sau đó
    text = re.sub(r',\d+/\d+/\d+\s+\d+:\d+,.*$', '', text)
    
    # Pattern 2: Loại bỏ các dấu phẩy liên tiếp ở cuối
    text = re.sub(r',+$', '', text)
    
    # Pattern 3: Loại bỏ các dấu ngoặc kép kép và dữ liệu HTML/URL
    text = re.sub(r'""[^"]*"":\s*""[^"]*""', '', text)
    
    # Pattern 4: Loại bỏ các dấu ngoặc kép thừa ở đầu và cuối
    text = re.sub(r'^"+|"+$', '', text)
    
    # Pattern 5: Loại bỏ các dấu phẩy thừa ở đầu và cuối
    text = text.strip(',')
    
    # Pattern 6: Loại bỏ các ký tự đặc biệt CSV còn sót lại
    text = re.sub(r'^,+|,+$', '', text)
    
    # Pattern 7: Loại bỏ các ký tự đặc biệt ở đầu câu như }",\"
    text = re.sub(r'^[}\",]+', '', text)
    
    # Pattern 8: Loại bỏ các dấu ngoặc kép thừa
    text = re.sub(r'^"+|"+$', '', text)
    
    return text.strip()

def is_valid_json_line(line):
    """
    Kiểm tra xem dòng có phải là JSON hợp lệ không
    """
    try:
        json.loads(line.strip())
        return True
    except json.JSONDecodeError:
        return False

def clean_jsonl_file(input_file, output_file):
    """
    Làm sạch file JSONL
    """
    cleaned_count = 0
    invalid_count = 0
    total_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            total_count += 1
            line = line.strip()
            
            if not line:
                continue
                
            try:
                # Parse JSON
                data = json.loads(line)
                
                # Làm sạch trường sentence nếu có
                if 'sentence' in data:
                    original_sentence = data['sentence']
                    cleaned_sentence = clean_sentence_text(original_sentence)
                    data['sentence'] = cleaned_sentence
                    
                    if original_sentence != cleaned_sentence:
                        cleaned_count += 1
                        print(f"Dòng {line_num}: Đã làm sạch sentence")
                
                outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
                
            except json.JSONDecodeError as e:
                invalid_count += 1
                print(f"Dòng {line_num}: JSON không hợp lệ - {e}")
                print(f"Nội dung: {line[:100]}...")
                continue
    
    print(f"\n=== KẾT QUẢ LÀM SẠCH ===")
    print(f"Tổng số dòng: {total_count}")
    print(f"Số dòng đã làm sạch: {cleaned_count}")
    print(f"Số dòng JSON không hợp lệ: {invalid_count}")
    print(f"File đã được lưu tại: {output_file}")

if __name__ == "__main__":
    input_file = "sentences.jsonl"
    output_file = "sentences_cleaned.jsonl"
    
    print("Bắt đầu làm sạch file JSONL...")
    clean_jsonl_file(input_file, output_file)
    print("Hoàn thành!")
