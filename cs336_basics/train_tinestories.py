import cProfile
import pstats
import time
import os
import json
import psutil  # 메모리 측정을 위해 필요 (pip install psutil)
import sys
from cs336_basics.tokenizer import train_bpe # 구현하신 함수 임포트

def run_tinystories_experiment():
    # --- [과제 설정 사항] ---
    input_path = "tests/fixtures/tinystories_sample_5M.txt" # 실제 데이터 경로
    vocab_size = 10000                 # 명세서 요구치 
    special_tokens = ["<|endoftext|>"] # 명세서 요구치 
    
    if not os.path.exists(input_path):
        print(f"에러: {input_path} 파일이 없습니다.")
        return

    # --- [훈련 및 프로파일링 시작] ---
    process = psutil.Process(os.getpid())
    start_time = time.time()
    
    pr = cProfile.Profile()
    pr.enable()

    print(f"TinyStories BPE 학습 시작 (Vocab: {vocab_size})...")
    vocab, merges = train_bpe(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens
    )

    pr.disable()
    end_time = time.time()
    
    # --- [과제 질문 답변용 데이터 계산] ---
    training_time_min = (end_time - start_time) / 60
    memory_usage_gb = process.memory_info().rss / (1024 ** 3) # 
    
    # 가장 긴 토큰 찾기 
    longest_token = max(vocab.values(), key=len)
    
    # --- [결과 출력] ---
    print("\n" + "="*50)
    print(f"1. 학습 소요 시간: {training_time_min:.2f} 분")
    print(f"2. 메모리 사용량: {memory_usage_gb:.2f} GB")
    print(f"3. 가장 긴 토큰: {longest_token}")
    print(f"4. 토큰 바이트 길이: {len(longest_token)}")
    print("="*50)

    # --- [결과 저장 (Serialization)]  ---
    output_vocab = "tinystories_vocab.json"
    # 바이트 객체는 JSON 저장이 안 되므로 변환 처리
    serializable_vocab = {k: v.hex() for k, v in vocab.items()}
    with open(output_vocab, "w") as f:
        json.dump(serializable_vocab, f)
    print(f"결과가 {output_vocab}에 저장되었습니다.")

    # --- [프로파일링 리포트 출력]  ---
    print("\n[병목 지점 분석 리포트]")
    ps = pstats.Stats(pr).sort_stats('cumulative')
    ps.print_stats(15)

if __name__ == "__main__":
    run_tinystories_experiment()