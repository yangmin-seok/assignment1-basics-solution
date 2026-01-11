from __future__ import annotations

import os
from typing import BinaryIO
import regex as re
from collections import defaultdict
from typing import Iterable

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    
    # Define vocabulary and merges
    vocab: dict[int, bytes] = {} # index to byte sequence
    merges: list[tuple[bytes, bytes]] = [] # list of byte pair merges, 만들어진 순서대로 저장, ex) [(b'a', b'b'), (b'ab', b'c')]
    
    # Initialize vocabulary with special tokens and all single byte values
    vocab_idx = 0
    for special_token in special_tokens:
        vocab[vocab_idx] = special_token.encode("utf-8")
        vocab_idx += 1

    for i in range(0, 256):
        # 0부터 255까지 모든 1byte 값 추가. 이때 bytes(i)는 안되고 bytes([i])로 해야됨
        vocab[vocab_idx] = bytes([i]) # [1] = b'\x00', [2] = b'\x01', ..., [256] = b'\xff'
        vocab_idx += 1

    # Open the input file 
    with open(input_path, "rb") as f:
        num_processes = 4

        # bytes 형태로 special token 변환
        special_tokens_bytes = [token.encode("utf-8") for token in special_tokens]
        boundaries = find_chunk_boundaries(f, num_processes, special_tokens_bytes) # 균등으로 간격으로 나누고 다음 [endoftext] 토큰에서 멈춤

        # frequency table: dict[tuple[bytes], int]
        # 각 단어를 byte로 변환 후, freq_table에 빈도 수 추가
        # tuple로 관리해야 나중에 합칠 때 편함
        freq_table: dict[tuple[bytes], int] = {}

        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore") # str 형태로 읽기

            # Windows 줄바꿈을 Unix 스타일로 정규화: Test 통과를 위해서
            chunk = chunk.replace("\r\n", "\n")
       
            # split on special tokens (e.g <|endoftext|>)
            escape_tokens = [re.escape(token) for token in special_tokens] # 각 special token을 문자열로 취급, '|'가 포함될 수 있기 때문에 escape를 사용해서 하나의 문자열로 취급해줘야함
            pattern = "|".join(escape_tokens) # speical token을 OR로 찾기
            removed_special_tokens = re.split(pattern, chunk) # speical toeken 기준으로 split

            # pre-tokenization pattern using re.finditer
            for segment in removed_special_tokens: # special token 기준으로 나눠진 segment들에 대해
                for match in re.finditer(PAT, segment): # 단어 별로 분해, 이때 하나씩 읽기
                    # tuple로 1byte씩 쪼개서 저장. 각각을 bytes로 저장해줘야됨!!
                    token_bytes = tuple(bytes([b]) for b in match.group(0).encode("utf-8"))
                    freq_table[token_bytes] = freq_table.get(token_bytes, 0) + 1 # get으로 기본값 0 설정 후 빈도수 추가

        # initialize byte pair counts
        # 각 단어의 byte pair 빈도 수 계산
        # 예: b'abc'라면 (b'a', b'b'), (b'b', b'c') 쌍이 각각 1회 등장
        # pair가 포함된 단어들도 추적    

    pair_counts: dict[tuple[bytes, bytes], int] = defaultdict(int)
    pair_to_words: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = defaultdict(set)

    for word, freq in freq_table.items():
        for i in range(len(word) - 1):
            pair = (word[i], word[i+1])
            pair_counts[pair] = pair_counts.get(pair, 0) + freq
            pair_to_words[pair].add(word)

    # 4. 반복적인 병합 수행
    while len(vocab) < vocab_size:
        if not pair_counts:
            break

        # 빈도가 높고 사전순으로 큰 쌍 선택 (Tie-breaking rule) 
        max_pair = max(pair_counts.items(), key=lambda x: (x[1], x[0]))[0]
        
        if pair_counts[max_pair] <= 0:
            break
            
        merges.append(max_pair)
        new_token = max_pair[0] + max_pair[1]
        vocab[vocab_idx] = new_token
        vocab_idx += 1

        # 영향을 받는 단어들만 핀셋 업데이트
        # max_pair를 가지고 있는 단어 리스트를 복사하여 순회
        target_words = list(pair_to_words[max_pair])
        for old_word in target_words:
            count = freq_table[old_word]
            
            # (1) 기존 단어에서 모든 인접 쌍의 빈도 제거
            for i in range(len(old_word) - 1):
                p = (old_word[i], old_word[i+1])
                pair_counts[p] -= count
                pair_to_words[p].discard(old_word)

            # (2) 실제 병합 수행 
            new_word = []
            i = 0
            while i < len(old_word):
                if i < len(old_word) - 1 and (old_word[i], old_word[i+1]) == max_pair:
                    new_word.append(new_token)
                    i += 2
                else:
                    new_word.append(old_word[i])
                    i += 1
            new_word = tuple(new_word)
            
            # (3) 새로운 단어로 데이터 구조 갱신
            freq_table.pop(old_word)
            freq_table[new_word] = freq_table.get(new_word, 0) + count
            
            # 새로운 인접 쌍들의 빈도 추가 및 인덱싱
            for i in range(len(new_word) - 1):
                p = (new_word[i], new_word[i+1])
                pair_counts[p] = pair_counts.get(p, 0) + count
                pair_to_words[p].add(new_word)

        # 빈도가 0이 된 쌍은 관리 대상에서 제거
        # dictionary 크기를 줄여 max() 연산 속도 유지
        pair_counts = {k: v for k, v in pair_counts.items() if v > 0}

        # 횟수 debug 출력
        if len(merges) % 100 == 0:
            print(f"Merges done: {len(merges)}, Vocab size: {len(vocab)}")

    return vocab, merges



def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_tokens: list[bytes], # 단일 bytes에서 list[bytes]로 변경,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_tokens, list), "Must represent special tokens as a list of bytestrings"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

           # 모든 special token의 위치를 찾음
            found_positions = [mini_chunk.find(t) for t in split_special_tokens]
            # -1(찾지 못함)을 제외한 최소 위치값 탐색
            valid_positions = [pos for pos in found_positions if pos != -1]

            if valid_positions:
                found_at = min(valid_positions) # 가장 먼저 나타난 토큰의 위치
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))
class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str]):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens

        # vocab: id -> bytes
        # Create reverse mapping for encoding
        self.bytes_to_id = {bytes: id for id, bytes in vocab.items()} # encoding 시에 사용

        if special_tokens:
            sorted_tokens = sorted(special_tokens, key=len, reverse=True) # 길이가 긴 순서대로 정렬
            escape_tokens = [re.escape(token) for token in sorted_tokens] # 각 special token을 문자열로 취급, '|'가 포함될 수 있기 때문에 escape를 사용해서 하나의 문자열로 취급해줘야함
            self.special_token_pattern = "(" + "|".join(escape_tokens) + ")"# 괄호를 사용해서 special token도 살리기
        else:
            self.special_token_pattern = None

    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] = None):
        """
        Create a tokenizer from saved vocabulary and merges files.
        
        Args:
            vocab_filepath: Path to the pickled vocabulary file
            merges_filepath: Path to the pickled merges file  
            special_tokens: Optional list of special tokens
            
        Returns:
            Tokenizer instance
        """
        import pickle
        
        with open(vocab_filepath, 'rb') as f:
            vocab = pickle.load(f)
            
        with open(merges_filepath, 'rb') as f:
            merges = pickle.load(f)
            
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        if not text:
            return []
                
        # 1. Special Token 분할
        if self.special_token_pattern:
            segments = re.split(self.special_token_pattern, text)
        else:
            segments = [text]
        
        token_ids: list[int] = []
        
        for segment in segments:
            if not segment:
                continue
            
            # Special Token 처리
            if segment in self.special_tokens:
                segment_bytes = segment.encode("utf-8")
                if segment_bytes in self.bytes_to_id:
                    token_ids.append(self.bytes_to_id[segment_bytes])
                continue

            # 일반 텍스트 처리 (Pre-tokenization)
            for match in re.finditer(PAT, segment):
                word = match.group(0)
                # 초기 단위: 개별 바이트를 튜플로 감쌈
                token_bytes = tuple(bytes([b]) for b in word.encode("utf-8"))
                #print("token_bytes 초기 상태:", [t.decode('utf-8', errors='replace') for t in token_bytes])
                
                while len(token_bytes) > 1:
                    # 현재 상태에서 가능한 모든 인접 쌍(pair) 생성
                    pairs = []
                    for i in range(len(token_bytes) - 1):
                        pairs.append((token_bytes[i], token_bytes[i+1]))

                    best_pair = None
                    # 리스트라고 가정할 때 가장 작은 index가 우선순위가 높음
                    best_merge_idx = float('inf')
                
                    for pair in pairs:
                        if pair in self.merges:
                            # 자료구조 무시: 리스트의 index 추출 시나리오
                            current_idx = self.merges.index(pair) 
                            if current_idx < best_merge_idx:
                                best_merge_idx = current_idx
                                best_pair = pair

                    # 더 이상 병합할 규칙이 없으면 종료
                    if best_pair is None:
                        break

                    # 병합 수행 (문장 내의 모든 best_pair를 하나로 합침)
                    p1, p2 = best_pair
                    new_tokens = []
                    i = 0
                    while i < len(token_bytes):
                        if i < len(token_bytes) - 1 and (token_bytes[i], token_bytes[i+1]) == (p1, p2):
                            new_tokens.append(p1 + p2)
                            i += 2
                        else:
                            new_tokens.append(token_bytes[i])
                            i += 1
                    
                    token_bytes = tuple(new_tokens)
                    
                # 최종적으로 만들어진 토큰들을 ID로 변환
                for token in token_bytes:
                    if token in self.bytes_to_id:
                        token_ids.append(self.bytes_to_id[token])
                    else:
                        print(f"  [경고] Vocab에 없는 토큰 발견: {token}")
                        # 개별 바이트로 쪼개서라도 추가
                        for b in token:
                            token_ids.append(self.bytes_to_id[bytes([b])])

        return token_ids
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        # Iterable한 문자열을 받아서 각 문자열을 인코딩한 토큰 ID를 순차적으로 반환
        for text in iterable:
            for token_id in self.encode(text):
                yield token_id

    def decode(self, ids: list[int]) -> str:
        """Decode a sequence of token IDs into text"""
        byte_sequence = b"".join(self.vocab[id] for id in ids)
        return byte_sequence.decode("utf-8", errors="replace")

# Example usage and testing
if __name__ == "__main__":
    # Test with a small corpus
    test_corpus = "Hello world! This is a test corpus for BPE training."
    
    # Write to temporary file
    test_file = "tests/fixtures/tinystories_sample.txt"
    
    # Train BPE
    vocab, merges = train_bpe(
        input_path=test_file,
        vocab_size=10000,  # 1 special + 256 bytes + 43 merges
        special_tokens=["<|endoftext|>"]
    )
    
    print(f"\nVocabulary size: {len(vocab)}")
    print(f"Number of merges: {len(merges)}")
    
    # Show first few merges
    print(f"\nFirst 10 merges:")
    for i, (b1, b2) in enumerate(merges[:100]):
        try:
            char1 = b1.decode('utf-8', errors='replace')
            char2 = b2.decode('utf-8', errors='replace')
            print(f"  {i+1:2d}. ({b1}, {b2}) -> '{char1}' + '{char2}'")
        except:
            print(f"  {i+1:2d}. {(b1, b2)}")

    # Create tokenizer instance
    tokenizer = Tokenizer(vocab, merges, special_tokens=["<|endoftext|>"])
    print(tokenizer.encode("Hello world! This is a test corpus for BPE training. <|endoftext|> Today is a good day."))