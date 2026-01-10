from __future__ import annotations

import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # return
    vocab: dict[int, bytes] = {} # index to byte sequence
    merges: list[tuple[bytes, bytes]] = [] # list of byte pair merges, 만들어진 순서대로 저장, ex) [(b'a', b'b'), (b'ab', b'c')]
    
    ### Initialize the vocabulary with special tokens ###
    for i in range(0, 256):
        vocab[i] = bytes([i]) # [0] = b'\x00', [1] = b'\x01', ..., [255] = b'\xff'

    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode("utf-8")

    with open(input_path, "rb") as f:
        num_processes = 4

        # bytes 형태로 special token 변환
        special_tokens_bytes = [token.encode("utf-8") for token in special_tokens]
        boundaries = find_chunk_boundaries(f, num_processes, special_tokens_bytes) # 균등으로 간격으로 나누고 다음 [endoftext] 토큰에서 멈춤

        # frequency table: dict[tuple[bytes], int]
        # 각 단어를 byte로 변환 후, freq_table에 빈도 수 추가
        freq_table: dict[tuple[bytes], int] = {}

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.

        # Update: frequency table by reading each chunk
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")

            # 추가: Windows 줄바꿈을 Unix 스타일로 정규화
            chunk = chunk.replace("\r\n", "\n")

            # Run pre-tokenization on your chunk and store the counts for each pre-token        
            # split on special tokens (e.g <|endoftext|>)
            escape_tokens = [re.escape(token) for token in special_tokens] # 각 special token을 문자열로 취급
            pattern = "|".join(escape_tokens) # speical token을 OR로 찾기
            removed_special_tokens = re.split(pattern, chunk) # speical toeken 기준으로 split

            # pre-tokenization pattern using re.finditer
            for segment in removed_special_tokens: # special token 기준으로 나눠진 segment들에 대해
                for match in re.finditer(PAT, segment): # 단어 별로 분해
                    # tuple로 1byte씩 쪼개서 저장
                    # 각각을 bytes로 저장해줘야됨!!
                    token_bytes = tuple(bytes([b]) for b in match.group(0).encode("utf-8"))
                    freq_table[token_bytes] = freq_table.get(token_bytes, 0) + 1

        # vocab_size까지 도달할 때까지 byte pair merge 수행
        # Merge special tokens into the vocabulary with high frequency until the vocab size is under vocab_size
        # 합친 두 byte를 새로운 byte로 취급하여 freq_table을 수정

        while len(vocab) < vocab_size:
            adjacent_byte_pair_freq: dict[tuple[bytes, bytes], int] = {}
            
            # 빈도수 table을 순회하면서 인접한 byte pair의 빈도 수 계산
            for token_bytes, freq in freq_table.items():
                # 길이가 1이면 pass
                if len(token_bytes) < 2:
                    continue
                # 인접한 byte pair 추출
                # token_bytes가 b'abc'라면 (b'a', b'b'), (b'b', b'c') 추출
                for i in range(len(token_bytes) - 1):
                    byte_pair = (token_bytes[i], token_bytes[i + 1])
                    adjacent_byte_pair_freq[byte_pair] = adjacent_byte_pair_freq.get(byte_pair, 0) + freq

            # count가 높고 사전순으로 뒤에 오는 쌍 추출
            max_pair, _ = max(adjacent_byte_pair_freq.items(), key=lambda x: (x[1], x[0])) 
            
            # update merge table 
            merges.append(max_pair) # (b'a', b'b') 형태로 저장

            # update vocab table
            vocab[len(vocab)] = max_pair[0] + max_pair[1] # 새로운 byte 추가, ex) [256] = b'ab'

            # max_pair에 해당하는 byte 쌍을 새로운 쌍으로 치환
            new_freq_table: dict[tuple[bytes], int] = {}
            for token_tuple, freq in freq_table.items():
                new_token_tuple = []
                i = 0
                while i < len(token_tuple):
                    if i < len(token_tuple) - 1 and (token_tuple[i], token_tuple[i+1]) == max_pair:
                        new_token_tuple.append(max_pair[0] + max_pair[1]) # 두 bytes를 합쳐 하나로
                        i += 2
                    else:
                        new_token_tuple.append(token_tuple[i])
                        i += 1
                new_freq_table[tuple(new_token_tuple)] = freq

            freq_table = new_freq_table
    
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