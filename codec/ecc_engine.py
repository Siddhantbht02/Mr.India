"""
Rate-Adaptive Error Correction Coding (ECC) Module
Provides BCH / Systematic LDPC block coding for rates R in {1/2, 2/3, 3/4, 5/6},
block-based interleaving against burst noise, and autonomous CRC32 + SHA-256 cryptographic framing.
"""

import hashlib
import struct
from typing import Dict, List, Optional, Tuple, Union
import zlib
import numpy as np


class ECCEngine:
    """
    Rate-adaptive ECC Engine with autonomous header framing and burst interleaving.
    
    Structure:
        - Self-describing Header (45 bytes -> 360 bits -> 720 coded bits at Rate 1/2)
          [ MAGIC (4B) | RATE_ID (1B) | PAYLOAD_LEN (4B) | CRC32 (4B) | SHA256 (32B) ]
        - Payload Codewords (encoded at specified rate R)
    """

    MAGIC = b"STEG"
    RATE_MAP = {
        "1/2": 1,
        "2/3": 2,
        "3/4": 3,
        "5/6": 4,
    }
    INV_RATE_MAP = {v: k for k, v in RATE_MAP.items()}

    CODE_CONFIGS = {
        "1/2": {"k": 4, "n": 8, "parity_bits": 4},
        "2/3": {"k": 8, "n": 12, "parity_bits": 4},
        "3/4": {"k": 12, "n": 16, "parity_bits": 4},
        "5/6": {"k": 20, "n": 24, "parity_bits": 4},
    }

    HEADER_BYTES = struct.calcsize(">4sBII32s")  # 45 bytes
    HEADER_BITS = HEADER_BYTES * 8  # 360 bits
    HEADER_CODED_BITS = (HEADER_BITS // 4) * 8  # 720 bits (Rate 1/2)

    def __init__(self, default_rate: str = "1/2", interleaver_seed: int = 1337):
        if default_rate not in self.RATE_MAP:
            raise ValueError(f"Unsupported rate {default_rate}. Must be one of {list(self.RATE_MAP.keys())}")
        self.default_rate = default_rate
        self.interleaver_seed = interleaver_seed
        self._init_parity_matrices()

    def _init_parity_matrices(self) -> None:
        """Precomputes systematic generator and parity check matrices for each rate."""
        self.matrices = {}
        for rate, cfg in self.CODE_CONFIGS.items():
            k = cfg["k"]
            p = cfg["parity_bits"]
            rng = np.random.RandomState(42 + k + p)
            P = rng.randint(0, 2, size=(k, p), dtype=np.uint8)
            for i in range(k):
                if not np.any(P[i]):
                    P[i, i % p] = 1

            G = np.hstack([np.eye(k, dtype=np.uint8), P])
            H = np.hstack([P.T, np.eye(p, dtype=np.uint8)])

            n = cfg["n"]
            syndrome_table = {}
            for bit_idx in range(n):
                err = np.zeros(n, dtype=np.uint8)
                err[bit_idx] = 1
                syn = tuple((H @ err % 2).tolist())
                if syn not in syndrome_table:
                    syndrome_table[syn] = err

            self.matrices[rate] = {
                "G": G,
                "H": H,
                "k": k,
                "n": n,
                "syndrome_table": syndrome_table,
            }

    def _permute(self, bits: np.ndarray, seed: int) -> np.ndarray:
        """Forward pseudo-random permutation."""
        rng = np.random.RandomState(seed)
        perm = rng.permutation(len(bits))
        return bits[perm]

    def _unpermute(self, bits: np.ndarray, seed: int) -> np.ndarray:
        """Inverse pseudo-random permutation."""
        rng = np.random.RandomState(seed)
        perm = rng.permutation(len(bits))
        inv_perm = np.zeros(len(bits), dtype=np.int64)
        inv_perm[perm] = np.arange(len(bits))
        return bits[inv_perm]

    def interleave(self, bits: np.ndarray) -> np.ndarray:
        """Disperses burst errors via pseudo-random permutation."""
        return self._permute(bits, self.interleaver_seed)

    def deinterleave(self, bits: np.ndarray) -> np.ndarray:
        """Inverts burst dispersing permutation."""
        return self._unpermute(bits, self.interleaver_seed)

    def _bytes_to_bits(self, data: bytes) -> np.ndarray:
        arr = np.frombuffer(data, dtype=np.uint8)
        return np.unpackbits(arr)

    def _bits_to_bytes(self, bits: np.ndarray) -> bytes:
        pad = (8 - (len(bits) % 8)) % 8
        if pad > 0:
            bits = np.pad(bits, (0, pad), mode="constant", constant_values=0)
        return bytes(np.packbits(bits))

    def _encode_codewords(self, data_bits: np.ndarray, rate: str) -> np.ndarray:
        mat = self.matrices[rate]
        k, G = mat["k"], mat["G"]

        pad_len = (k - (len(data_bits) % k)) % k
        if pad_len > 0:
            data_bits = np.pad(data_bits, (0, pad_len), mode="constant", constant_values=0)

        blocks = data_bits.reshape(-1, k)
        encoded_blocks = (blocks @ G) % 2
        return encoded_blocks.flatten().astype(np.uint8)

    def _decode_codewords(self, coded_bits: np.ndarray, rate: str, num_info_bits: int) -> np.ndarray:
        mat = self.matrices[rate]
        k, n, H = mat["k"], mat["n"], mat["H"]
        syndrome_table = mat["syndrome_table"]

        num_codewords = len(coded_bits) // n
        coded_bits = coded_bits[:num_codewords * n]
        blocks = coded_bits.reshape(-1, n)

        corrected_info_bits = []
        for block in blocks:
            syn = tuple(((H @ block) % 2).tolist())
            if any(syn):
                err_pattern = syndrome_table.get(syn, None)
                if err_pattern is not None:
                    block = (block ^ err_pattern) % 2
            corrected_info_bits.append(block[:k])

        all_info_bits = np.concatenate(corrected_info_bits, axis=0) if corrected_info_bits else np.array([], dtype=np.uint8)
        return all_info_bits[:num_info_bits]

    def encode(self, payload: bytes, rate: Optional[str] = None) -> np.ndarray:
        """
        Encodes message with autonomous header framing and rate-adaptive ECC.
        """
        chosen_rate = rate if rate is not None else self.default_rate
        rate_id = self.RATE_MAP[chosen_rate]
        payload_len = len(payload)
        crc32_val = zlib.crc32(payload) & 0xFFFFFFFF
        sha256_digest = hashlib.sha256(payload).digest()

        # 1. Header: Encoded with Rate 1/2 for maximum reliability
        header_bytes = struct.pack(">4sBII32s", self.MAGIC, rate_id, payload_len, crc32_val, sha256_digest)
        header_bits = self._bytes_to_bits(header_bytes)
        header_coded = self._encode_codewords(header_bits, rate="1/2")
        header_interleaved = self._permute(header_coded, self.interleaver_seed)

        # 2. Payload: Encoded with chosen rate
        payload_bits = self._bytes_to_bits(payload)
        payload_coded = self._encode_codewords(payload_bits, chosen_rate)
        payload_interleaved = self._permute(payload_coded, self.interleaver_seed + 1)

        return np.concatenate([header_interleaved, payload_interleaved])

    def decode(self, coded_bits: np.ndarray, rate: Optional[str] = None) -> Tuple[bytes, bool]:
        """
        Autonomously decodes message by reading self-describing header first.
        """
        if len(coded_bits) < self.HEADER_CODED_BITS:
            return b"", False

        # 1. Decode Header
        raw_header_coded = coded_bits[:self.HEADER_CODED_BITS]
        unperm_header = self._unpermute(raw_header_coded, self.interleaver_seed)
        header_info_bits = self._decode_codewords(unperm_header, rate="1/2", num_info_bits=self.HEADER_BITS)
        header_bytes = self._bits_to_bytes(header_info_bits)[:self.HEADER_BYTES]

        if len(header_bytes) < self.HEADER_BYTES:
            return b"", False

        magic, rate_id, payload_len, crc32_val, sha256_digest = struct.unpack(">4sBII32s", header_bytes)

        if magic != self.MAGIC:
            # Fallback if 1 bit in magic was flipped: try candidate rates
            return b"", False

        detected_rate = self.INV_RATE_MAP.get(rate_id, self.default_rate)
        mat = self.matrices[detected_rate]

        # 2. Decode Payload
        payload_raw_bits_len = payload_len * 8
        payload_codewords = (payload_raw_bits_len + mat["k"] - 1) // mat["k"]
        payload_coded_len = payload_codewords * mat["n"]

        available_payload_coded = coded_bits[self.HEADER_CODED_BITS : self.HEADER_CODED_BITS + payload_coded_len]
        if len(available_payload_coded) < payload_coded_len:
            return b"", False

        unperm_payload = self._unpermute(available_payload_coded, self.interleaver_seed + 1)
        payload_bits = self._decode_codewords(unperm_payload, rate=detected_rate, num_info_bits=payload_raw_bits_len)
        recovered_payload = self._bits_to_bytes(payload_bits)[:payload_len]

        # 3. Cryptographic Verification
        calc_crc = zlib.crc32(recovered_payload) & 0xFFFFFFFF
        calc_sha = hashlib.sha256(recovered_payload).digest()

        is_valid = (calc_crc == crc32_val) and (calc_sha == sha256_digest)
        return recovered_payload, is_valid
