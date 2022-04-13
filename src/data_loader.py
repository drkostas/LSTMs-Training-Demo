import os
import numpy as np
import re
from typing import *
import pandas as pd
import pickle


model_path = os.path.join(os.path.dirname(__file__), '..', 'models')


class DataLoader:
    data_path: str = os.path.join(os.path.dirname(__file__), '..', 'data')
    raw_data_str: str
    clean_data_str: str
    clean_data_lst: List[str]
    encoded_data_lst: List[str]
    tokenized_data_np: np.ndarray
    dictionary: Dict[int, str]

    def __init__(self, file_name: str, window_size: int, stride: int):
        self.file_path = os.path.join(self.data_path, file_name)
        self.window_size = window_size
        self.stride = stride

    def load(self, n_rows: int = -1):
        raw_data_np = np.genfromtxt(self.file_path, dtype='str', delimiter='\n',
                                    max_rows=n_rows if n_rows != -1 else None)
        self.raw_data_str = ' '.join(raw_data_np.tolist())
        return self.raw_data_str

    def sanitize(self) -> List[str]:
        pattern = re.compile(r'[^a-z0-9 ]+')
        self.clean_data_str = pattern.sub('', self.raw_data_str.lower())
        self.clean_data_lst = list(pattern.sub('', self.raw_data_str.lower()))
        return self.clean_data_lst

    def encode(self) -> List[str]:
        vocabulary = set(self.clean_data_lst)
        self.dictionary = {letter: i for i, letter in enumerate(vocabulary)}
        self.encoded_data_lst = list(map(lambda x: self.dictionary[x], self.clean_data_lst))
        return self.encoded_data_lst

    def tokenize(self, encoded: bool = True) -> np.ndarray:
        if encoded:
            self.tokenized_data_np = self._tokenize(self.encoded_data_lst)
            return self.tokenized_data_np
        else:
            tokenized_unencoded_data_np = self._tokenize(self.clean_data_lst)
            return tokenized_unencoded_data_np

    def _tokenize(self, data_list: List) -> np.ndarray:
        tokenized_lst = []
        for letter in range(0, len(data_list) - self.window_size + 1, self.stride):
            tokenized_lst.append(data_list[letter:letter + self.window_size])
        return np.array(tokenized_lst)


def save_pickle(data, file_name: str, attr: str, task: str,
                protocol=pickle.HIGHEST_PROTOCOL):
    file_path = os.path.join(model_path, f'{attr}_attr', f'task_{task}')
    os.makedirs(file_path, exist_ok=True)
    file_path = os.path.join(file_path, file_name)
    with open(file_path, 'wb') as f:
        pickle.dump(data, f, protocol=protocol)


def load_pickle(file_name: str, attr: str, task: str):
    file_path = os.path.join(model_path, f'{attr}_attr', f'task_{task}', file_name)
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data
