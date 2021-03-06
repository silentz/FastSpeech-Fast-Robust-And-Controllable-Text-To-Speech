import torch
import torchaudio
from torch.utils.data import Dataset


class LJSpeechDataset(torchaudio.datasets.LJSPEECH):

    def __init__(self, root):
        super().__init__(root=root, download=True)
        self._tokenizer = torchaudio.pipelines.TACOTRON2_GRIFFINLIM_CHAR_LJSPEECH.get_text_processor()

    def __getitem__(self, index: int):
        waveform, _, _, transcript = super().__getitem__(index)
        waveforn_length = torch.tensor([waveform.shape[-1]]).int()

        tokens, token_lengths = self._tokenizer(transcript)
        return waveform, waveforn_length, transcript, tokens, token_lengths

    def decode(self, tokens, lengths):
        result = []
        for tokens_, length in zip(tokens, lengths):
            text = "".join([
                self._tokenizer.tokens[token]
                for token in tokens_[:length]
            ])
            result.append(text)
        return result


class AdvancedLJSpeechDataset(LJSpeechDataset):

    def __init__(self, root, durations_path):
        super().__init__(root)
        self.durations = torch.load(durations_path, map_location='cpu')

    def __getitem__(self, idx: int):
        from_parent = super().__getitem__(idx)
        duration = self.durations[idx]
        return *from_parent, duration


class TestLJSpeechDataset(LJSpeechDataset):

    def __getitem__(self, idx: int):
        _, _, text, tokens, tokens_len = super().__getitem__(idx)
        return tokens.squeeze(dim=0), tokens_len, text


class OverfitDataset(Dataset):

    def __init__(self, dataset: Dataset,
                       num_samples: int,
                       length: int):
        self.dataset = dataset
        self.num_samples = num_samples
        self.length = length

    def __getitem__(self, idx: int):
        return self.dataset[idx % self.num_samples]

    def __len__(self):
        return self.length


class PartialDataset(Dataset):

    def __init__(self, dataset: Dataset,
                       start_idx: int,
                       finish_idx: int):
        # interval has form: [start_idx; finish_idx)
        self.dataset = dataset
        self.start_idx = start_idx
        self.finish_idx = finish_idx

    def __getitem__(self, idx: int):
        return self.dataset[idx + self.start_idx]

    def __len__(self):
        return self.finish_idx - self.start_idx


class TestDataset(Dataset):

    def __init__(self):
        self._lines = [
            'A defibrillator is a device that gives a high energy electric' \
                ' shock to the heart of someone who is in cardiac arrest',

            'Massachusetts Institute of Technology may be best known for its' \
                ' math, science and engineering education',

            'Wasserstein distance or Kantorovich Rubinstein metric is a distance' \
                ' function defined between probability distributions on a given metric space'
        ]

        self._tokenizer = torchaudio.pipelines.TACOTRON2_GRIFFINLIM_CHAR_LJSPEECH.get_text_processor()

    def __len__(self):
        return len(self._lines)

    def __getitem__(self, idx: int):
        text = self._lines[idx]
        tokens, token_lengths = self._tokenizer(text)
        return tokens.squeeze(dim=0), token_lengths, text
