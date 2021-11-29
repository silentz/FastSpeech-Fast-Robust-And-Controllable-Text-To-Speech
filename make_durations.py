import tqdm
import torch
from torch.utils.data import DataLoader
from src.aligner import GraphemeAligner
from src.dataset import LJSpeechDataset
from src.collate import LJSpeechCollator


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    df = LJSpeechDataset('data/')
    loader =  DataLoader(df, batch_size=32, shuffle=False, pin_memory=True, collate_fn=LJSpeechCollator())

    aligner = GraphemeAligner()
    aligner.to(device)

    all_durations = []

    for batch in tqdm.tqdm(loader):
        batch = batch.to(device)
        durations = aligner(batch.waveform, batch.waveform_length, batch.transcript)
        all_durations.append(durations)

    max_length = max(x.shape[1] for x in all_durations)
    result = torch.zeros(len(df), max_length, device=device)
    start_idx = 0

    for tensors in all_durations:
        cur_len = tensors.shape[1]
        result[start_idx:start_idx+32, :cur_len] = tensors
        start_idx += 32

    torch.save(result, './checkpoints/durations.pt')

