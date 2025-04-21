from dataclasses import dataclass
import torch
from torch.utils.data import Dataset


@dataclass
class DatasetConfig:
    max_size: int = 100000
    batch_size: int = 512
    shuffle: bool = True
    num_workers: int = 4
    pin_memory: bool = True


class ReplayBufferDataset(Dataset):
    def __init__(self, config: DatasetConfig):
        super(ReplayBufferDataset, self).__init__()

        self.config = config
        self.buffer: list[
            tuple[
                torch.Tensor,  # state
                torch.Tensor,  # action
                torch.Tensor,  # reward
                torch.Tensor,  # next_state
                torch.Tensor,  # done
            ]
        ] = []

    def add(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
    ):
        if len(self.buffer) < self.config.max_size:
            self.buffer.append(
                (
                    torch.tensor(state, dtype=torch.float32),
                    torch.tensor(action, dtype=torch.float32),
                    torch.tensor([reward], dtype=torch.float32),
                    torch.tensor(next_state, dtype=torch.float32),
                    torch.tensor([float(done)], dtype=torch.float32),
                )
            )
        else:
            self.buffer.pop(0)
            self.buffer.append(
                (
                    torch.tensor(state, dtype=torch.float32),
                    torch.tensor(action, dtype=torch.float32),
                    torch.tensor([reward], dtype=torch.float32),
                    torch.tensor(next_state, dtype=torch.float32),
                    torch.tensor([float(done)], dtype=torch.float32),
                )
            )

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, idx):
        return self.buffer[idx]

    def create_dataloader(self):
        return torch.utils.data.DataLoader(
            self,
            batch_size=self.config.batch_size,
            shuffle=self.config.shuffle,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
        )
