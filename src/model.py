# model.py. baseline predictor (LSTM)
import torch # type: ignore
import torch.nn as nn # type: ignore

class SimpleLSTM(nn.Module):
    """
    Input: history sequence per agent (seq_len, 2)
    Output: predicted future flattened (future_len*2)
    """
    def __init__(self, input_dim=2, hidden_dim=64, num_layers=1, future_len=50):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.future_len = future_len
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, future_len * 2)
        )

    def forward(self, x):
        # x: (batch, seq, 2)
        out, _ = self.lstm(x)  # out: (batch, seq, hidden)
        # use last timestep
        last = out[:, -1, :]
        out = self.fc(last)
        out = out.view(-1, self.future_len, 2)
        return out
