"""
full QRGRU implementation
"""
import torch
import torch.nn as nn

class GateUnit(nn.Module):
    """
    Controls that new information can be added to old representations
    """

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.gate = nn.Sequential(nn.Linear(self.hidden_size * 3, self.hidden_size * 2),
                                  nn.Sigmoid())
        self.hidden_trans = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.Tanh()
        )

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, message: torch.Tensor, query_r: torch.Tensor, hidden_state: torch.Tensor):
        """
        Entity representations are updated via a GRU-like gating mechanism

        :param message: message[batch_size,input_size]
        :param query_r: query_r[batch_size,input_size]
        :param hidden_state: if it is none,it will be allocated a zero tensor hidden state
        :return:
        """
        update_value, reset_value = self.gate(torch.cat([message, query_r, hidden_state], dim=1)).chunk(2, dim=1)
        hidden_candidate = self.hidden_trans(torch.cat([message, reset_value * hidden_state], dim=1))
        hidden_state = (1 - update_value) * hidden_state + update_value * hidden_candidate
        return hidden_state
