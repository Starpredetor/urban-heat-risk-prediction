"""Step 6: the ConvLSTM.

Two ideas combined:

  Conv  (convolution)  looks at a pixel together with its neighbours, so it
                       understands shapes -- a park, a row of buildings.
  LSTM  (memory)       carries information from one year to the next, so it
                       can notice a trend rather than judging each year alone.

A ConvLSTM does both at once: it remembers maps, not just numbers.

One design choice worth explaining: the model predicts the CHANGE from last
year, not the temperature itself. Most of a city's heat pattern is the same
every year -- the airport is always hot, the park is always cool. If the model
predicted the temperature directly it could score well by copying last year and
learning nothing. Making it predict only the change forces it to earn its keep.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    """One year's worth of memory update."""

    def __init__(self, in_channels: int, memory_channels: int):
        super().__init__()
        self.memory_channels = memory_channels
        # One convolution computes all four LSTM gates at once; we split them
        # apart afterwards. padding=1 keeps the map the same size.
        self.conv = nn.Conv2d(in_channels + memory_channels, 4 * memory_channels, 3, padding=1)
        nn.init.zeros_(self.conv.bias)
        # Starting the "forget" gate at 1 means "remember everything" initially,
        # which stops the memory being wiped before it learns anything.
        with torch.no_grad():
            self.conv.bias[memory_channels: 2 * memory_channels].fill_(1.0)

    def forward(self, x, state):
        short_term, long_term = state
        gates = self.conv(torch.cat([x, short_term], dim=1))
        keep_new, forget_old, output, candidate = torch.split(gates, self.memory_channels, dim=1)

        keep_new = torch.sigmoid(keep_new)
        forget_old = torch.sigmoid(forget_old)
        output = torch.sigmoid(output)
        candidate = torch.tanh(candidate)

        long_term = forget_old * long_term + keep_new * candidate
        short_term = output * torch.tanh(long_term)
        return short_term, long_term


class HeatForecaster(nn.Module):
    """Reads several years of maps, predicts next year's temperature map."""

    def __init__(self, n_layers: int = 4, memory: int = 32):
        super().__init__()
        self.cell = ConvLSTMCell(n_layers, memory)
        self.readout = nn.Sequential(
            nn.Conv2d(memory, memory, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(memory, 1, 1),
        )
        # Start with an almost-zero change, i.e. "next year looks like this
        # year", then learn how it actually differs. Small rather than exactly
        # zero, because exactly zero would stop the model learning at all.
        nn.init.normal_(self.readout[-1].weight, std=1e-3)
        nn.init.zeros_(self.readout[-1].bias)

    def forward(self, years):
        """years: (batch, n_years, n_layers, height, width)"""
        assert years.dim() == 5, f"expected 5 dimensions, got {tuple(years.shape)}"
        batch, n_years, _, height, width = years.shape

        blank = torch.zeros(batch, self.cell.memory_channels, height, width,
                            device=years.device, dtype=years.dtype)
        state = (blank, blank.clone())

        for i in range(n_years):
            state = self.cell(years[:, i], state)

        change = self.readout(state[0])
        last_year_temperature = years[:, -1, 0:1]     # layer 0 is lst
        return last_year_temperature + change
