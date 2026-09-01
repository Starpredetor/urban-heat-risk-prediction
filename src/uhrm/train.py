"""Step 7: train the model, and check it against simple rivals.

The most important idea here is the COMPARISON. A deep learning model is only
worth using if it beats the obvious simple answers. We always test two:

    "no change"      next year is exactly the same as this year
    "straight line"  draw a line through recent years and extend it

If the ConvLSTM cannot beat both, we say so. That is a real result, not a
failure to hide.

The score is RMSE in degrees Celsius: on average, how many degrees out is the
prediction. Lower is better.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import torch
from torch.utils.data import DataLoader

from uhrm.dataset import HeatPatches
from uhrm.model import HeatForecaster

log = logging.getLogger(__name__)


def pick_device() -> str:
    """Use the NVIDIA graphics card if there is one, otherwise the processor."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def score(predictions: np.ndarray, answers: np.ndarray, degrees_per_unit: float) -> dict:
    """Turn the model's error back into degrees Celsius."""
    error = (predictions - answers) * degrees_per_unit
    answers_c = answers * degrees_per_unit

    spread = float(np.var(answers_c))
    return {
        "rmse_c": float(np.sqrt(np.mean(error**2))),   # typical error in degrees
        "mae_c": float(np.mean(np.abs(error))),        # average error in degrees
        # 1.0 = perfect, 0.0 = no better than always guessing the average
        "r2": float(1 - np.mean(error**2) / spread) if spread > 0 else float("nan"),
    }


def collect(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    predictions, answers = [], []
    with torch.no_grad():
        for past, answer in loader:
            guess = model(past.to(device)).cpu().numpy()
            predictions.append(guess)
            answers.append(answer.numpy())
    return np.concatenate(predictions), np.concatenate(answers)


def train_model(train_set, val_set, settings, device) -> HeatForecaster:
    torch.manual_seed(settings.seed)
    np.random.seed(settings.seed)

    model = HeatForecaster(n_layers=train_set.data.shape[1]).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=settings.learning_rate)

    train_loader = DataLoader(train_set, batch_size=settings.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=settings.batch_size)
    degrees = train_set.scaler.lst_std

    best_score, best_weights, patience = float("inf"), None, 0

    for epoch in range(1, settings.epochs + 1):
        model.train()
        total = 0.0
        for past, answer in train_loader:
            past, answer = past.to(device), answer.to(device)
            # Average absolute error, not squared: leftover cloud edges create
            # a few wild pixels, and squaring would let them dominate.
            loss = (model(past) - answer).abs().mean()
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            total += loss.item()

        val = score(*collect(model, val_loader, device), degrees)["rmse_c"]
        log.info("epoch %2d   training loss %.4f   validation error %.2f C",
                 epoch, total / len(train_loader), val)

        if val < best_score:
            best_score, patience = val, 0
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            # Stop once it stops improving, so it does not start memorising.
            if patience >= 10:
                log.info("no improvement for 10 epochs, stopping early")
                break

    if best_weights:
        model.load_state_dict(best_weights)
    return model


def no_change_baseline(dataset, degrees) -> dict:
    """Predict that next year looks exactly like this year."""
    guesses, answers = [], []
    for i in range(len(dataset)):
        _, answer = dataset[i]
        guesses.append(dataset.last_known_temperature(i))
        answers.append(answer.numpy()[0])
    return score(np.array(guesses), np.array(answers), degrees)


def straight_line_baseline(dataset, degrees) -> dict:
    """Fit a line through the past few years and extend it one year."""
    history = dataset.settings.years_in
    steps = np.arange(history, dtype=np.float32)
    centred = steps - steps.mean()
    denominator = float((centred**2).sum())

    guesses, answers = [], []
    for i in range(len(dataset)):
        past, answer = dataset[i]
        series = past[:, dataset.lst].numpy()               # temperature over time
        average = series.mean(axis=0)
        slope = np.tensordot(centred, series - average, axes=(0, 0)) / denominator
        guesses.append(average + slope * (history - steps.mean()))
        answers.append(answer.numpy()[0])
    return score(np.array(guesses), np.array(answers), degrees)


def run(cube, settings, out_dir) -> dict:
    """Train, test, compare against both baselines, and save everything."""
    from uhrm.dataset import Scaler

    device = pick_device()
    log.info("training on: %s", "NVIDIA GPU" if device == "cuda" else "processor (no GPU found)")

    n_train_years = settings.years.index(max(settings.years_of("train"))) + 1
    scaler = Scaler.fit(cube["data"].values, n_train_years)

    sets = {name: HeatPatches(cube, settings, name, scaler)
            for name in ("train", "val", "test")}
    for name, ds in sets.items():
        log.info("%-5s : %4d examples, predicting years %s", name, len(ds), ds.target_years)

    model = train_model(sets["train"], sets["val"], settings, device)

    degrees = scaler.lst_std
    test_loader = DataLoader(sets["test"], batch_size=settings.batch_size)
    results = {
        "convlstm": score(*collect(model, test_loader, device), degrees),
        "no_change": no_change_baseline(sets["test"], degrees),
        "straight_line": straight_line_baseline(sets["test"], degrees),
    }
    results["convlstm_wins"] = bool(
        results["convlstm"]["rmse_c"] < results["no_change"]["rmse_c"]
        and results["convlstm"]["rmse_c"] < results["straight_line"]["rmse_c"]
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "model.pt")
    scaler.save(out_dir / "scaler.json")
    (out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    report(results)
    return results


def report(results: dict) -> None:
    log.info("")
    log.info("RESULTS on years the model never saw (lower error is better)")
    log.info("  %-16s %10s %10s %8s", "method", "error (C)", "avg (C)", "R2")
    names = {"convlstm": "ConvLSTM", "no_change": "no change",
             "straight_line": "straight line"}
    for key, label in names.items():
        m = results[key]
        log.info("  %-16s %10.2f %10.2f %8.2f", label, m["rmse_c"], m["mae_c"], m["r2"])
    log.info("")
    log.info("  Does ConvLSTM beat both simple methods?  %s",
             "YES" if results["convlstm_wins"] else "NO")
