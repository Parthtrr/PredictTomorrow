"""Load the official Kronos implementation and predict one future close."""

import subprocess
import sys
from pathlib import Path

import pandas as pd
import torch


class KronosModel:
    """Use frozen pretrained weights through the official inference interface."""

    def __init__(self, settings):
        """Load pinned source and weights once for the complete evaluation."""
        project = Path(__file__).resolve().parents[1]
        repository = project / settings["kronos_repo"]
        if not (repository / "model/kronos.py").exists():
            raise FileNotFoundError("Download the official Kronos source using the README setup steps.")
        revision = subprocess.check_output(
            ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
        ).strip()
        if revision != settings["kronos_revision"]:
            raise ValueError("The Kronos source revision does not match config.json.")
        subprocess.run(["git", "-C", str(repository), "diff", "--quiet"], check=True)

        # The official code imports its own top-level 'model' package.
        sys.path.insert(0, str(repository))
        from model import Kronos, KronosPredictor, KronosTokenizer

        torch.set_num_threads(settings["cpu_threads"])
        torch.manual_seed(settings["seed"])
        cache = project / ".cache/huggingface"
        tokenizer = KronosTokenizer.from_pretrained(
            settings["tokenizer_id"], revision=settings["tokenizer_revision"], cache_dir=cache
        )
        model = Kronos.from_pretrained(
            settings["model_id"], revision=settings["model_revision"], cache_dir=cache
        )

        # Disable dropout and gradients; there are no optimizer or training steps.
        tokenizer.eval()
        model.eval()
        tokenizer.requires_grad_(False)
        model.requires_grad_(False)
        self.predictor = KronosPredictor(model, tokenizer, device=settings["device"], max_context=512)
        self.sample_count = settings["sample_count"]

    def predict_close(self, history, target_date):
        """Predict one close from past OHLCV and the known target date only."""
        # day_change_pct is not an input supported by the pretrained architecture.
        values = history[["open", "high", "low", "close", "volume"]].reset_index(drop=True)
        past_dates = history["date"].reset_index(drop=True)
        future_dates = pd.Series([pd.Timestamp(target_date)])
        with torch.inference_mode():
            forecast = self.predictor.predict(
                df=values,
                x_timestamp=past_dates,
                y_timestamp=future_dates,
                pred_len=1,
                T=1.0,
                top_p=0.9,
                sample_count=self.sample_count,
                verbose=False,
            )
        return float(forecast.iloc[0]["close"])
