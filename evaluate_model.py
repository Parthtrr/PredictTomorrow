"""Run the configured pretrained Kronos evaluation and save its results."""

import argparse
import json
from pathlib import Path

import torch

from config import load_config
from dao.daily_price_dao import DailyPriceDao
from dao.elasticsearch_client import create_client
from datasets.stock_dataset import StockDataset
from model_train.kronos_evaluator import KronosEvaluator
from model_train.kronos_model import KronosModel


def main():
    """Evaluate the requested date range; default dates come from config.json."""
    parser = argparse.ArgumentParser(description="Evaluate pretrained Kronos without training.")
    parser.add_argument("--model-start-date")
    parser.add_argument("--model-end-date")
    args = parser.parse_args()
    config = load_config()
    settings = config["model_train"]
    start_date = args.model_start_date or settings["model_start_date"]
    end_date = args.model_end_date or settings["model_end_date"]

    client = create_client(config["elasticsearch"])
    try:
        storage = config["elasticsearch"]
        dao = DailyPriceDao(client, storage["index_name"], storage["batch_size"])
        dataset = StockDataset(dao, config["data"]["ticker"], config["datasets"])
        print("Loading pretrained Kronos weights...", flush=True)
        model = KronosModel(settings)
        evaluator = KronosEvaluator(dataset, model, settings)
        summary, predictions = evaluator.evaluate(start_date, end_date)

        # Record the exact weights, source, and random seed with the metrics.
        summary["ticker"] = config["data"]["ticker"]
        summary["model_settings"] = settings.copy()
        summary["model_settings"]["model_start_date"] = start_date
        summary["model_settings"]["model_end_date"] = end_date
        summary["torch_version"] = torch.__version__
        summary["evaluation_note"] = "Pretraining overlap with these historical dates is unknown."
        output_dir = Path(__file__).resolve().parent / settings["output_dir"]
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = "kronos_" + start_date + "_" + end_date
        predictions.to_csv(output_dir / (stem + "_predictions.csv"), index=False)
        report_path = output_dir / (stem + "_metrics.json")
        report_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")

        print(f"MAE: {summary['mae']:.4f} price units")
        print(f"MAPE: {summary['mape_pct']:.4f}%")
        print(f"Directional error: {summary['directional_error_pct']:.4f}%")
        print("Metrics report:", report_path)
    finally:
        # Evaluation only reads Elasticsearch; no predictions are written into it.
        client.close()


if __name__ == "__main__":
    main()
