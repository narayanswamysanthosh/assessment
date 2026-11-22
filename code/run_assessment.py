import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from synthetic_data_generator.data_generator import generate_hospital_data
from train_and_score import run_pipeline


def main() -> None:
    data_dir = ROOT_DIR / "synthetic_data_generator" / "data"
    generate_hospital_data(output_dir=str(data_dir))
    run_pipeline(
        train_path=str(data_dir / "train_admissions.csv"),
        holdout_path=str(data_dir / "holdout_admissions.csv"),
        predictions_path=str(ROOT_DIR / "predictions.csv"),
        model_dir=str(ROOT_DIR / "code" / "artifacts"),
    )


if __name__ == "__main__":
    main()
