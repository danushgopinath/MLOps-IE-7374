import argparse

from joblib import dump
from sklearn.ensemble import RandomForestClassifier

from src.common import DEFAULT_OUT_DIR, load_splits, model_path


def train(version, out_dir=DEFAULT_OUT_DIR, n_estimators=100, max_depth=None, seed=0):
    """
    Trains a random forest on the training split and saves it as a new model version.
    Args:
        version (str): Version id, used in the file name (the workflow uses a timestamp).
        out_dir (str/Path): Folder that contains models/ and metrics/.
        n_estimators (int): Number of trees.
        max_depth (int/None): Maximum tree depth, None for no limit.
        seed (int): Random seed for the forest.
    Returns:
        Path: Where the model was saved.
    """
    X_train, y_train = load_splits()["train"]

    model = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, random_state=seed, n_jobs=-1
    )
    model.fit(X_train, y_train)

    path = model_path(out_dir, version)
    path.parent.mkdir(parents=True, exist_ok=True)
    dump(model, path, compress=3)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train a new model version.")
    parser.add_argument("--version", required=True, help="Version id, e.g. a timestamp")
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=0, help="0 means no limit")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)

    if args.n_estimators < 1 or args.max_depth < 0:
        parser.error("--n-estimators must be at least 1 and --max-depth cannot be negative")

    path = train(
        args.version,
        out_dir=args.out_dir,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth or None,
    )
    print(f"Trained model version {args.version} "
          f"(n_estimators={args.n_estimators}, max_depth={args.max_depth or None})")
    print(f"Saved to {path}")


if __name__ == "__main__":
    main()
