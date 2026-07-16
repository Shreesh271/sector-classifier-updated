"""Batch pipeline: given a list of organization names, look up each
organization's definition on Wikipedia, predict its Sector and Subsector,
and write everything to an output Excel file.

Usage:
    python batch_predict.py --input organizations.xlsx --output results.xlsx
    python batch_predict.py --input organizations.xlsx --column "Org Name"
"""

import argparse
import os
import sys

import pandas as pd

from predict import predict
from utils import fetch_definition_from_web

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Fetch definitions from Wikipedia and predict Sector/Subsector "
        "for a list of organization names."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input Excel file containing organization names.",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(BASE_DIR, "batch_results.xlsx"),
        help="Path to save the output Excel file (default: batch_results.xlsx).",
    )
    parser.add_argument(
        "--column",
        default=None,
        help="Name of the column containing organization names. "
        "If omitted, the first column is used.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Seconds to wait between Wikipedia requests (default: 0.5).",
    )
    return parser.parse_args()


def load_organization_names(path: str, column: str = None) -> pd.Series:
    """Load organization names from an Excel file.

    Args:
        path: Path to the input Excel file.
        column: Column name containing organization names. Uses the first
            column if not provided.

    Returns:
        Series of organization names with blanks removed.

    Raises:
        FileNotFoundError: If the input file does not exist.
        ValueError: If the specified column is not found.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")

    df = pd.read_excel(path)
    if df.empty:
        raise ValueError("Input file contains no data.")

    if column is None:
        column = df.columns[0]
    elif column not in df.columns:
        raise ValueError(
            f"Column '{column}' not found. Available columns: {list(df.columns)}"
        )

    names = df[column].dropna().astype(str).str.strip()
    names = names[names != ""]
    return names.reset_index(drop=True)


def process_organizations(names: pd.Series, delay: float = 0.5) -> pd.DataFrame:
    """Fetch definitions and predict Sector/Subsector for each organization.

    Args:
        names: Series of organization names.
        delay: Seconds to wait between Wikipedia requests.

    Returns:
        DataFrame with columns: Organization_Name, Definition, Source,
        Sector, Subsector.
    """
    records = []
    total = len(names)

    for idx, name in enumerate(names, start=1):
        print(f"[{idx}/{total}] Looking up: {name}")

        result = fetch_definition_from_web(name, delay=delay)
        definition = result["definition"]
        source = result["source"]

        sector = ""
        subsector = ""

        if definition:
            try:
                prediction = predict(definition)
                sector = prediction["sector"]
                subsector = prediction["subsector"]
            except (ValueError, FileNotFoundError) as exc:
                print(f"  Warning: prediction failed for '{name}': {exc}")
        else:
            print(f"  Warning: no definition found for '{name}' (source={source})")

        records.append(
            {
                "Organization_Name": name,
                "Definition": definition,
                "Source": source,
                "Sector": sector,
                "Subsector": subsector,
            }
        )

    return pd.DataFrame(records)


def main() -> None:
    """Run the batch pipeline end to end."""
    args = parse_args()

    print("Loading organization names...")
    names = load_organization_names(args.input, args.column)
    print(f"Loaded {len(names)} organization names.")

    print("Fetching definitions and predicting Sector/Subsector...")
    results_df = process_organizations(names, delay=args.delay)

    results_df.to_excel(args.output, index=False)
    print(f"\nDone. Results saved to: {args.output}")

    not_found = (results_df["Source"] == "not_found").sum()
    errors = (results_df["Source"] == "error").sum()
    if not_found or errors:
        print(
            f"Note: {not_found} organization(s) had no Wikipedia match, "
            f"{errors} had a network/request error. These rows have empty "
            f"Sector/Subsector."
        )


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        sys.exit(1)
