import pandas as pd
import argparse
import sys
from pathlib import Path


def compare(restricted, controlled):
    """
    Compares two phenotype dataframs and check if one is a subset of the other.
    (ie. all entries of the controlled dataframe is present in the restricted).
    
    Args:
        restricted: Path to the file with restricted access phenotypes.
        controlled: Path to the file with controlled access phenotypes.
    
    Returns:
        True if the controlled dataframe's row counts are a subset of
        the restricted one, False otherwise.
    """
    restricted_df = pd.read_csv(restricted, delimiter="\t")
    controlled_df = pd.read_csv(controlled, delimiter="\t")
    
    controlled_df = controlled_df[restricted_df.columns]

    restricted_df = restricted_df.astype(str)
    controlled_df = controlled_df.astype(str)

    restricted_counts = restricted_df.value_counts()
    controlled_counts = controlled_df.value_counts()

    restricted_aligned = restricted_counts.reindex(controlled_counts.index, fill_value=0)

    is_subset = (controlled_counts <= restricted_aligned).all()
    return is_subset
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("restricted_folder", type=str, help="Path to Dataframe containing the restricted access phenotype")
    parser.add_argument("controlled_folder", type=str, help="Path to Dataframe containing the controlled phenotype file")
    
    args = parser.parse_args()
    
    all_controlled_tsv = [str(p) for p in Path(args.controlled_folder).rglob('*.tsv')]
    #print(all_controlled_tsv)
    for tsv_path in all_controlled_tsv:
        restricted_tsv_path = tsv_path.replace(args.controlled_folder, args.restricted_folder)
        try:
            is_same = (compare(restricted_tsv_path, tsv_path))
            if not is_same:
                print(f"{tsv_path} does not match with {restricted_tsv_path}")
        except (FileNotFoundError, pd.errors.ParserError) as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)