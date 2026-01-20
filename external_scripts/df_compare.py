import pandas as pd
import argparse
import sys

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
    
    restricted_counts = restricted_df.value_counts()
    controlled_counts = controlled_df.value_counts()

    restricted_aligned = restricted_counts.reindex(controlled_counts.index, fill_value=0)

    is_subset = (controlled_counts <= restricted_aligned).all()
    return is_subset
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("restricted_file", type=str, help="Path to Dataframe containing the restricted access phenotype")
    parser.add_argument("controlled_file", type=str, help="Path to Dataframe containing the controlled phenotype file")
    
    args = parser.parse_args()
    try:
        print(compare(args.restricted_file, args.controlled_file))
    except (FileNotFoundError, pd.errors.ParserError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)