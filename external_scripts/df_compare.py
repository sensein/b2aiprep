import pandas as pd
import argparse
import sys
from pathlib import Path


def compare(registered, controlled):
    """
    Compares two phenotype dataframs and check if one is a subset of the other 
    and whether the columns match.
    (ie. all entries of the controlled dataframe is present in the registered).
    
    Args:
        registered: Path to the file with registered access phenotypes.
        controlled: Path to the file with controlled access phenotypes.
    
    Returns:
        True if the controlled dataframe's row counts are a subset of
        the registered one, False otherwise.
    """

    registered_df = pd.read_csv(registered, delimiter="\t")
    controlled_df = pd.read_csv(controlled, delimiter="\t")
    
    control_col = set(controlled_df.columns)
    registered_col = set(registered_df.columns)
    
    control_missing_col = control_col - registered_col
    registered_missing_col = registered_col - control_col
    
    if control_missing_col != set():
        print(f"The following columns are missing in registered acesss {control_missing_col} in {registered}")
        return
    
    if registered_missing_col != set():
        print(f"The following columns are missing in registered acesss {registered_missing_col} in {controlled}")
        return
    
    # reorder columns to match columns order, since we check that clumns match earlier, this shouldn't be an issue
    controlled_df = controlled_df[registered_df.columns]

    registered_df = registered_df.astype(str)
    controlled_df = controlled_df.astype(str)

    registered_counts = registered_df.value_counts()
    controlled_counts = controlled_df.value_counts()


    registered_aligned = registered_counts.reindex(controlled_counts.index, fill_value=0)

    is_subset = (controlled_counts <= registered_aligned).all()
      
    if not is_subset:
        filestem = Path(controlled).stem.replace(".tsv", "")
        offending_rows_values = controlled_counts[controlled_counts > registered_aligned].index

        controlled_offending_df = controlled_df[
            controlled_df.apply(lambda row: tuple(row) in offending_rows_values, axis=1)
        ]    
        controlled_offending_df.to_csv(f"error_{filestem}.tsv", sep="\t")

    return is_subset



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("registered_folder", type=str, help="Path to Dataframe containing the registered access phenotype")
    parser.add_argument("controlled_folder", type=str, help="Path to Dataframe containing the controlled phenotype file")
    
    
    args = parser.parse_args()
    
    all_controlled_tsv = [str(p) for p in Path(args.controlled_folder).rglob('*.tsv')]
    #print(all_controlled_tsv)
    for tsv_path in all_controlled_tsv:
        registered_tsv_path = tsv_path.replace(args.controlled_folder, args.registered_folder)
        try:
            is_same = (compare(registered_tsv_path, tsv_path))
            if not is_same:
                print(f"{tsv_path} does not match with {registered_tsv_path}")
        except (FileNotFoundError, pd.errors.ParserError) as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)