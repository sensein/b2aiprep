import pandas as pd
import argparse

def compare(restricted, controlled):
    restricted_df = pd.read_csv(restricted, delimiter="\t")
    controlled_df = pd.read_csv(controlled, delimiter="\t")
    
    restricted_counts = restricted_df.value_counts()
    controlled_counts = controlled_df.value_counts()

    restricted_aligned = restricted_counts.reindex(controlled_counts.index, fill_value=0)

    is_subset = (controlled_counts <= restricted_aligned).all()
    return is_subset
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("df1", type=str, help="Path to Dataframe containing the rescrticted access phenotype")
    parser.add_argument("df2", type=str, help="Path to Dataframe containing the controlled phenotype file")
    
    args = parser.parse_args()
    print(compare(args.df1, args.df2))