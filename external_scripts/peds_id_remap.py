import hmac
import hashlib
import secrets
import argparse
import json
import pandas as pd
from pathlib import Path

SEED_SIZE = 32
ID_LENGTH = 6


def generate_or_load_seed():
    """Generate a cryptographic seed"""
    seed = secrets.token_bytes(SEED_SIZE)
    return seed


def generate_pseudonym(id_str, secret_key, length, salt=""):
    """Generates a fixed-length pseudonym."""
    message = id_str + salt
    digest = hmac.new(secret_key, message.encode(), hashlib.sha256).hexdigest()
    number = int(digest[:12], 16) % 10**length
    return str(number).zfill(length)


def load_lookup_table(lookup_path):
    lookup_path = Path(lookup_path)

    if not lookup_path.exists():
        raise FileNotFoundError(f"Lookup table not found at: {lookup_path}")

    with open(lookup_path, "r") as f:
        lookup_table = json.load(f)

    remapped_ids = set(lookup_table.values())

    return lookup_table, remapped_ids


def build_lookup_table(original_ids, secret_key, load_lookup=None):
    """
    Generates the look up table using the secret and original ids.
    """
    if load_lookup is None:
        lookup_table = {}
        used_values = set()
    else:
        lookup_table, used_values = load_lookup_table(load_lookup)
    for oid in original_ids:
        if oid in lookup_table:
            continue
        salt_counter = 0
        while True:
            salt = str(salt_counter) if salt_counter > 0 else ""
            pseudo_id = generate_pseudonym(oid, secret_key, ID_LENGTH, salt)
            if pseudo_id not in used_values:
                lookup_table[oid] = pseudo_id
                used_values.add(pseudo_id)
                break
            salt_counter += 1  # for collisions

    return lookup_table


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", help="Path to TSV file with ids")
    parser.add_argument("output_path", help="Directory for remap file and salt")
    parser.add_argument("--load_lookup", type=str, help="Path to a look up table")

    args = parser.parse_args()
    df = pd.read_csv(args.input_path, sep='\t')
    id_column = ""
    if "record_id" in df.columns:
        id_column = "record_id"
    elif "participant_id" in df.columns:
        id_column = "participant_id"
    else:
        raise ValueError(
            "No valid ID column found (expected 'record_id' or 'participant_id')"
        )

    id_list = list(set(df[id_column].tolist()))

    secret_key = generate_or_load_seed()
    lookup = build_lookup_table(id_list, secret_key, args.load_lookup)

    output_dir = Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "id_lookup_table.json"
    with open(output_file, "w") as f:
        json.dump(lookup, f, indent=2)

    print(f"Lookup table saved to '{output_file}'")
