import argparse
from config import seed_everything


def test_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dir", type=str, help="Directory to test file, i.e., /data/test.parquet.pq")
    args = parser.parse_args()
    
    seed_everything()
    
    return args