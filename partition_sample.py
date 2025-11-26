import csv
import os
import random
import sys
from typing import List, Tuple

class PartitionGenerator:
    def __init__(self, partition_count: int, user_id_count: int, seed: int | None = None):
        self.partition_count = partition_count
        self.user_id_count = user_id_count
        self.seed = seed
        self.input_csv = "./data/orders.csv"
        self.output_csv = "./user_partitions.csv"

    def _read_user_ids(self) -> List[str]:
        if not os.path.exists(self.input_csv):
            sys.exit(f"ERROR: Missing input CSV at {self.input_csv}")

        user_ids: List[str] = []
        with open(self.input_csv, newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            if r.fieldnames is None or "user_id" not in r.fieldnames:
                sys.exit("ERROR: CSV must have a header with a 'user_id' column.")
            for row in r:
                uid = row.get("user_id")
                if uid:
                    user_ids.append(str(uid))
        return user_ids

    def _validate(self, total_unique: int):
        if self.partition_count <= 0 or self.user_id_count <= 0:
            sys.exit("ERROR: partition_count and user_id_count must be positive integers.")
        if self.user_id_count > total_unique:
            sys.exit(
                f"ERROR: user_id_count ({self.user_id_count}) exceeds available unique users ({total_unique})."
            )

    def run(self) -> Tuple[int, int, str]:
        if self.seed is not None:
            random.seed(self.seed)

        all_user_ids = self._read_user_ids()
        total_orders = len(all_user_ids)
        unique_ids = sorted(set(all_user_ids))
        total_unique = len(unique_ids)

        print(f"Total orders: {total_orders}")
        print(f"Total unique user IDs: {total_unique}")

        self._validate(total_unique)

        rows: List[Tuple[str, int]] = []
        for p in range(1, self.partition_count + 1):
            sampled = random.sample(unique_ids, self.user_id_count)
            rows.extend((uid, p) for uid in sampled)
        with open(self.output_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["user_id", "partition"])
            w.writerows(rows)

        print(
            f"Wrote {len(rows)} rows to {self.output_csv} "
            f"({self.partition_count} partitions × {self.user_id_count} users)."
        )
        return total_orders, total_unique, self.output_csv

if __name__ == "__main__":
    partition_count = 1
    user_id_count = 100
    seed = 555
    pg = PartitionGenerator(partition_count, user_id_count, seed)
    pg.run()
