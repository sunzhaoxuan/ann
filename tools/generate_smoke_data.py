#!/usr/bin/env python3
"""Generate a deterministic, tiny fbin/ground-truth dataset for smoke tests."""

import argparse
import math
import random
import struct
from pathlib import Path


def normalized_vector(rng: random.Random, dimension: int) -> list[float]:
    values = [rng.uniform(-1.0, 1.0) for _ in range(dimension)]
    length = math.sqrt(sum(value * value for value in values)) or 1.0
    return [value / length for value in values]


def write_matrix(path: Path, rows: list[list[float]], value_format: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as output:
        output.write(struct.pack("<II", len(rows), len(rows[0])))
        for row in rows:
            output.write(struct.pack(f"<{len(row)}{value_format}", *row))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/smoke")
    parser.add_argument("--base-count", type=int, default=256)
    parser.add_argument("--query-count", type=int, default=32)
    parser.add_argument("--dimension", type=int, default=32)
    parser.add_argument("--ground-truth-width", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    if min(args.base_count, args.query_count, args.dimension, args.ground_truth_width) <= 0:
        raise SystemExit("all sizes must be positive")
    if args.ground_truth_width > args.base_count:
        raise SystemExit("ground-truth width cannot exceed base count")

    rng = random.Random(args.seed)
    base = [normalized_vector(rng, args.dimension) for _ in range(args.base_count)]
    queries = [normalized_vector(rng, args.dimension) for _ in range(args.query_count)]
    truth: list[list[int]] = []

    for query in queries:
        ranked = sorted(
            range(args.base_count),
            key=lambda index: sum(a * b for a, b in zip(base[index], query)),
            reverse=True,
        )
        truth.append(ranked[: args.ground_truth_width])

    output = Path(args.output)
    write_matrix(output / "DEEP100K.base.100k.fbin", base, "f")
    write_matrix(output / "DEEP100K.query.fbin", queries, "f")
    write_matrix(output / "DEEP100K.gt.query.100k.top100.bin", truth, "i")
    print(f"generated smoke dataset in {output.resolve()}")


if __name__ == "__main__":
    main()
