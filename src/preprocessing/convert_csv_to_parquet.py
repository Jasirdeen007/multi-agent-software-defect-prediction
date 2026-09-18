"""Convert the processed canonical CSV dataset to Parquet.

This script performs a format conversion only. It does not filter rows, change
labels, or modify any raw/source data.
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_INPUT = Path("data/processed/bughubs_canonical.csv")
DEFAULT_OUTPUT = Path("data/processed/bughubs_canonical.parquet")
DEFAULT_CHUNK_SIZE = 100_000
CANONICAL_COLUMNS = [
    "issue_id",
    "source",
    "project",
    "title",
    "description",
    "component",
    "severity",
    "priority",
    "status",
    "resolution",
    "original_label",
    "label_source",
    "created_at",
    "updated_at",
    "model_text",
]
RECORD_START_RE = re.compile(r"^\d+,(bugzilla|github|jira),", re.IGNORECASE)
LABEL_SUFFIX_RE = re.compile(
    r",(bug|non-bug),BugHub source view,([^,]+),([^,]+),(.*)\Z",
    re.DOTALL,
)


def is_record_start(line: str) -> bool:
    return bool(RECORD_START_RE.match(line))


def parse_csv_line(text: str) -> list[str]:
    return next(csv.reader([text]))


def parse_single_field(text: str) -> str:
    try:
        fields = parse_csv_line(text)
    except csv.Error:
        fields = []

    if len(fields) == 1:
        return fields[0]
    return text.strip().removeprefix('"').removesuffix('"')


def parse_title_description(text: str) -> tuple[str, str]:
    try:
        fields = parse_csv_line(text)
    except csv.Error:
        fields = []

    if len(fields) == 2:
        return fields[0], fields[1]

    title, separator, description = text.partition(",")
    if not separator:
        return parse_single_field(text), ""
    return parse_single_field(title), parse_single_field(description)


def parse_record(record: str) -> list[str]:
    try:
        fields = parse_csv_line(record)
    except csv.Error:
        fields = []

    if len(fields) == len(CANONICAL_COLUMNS):
        return fields

    issue_id, source, project, rest = record.split(",", 3)
    suffix_match = LABEL_SUFFIX_RE.search(rest)
    if suffix_match is None:
        raise ValueError(f"Could not parse canonical record starting with {issue_id!r}")

    original_label, created_at, updated_at, model_text = suffix_match.groups()
    label_source = "BugHub source view"
    prefix = rest[: suffix_match.start()]

    try:
        prefix_fields = parse_csv_line(prefix)
    except csv.Error:
        prefix_fields = []

    if len(prefix_fields) == 7:
        title, description, component, severity, priority, status, resolution = prefix_fields
    else:
        title_description, component, severity, priority, status, resolution = prefix.rsplit(",", 5)
        title, description = parse_title_description(title_description)

    return [
        issue_id,
        source,
        project,
        title,
        description,
        component,
        severity,
        priority,
        status,
        resolution,
        original_label,
        label_source,
        created_at,
        updated_at,
        parse_single_field(model_text),
    ]


def iter_records(input_path: Path):
    pending: list[str] = []

    with input_path.open("r", encoding="utf-8", newline="") as handle:
        for line in handle:
            if is_record_start(line) and pending:
                yield "".join(pending).rstrip("\r\n")
                pending = [line]
            else:
                pending.append(line)

    if pending:
        yield "".join(pending).rstrip("\r\n")


def convert_csv_to_parquet(
    input_path: Path,
    output_path: Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    compression: str = "snappy",
) -> int:
    """Stream a CSV file into a Parquet file and return the converted row count."""
    if not input_path.exists():
        raise FileNotFoundError(f"CSV file not found: {input_path}")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input CSV path and output Parquet path must be different.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows_written = 0
    writer: pq.ParquetWriter | None = None

    try:
        batch: list[list[str]] = []

        for record in iter_records(input_path):
            batch.append(parse_record(record))

            if len(batch) >= chunk_size:
                table = pa.Table.from_pylist(
                    [dict(zip(CANONICAL_COLUMNS, row)) for row in batch]
                )
                if writer is None:
                    writer = pq.ParquetWriter(
                        output_path,
                        table.schema,
                        compression=compression,
                    )
                writer.write_table(table)
                rows_written += table.num_rows
                batch = []

        if batch:
            table = pa.Table.from_pylist(
                [dict(zip(CANONICAL_COLUMNS, row)) for row in batch]
            )
            if writer is None:
                writer = pq.ParquetWriter(
                    output_path,
                    table.schema,
                    compression=compression,
                )
            writer.write_table(table)
            rows_written += table.num_rows
    finally:
        if writer is not None:
            writer.close()

    return rows_written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert data/processed canonical CSV data to Parquet."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input CSV path. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output Parquet path. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Rows to process per chunk. Default: {DEFAULT_CHUNK_SIZE}",
    )
    parser.add_argument(
        "--compression",
        default="snappy",
        choices=("none", "snappy", "gzip", "brotli", "zstd", "lz4"),
        help="Parquet compression codec. Default: snappy",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    compression = None if args.compression == "none" else args.compression

    rows_written = convert_csv_to_parquet(
        input_path=args.input,
        output_path=args.output,
        chunk_size=args.chunk_size,
        compression=compression,
    )

    print(f"Converted {rows_written:,} rows")
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
