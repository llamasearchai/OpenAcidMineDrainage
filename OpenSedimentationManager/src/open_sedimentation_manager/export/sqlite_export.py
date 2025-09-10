from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Dict, Iterable, Tuple


def _infer_columns(sample_props: Dict) -> Dict[str, str]:
    schema: Dict[str, str] = {}
    for k, v in sample_props.items():
        if isinstance(v, bool):
            schema[k] = "INTEGER"
        elif isinstance(v, int):
            schema[k] = "INTEGER"
        elif isinstance(v, float):
            schema[k] = "REAL"
        else:
            schema[k] = "TEXT"
    return schema


def export_geojson_to_sqlite(source_path: str, sqlite_path: str, table_name: str = "work_orders") -> int:
    """Export a GeoJSON FeatureCollection to a SQLite database.

    - Flattens feature properties into columns.
    - Adds a 'geometry' TEXT column storing the GeoJSON geometry.
    - Returns the number of rows inserted.
    """
    data = json.loads(Path(source_path).read_text())
    if not isinstance(data, dict) or data.get("type") != "FeatureCollection":
        raise ValueError("Input must be a GeoJSON FeatureCollection.")

    features = data.get("features", [])
    if not features:
        # Create empty table if no features
        with sqlite3.connect(sqlite_path) as conn:
            conn.execute(f"CREATE TABLE IF NOT EXISTS {table_name} (geometry TEXT)")
        return 0

    # Infer schema from first feature's properties
    first_props = features[0].get("properties", {})
    schema = _infer_columns(first_props)

    columns_sql = ", ".join([f"{k} {t}" for k, t in schema.items()])
    ddl = f"CREATE TABLE IF NOT EXISTS {table_name} ({columns_sql}, geometry TEXT)"

    with sqlite3.connect(sqlite_path) as conn:
        conn.execute(ddl)
        # Prepare insert
        cols = list(schema.keys())
        placeholders = ",".join(["?"] * (len(cols) + 1))
        insert_sql = f"INSERT INTO {table_name} ({', '.join(cols)}, geometry) VALUES ({placeholders})"
        rows = []
        for feat in features:
            props = feat.get("properties", {})
            geom = json.dumps(feat.get("geometry"))
            row = [props.get(c) for c in cols] + [geom]
            rows.append(row)
        conn.executemany(insert_sql, rows)
        return len(rows)

