"""
Parse CASIA v2.0 source identifiers and build connected-component groups.

This script implements the source-aware grouping protocol described in the paper:
1. Parse all recoverable source IDs from CASIA filenames
2. Build a source graph (nodes=IDs, edges=co-occurrence in same filename)
3. Compute connected components
4. Output group assignments for GroupKFold splitting

Usage:
    python src/parse_casia_sources.py --casia-dir /path/to/CASIA2 --output splits/casia_source_groups.csv
"""

import argparse
import os
import re
from pathlib import Path

import pandas as pd


def parse_source_ids(filename):
    """Parse all recoverable CASIA-style source identifiers from a filename.
    
    CASIA v2.0 naming conventions:
      Authentic: Au_<category>_<index>.<ext>
        e.g. Au_ani_00018.jpg -> ['ani00018']
      
      Tampered:  Tp_<type>_<method>_<size>_<postproc>_<src1>_<src2>_<serial>.<ext>
        e.g. Tp_D_CND_M_N_ani00018_sec00096_00138.tif -> ['ani00018', 'sec00096']
    
    Source ID pattern: 3 lowercase letters + 5 digits (e.g., ani00018, sec00096)
    """
    stem = Path(filename).stem
    
    if stem.startswith("Au_"):
        parts = stem.split("_")
        if len(parts) >= 3:
            return [parts[1] + parts[2]]
        return [stem]
    
    elif stem.startswith("Tp_"):
        ids = re.findall(r'([a-z]{3}\d{5})', stem)
        if ids:
            return list(dict.fromkeys(ids))  # deduplicate preserving order
        return [stem]
    
    return [stem]


def build_connected_components(filenames, labels):
    """Build connected components from co-occurring source IDs.
    
    Two source IDs are connected if they appear in the same tampered filename.
    All images sharing any connected source ID are placed in the same group,
    preventing source-identity leakage across GroupKFold partitions.
    
    Returns:
        file_ids: dict mapping filename -> list of parsed source IDs
        file_components: dict mapping filename -> component label string
    """
    # Parse source IDs for each file
    file_ids = {}
    all_ids = set()
    for fn in filenames:
        ids = parse_source_ids(fn)
        file_ids[fn] = ids
        all_ids.update(ids)
    
    # Union-Find with path compression
    parent = {sid: sid for sid in all_ids}
    
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    
    # Connect IDs co-occurring in the same filename
    for fn, ids in file_ids.items():
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                union(ids[i], ids[j])
    
    # Assign component labels
    comp_counter = {}
    next_comp = 0
    file_components = {}
    
    for fn, ids in file_ids.items():
        root = find(ids[0])
        if root not in comp_counter:
            comp_counter[root] = next_comp
            next_comp += 1
        file_components[fn] = f"component_{comp_counter[root]:04d}"
    
    return file_ids, file_components


def main():
    parser = argparse.ArgumentParser(description="Parse CASIA v2.0 source groups")
    parser.add_argument("--casia-dir", required=True, help="Path to CASIA2/ folder containing Au/ and Tp/")
    parser.add_argument("--output", default="splits/casia_source_groups.csv", help="Output CSV path")
    args = parser.parse_args()

    casia_dir = Path(args.casia_dir)
    au_dir = casia_dir / "Au"
    tp_dir = casia_dir / "Tp"

    # List files
    valid_ext = {'.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    au_files = sorted([f for f in os.listdir(au_dir)
                       if Path(f).suffix.lower() in valid_ext and not f.startswith('.')])
    tp_files = sorted([f for f in os.listdir(tp_dir)
                       if Path(f).suffix.lower() in valid_ext
                       and not f.startswith('.') and f != 'Thumbs.db'])

    all_files = au_files + tp_files
    labels = ['authentic'] * len(au_files) + ['tampered'] * len(tp_files)
    
    print(f"Authentic: {len(au_files)}, Tampered: {len(tp_files)}, Total: {len(all_files)}")

    # Build groups
    file_ids, file_components = build_connected_components(all_files, labels)
    n_components = len(set(file_components.values()))
    print(f"Connected components (source groups): {n_components}")

    # Write output
    rows = []
    for fn, label in zip(all_files, labels):
        ext = Path(fn).suffix.lower().lstrip('.')
        rows.append({
            'filename': fn,
            'label': label,
            'extension': ext,
            'parsed_source_ids': ';'.join(file_ids[fn]),
            'component_group_id': file_components[fn],
            'is_jpeg': ext in ('jpg', 'jpeg'),
        })
    
    df = pd.DataFrame(rows)
    os.makedirs(Path(args.output).parent, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Written: {args.output} ({len(df)} rows)")
    
    # Summary
    jpeg_count = df['is_jpeg'].sum()
    print(f"\nJPEG-only subset: {jpeg_count} images")
    print(f"JPEG authentic: {((df['label']=='authentic') & (df['is_jpeg'])).sum()}")
    print(f"JPEG tampered: {((df['label']=='tampered') & (df['is_jpeg'])).sum()}")


if __name__ == "__main__":
    main()
