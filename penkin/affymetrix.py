import gzip
import os
import re
import struct
from collections import OrderedDict

import numpy as np


def _open_binary(path):
    return gzip.open(path, 'rb') if str(path).endswith('.gz') else open(path, 'rb')


def _open_text(path):
    return gzip.open(path, 'rt', errors='replace') if str(path).endswith('.gz') else open(path, 'rt', errors='replace')


def read_xda_cdf(path):
    """Read an Affymetrix XDA CDF and return PM coordinates by probeset."""
    with _open_binary(path) as fh:
        buf = fh.read()
    off = 0

    def take(fmt):
        nonlocal off
        n = struct.calcsize('<' + fmt)
        vals = struct.unpack_from('<' + fmt, buf, off)
        off += n
        return vals[0] if len(vals) == 1 else vals

    magic = take('i')
    version = take('i')
    if magic != 67 or version not in (1, 2):
        raise ValueError(f'Unsupported CDF: magic={magic}, version={version}')
    cols = take('H')
    rows = take('H')
    n_units = take('i')
    n_qc = take('i')
    ref_len = take('i')
    off += ref_len

    names = []
    for _ in range(n_units):
        raw = buf[off:off + 64]
        off += 64
        names.append(raw.split(b'\x00', 1)[0].decode('latin1'))

    off += 4 * n_qc
    unit_offsets = list(struct.unpack_from('<' + 'i' * n_units, buf, off))

    units = OrderedDict()
    for name, pos in zip(names, unit_offsets):
        p = pos
        unit_type = struct.unpack_from('<H', buf, p)[0]
        p += 2
        direction = struct.unpack_from('<B', buf, p)[0]
        p += 1
        n_atoms, n_blocks, n_cells, unit_number = struct.unpack_from('<iiii', buf, p)
        p += 16
        cells_per_atom = struct.unpack_from('<B', buf, p)[0]
        p += 1
        pm = []
        all_cells = []
        for _ in range(n_blocks):
            b_atoms, b_cells = struct.unpack_from('<ii', buf, p)
            p += 8
            b_cells_per_atom = struct.unpack_from('<B', buf, p)[0]
            p += 1
            b_direction = struct.unpack_from('<B', buf, p)[0]
            p += 1
            first_atom, unused = struct.unpack_from('<ii', buf, p)
            p += 8
            block_name = buf[p:p + 64].split(b'\x00', 1)[0].decode('latin1')
            p += 64
            if version >= 2:
                p += 4
            for _ in range(b_cells):
                atom = struct.unpack_from('<i', buf, p)[0]
                p += 4
                x, y = struct.unpack_from('<HH', buf, p)
                p += 4
                index = struct.unpack_from('<i', buf, p)[0]
                p += 4
                pbase = chr(buf[p]).lower()
                tbase = chr(buf[p + 1]).lower()
                p += 2
                if version >= 2:
                    probe_len, group = struct.unpack_from('<HH', buf, p)
                    p += 4
                cell = (atom, x, y, index, pbase, tbase)
                all_cells.append(cell)
                if pbase == tbase:
                    pm.append(cell)
        units[name] = {
            'unit_type': int(unit_type),
            'direction': int(direction),
            'unit_number': int(unit_number),
            'n_atoms': int(n_atoms),
            'n_cells': int(n_cells),
            'cells_per_atom': int(cells_per_atom),
            'pm': pm,
            'all': all_cells,
        }
    return {
        'version': version,
        'rows': rows,
        'cols': cols,
        'n_units': n_units,
        'n_qc': n_qc,
        'units': units,
    }


def read_cel_v3(path):
    """Read Affymetrix text CEL v3 intensities into a rows x cols matrix."""
    rows = cols = None
    arr = None
    in_intensity = False
    with _open_text(path) as fh:
        for raw in fh:
            line = raw.rstrip('\r\n')
            if line.startswith('Cols='):
                cols = int(line.split('=', 1)[1])
            elif line.startswith('Rows='):
                rows = int(line.split('=', 1)[1])
            elif line == '[INTENSITY]':
                if rows is None or cols is None:
                    raise ValueError('CEL header missing dimensions')
                arr = np.full((rows, cols), np.nan, dtype=np.float32)
                in_intensity = True
            elif in_intensity:
                parts = line.split()
                if len(parts) >= 5 and parts[0].isdigit() and parts[1].isdigit():
                    x, y = int(parts[0]), int(parts[1])
                    arr[y, x] = float(parts[2])
                elif line.startswith('[') and line != '[INTENSITY]':
                    in_intensity = False
    if arr is None:
        raise ValueError(f'No intensity section in {path}')
    return arr



def read_cel_v4(path):
    """Read Affymetrix binary/XDA CEL v4 intensities."""
    with _open_binary(path) as fh:
        buf = fh.read()
    off = 0
    def i32(signed=True):
        nonlocal off
        fmt = '<i' if signed else '<I'
        v = struct.unpack_from(fmt, buf, off)[0]; off += 4; return v
    def lp_string():
        nonlocal off
        n = i32()
        raw = buf[off:off+n]; off += n
        return raw.decode('ascii', 'ignore')
    magic = i32(); version = i32(); cols = i32(); rows = i32(); total = i32()
    if magic != 64 or version != 4:
        raise ValueError(f'Unsupported binary CEL: magic={magic}, version={version}')
    header = lp_string()
    algorithm = lp_string()
    parameters = lp_string()
    cellmargin = i32()
    noutliers = i32(signed=False)
    nmasked = i32(signed=False)
    nsubgrids = i32()
    need = total * 10
    if off + need > len(buf):
        raise ValueError(f'Truncated CEL v4 intensity block: need {need}, have {len(buf)-off}')
    rec = np.frombuffer(buf, dtype=np.dtype([('mean','<f4'),('sd','<f4'),('npix','<i2')]), count=total, offset=off)
    arr = rec['mean'].astype(np.float32, copy=True).reshape(rows, cols)
    return arr

def read_cel(path):
    with _open_binary(path) as fh:
        magic = fh.read(4)
    if magic == b'[CEL':
        return read_cel_v3(path)
    if len(magic) == 4 and struct.unpack('<i', magic)[0] == 64:
        return read_cel_v4(path)
    raise ValueError(f'Unrecognized CEL format: {path}')


def probeset_pm_index(cdf):
    xs, ys = [], []
    slices = OrderedDict()
    for name, unit in cdf['units'].items():
        idx = []
        for _, x, y, _, _, _ in unit['pm']:
            idx.append(len(xs))
            xs.append(x)
            ys.append(y)
        slices[name] = np.asarray(idx, dtype=np.int32)
    return np.asarray(xs, dtype=np.int32), np.asarray(ys, dtype=np.int32), slices


def load_pm_matrix(cel_paths, cdf_path):
    cdf = read_xda_cdf(cdf_path)
    xs, ys, slices = probeset_pm_index(cdf)
    mat = np.empty((len(xs), len(cel_paths)), dtype=np.float32)
    sample_names = []
    qc = []
    for j, path in enumerate(cel_paths):
        arr = read_cel(path)
        vals = arr[ys, xs]
        mat[:, j] = vals
        sample = os.path.basename(path).split('.')[0]
        sample_names.append(sample)
        finite = vals[np.isfinite(vals)]
        qc.append({
            'sample': sample,
            'n_pm': int(finite.size),
            'median_pm': float(np.median(finite)),
            'q25_pm': float(np.percentile(finite, 25)),
            'q75_pm': float(np.percentile(finite, 75)),
            'mean_pm': float(np.mean(finite)),
            'pct_pm_ge_20000': float(100 * np.mean(finite >= 20000)),
        })
    return sample_names, mat, slices, qc


def quantile_normalize_log2(pm_matrix):
    x = np.log2(np.maximum(np.asarray(pm_matrix, dtype=np.float64), 1.0))
    order = np.argsort(x, axis=0)
    sorted_x = np.take_along_axis(x, order, axis=0)
    target = np.nanmean(sorted_x, axis=1)
    out = np.empty_like(x)
    for j in range(x.shape[1]):
        out[order[:, j], j] = target
    return out


def summarize_probesets(qn_log2, slices, method='mean'):
    names = []
    rows = []
    for name, idx in slices.items():
        if idx.size == 0:
            continue
        block = qn_log2[idx, :]
        if method == 'median':
            s = np.nanmedian(block, axis=0)
        else:
            s = np.nanmean(block, axis=0)
        names.append(name)
        rows.append(s)
    return names, np.asarray(rows, dtype=np.float64)


def parse_series_matrix(path):
    sample_ids = []
    sample_titles = []
    rows = []
    ids = []
    in_table = False
    with _open_text(path) as fh:
        for raw in fh:
            line = raw.rstrip('\r\n')
            if line.startswith('!Sample_geo_accession'):
                sample_ids = [x.strip('"') for x in line.split('\t')[1:]]
            elif line.startswith('!Sample_title'):
                sample_titles = [x.strip('"') for x in line.split('\t')[1:]]
            elif line == '!series_matrix_table_begin':
                in_table = True
            elif line == '!series_matrix_table_end':
                in_table = False
            elif in_table:
                parts = line.split('\t')
                if parts[0].strip('"') == 'ID_REF':
                    continue
                ids.append(parts[0].strip('"'))
                rows.append([float(x.strip('"')) if x.strip('"') not in ('', 'null', 'NA') else np.nan for x in parts[1:]])
    return sample_ids, sample_titles, ids, np.asarray(rows, dtype=float)


def infer_geo_group(title):
    t = title.lower()
    if 'ds17690' in t:
        strain = 'DS17690'
    elif 'wisconsin' in t:
        strain = 'Wisconsin54-1255'
    elif 'ds50661' in t or 'cluster free' in t or 'cluster-free' in t:
        strain = 'DS50661'
    else:
        strain = 'other'
    paa = '+PAA' if ('presence of paa' in t or 'with paa' in t or ' +paa' in t) else '-PAA'
    return strain, paa
