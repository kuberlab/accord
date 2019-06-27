# -*- coding: utf-8 -*-

from __future__ import division

import logging

import numpy as np

from utils.image import (adaptive_threshold, find_lines,
                         find_contours, find_joints)

logger = logging.getLogger('lattice')


def merge_close_lines(ar, line_tol=2):
    ret = []
    for a in ar:
        if not ret:
            ret.append(a)
        else:
            temp = ret[-1]
            if np.isclose(temp, a, atol=line_tol):
                temp = (temp + a) / 2.0
                ret[-1] = temp
            else:
                ret.append(a)
    return ret


def generate_table_bbox(img):
    image, threshold = adaptive_threshold(img)
    vertical_mask, vertical_segments = find_lines(
        threshold, direction='vertical',line_scale=30)
    horizontal_mask, horizontal_segments = find_lines(
        threshold, direction='horizontal',line_scale=50)

    contours = find_contours(vertical_mask, horizontal_mask)
    table_bbox = find_joints(contours, vertical_mask, horizontal_mask)
    return table_bbox, (vertical_segments, horizontal_segments)


def generate_cells(table_bbox, tk):
    cols, rows = zip(*table_bbox[tk])
    cols, rows = list(cols), list(rows)
    cols.extend([tk[0], tk[2]])
    rows.extend([tk[1], tk[3]])
    cols = merge_close_lines(sorted(cols), line_tol=2)
    rows = merge_close_lines(sorted(rows, reverse=True), line_tol=4)
    cols = [(cols[i], cols[i + 1])
            for i in range(0, len(cols) - 1)]
    rows = [(rows[i], rows[i + 1])
            for i in range(0, len(rows) - 1)]

    cells = [[(c[0], r[1], c[1], r[0]) for c in cols if (c[1] - c[0]) > 10 and (r[0] - r[1]) > 10] for r in rows]
    cells.reverse()
    cells = list(filter(lambda c: len(c) > 0, cells))
    return cells


def extract_tables(img):
    table_bbox, segments = generate_table_bbox(img)
    _tables = []
    for tk in sorted(table_bbox.keys(), key=lambda x: x[1], reverse=True):
        cells = generate_cells(table_bbox, tk)
        if len(cells) > 0:
            _tables.append(cells)
    _tables.reverse()
    return _tables, segments
