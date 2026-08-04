"""Scoring-matrix and affine-gap alignment utilities from bx-python."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from bx._lib import addr, lib


class ScoringScheme:
    def __init__(self, gap_open, gap_extend, default=-100, alphabet1="ACGT", alphabet2=None,
                 gap1="-", gap2=None, text1_range=128, text2_range=None, typecode=np.int32):
        if text2_range is None:
            text2_range = text1_range
        if alphabet2 is None:
            alphabet2 = alphabet1
        if gap2 is None:
            gap2 = gap1
        if isinstance(alphabet1, str):
            alphabet1 = list(alphabet1)
        if isinstance(alphabet2, str):
            alphabet2 = list(alphabet2)
        self.table = np.ones((text1_range, text2_range), dtype=typecode) * default
        self.gap_open = gap_open
        self.gap_extend = gap_extend
        self.gap1, self.gap2 = gap1, gap2
        self.alphabet1, self.alphabet2 = alphabet1, alphabet2

    def _set_score(self, a_b_pair, val):
        self.table[a_b_pair] = val

    def _get_score(self, a_b_pair):
        return self.table[a_b_pair]

    def set_score(self, a, b, val, foldcase1=False, foldcase2=False):
        self._set_score((a, b), val)
        if foldcase1:
            ch = chr(a)
            if ch.isupper():
                aa = ord(ch.lower())
            elif ch.islower():
                aa = ord(ch.upper())
            else:
                foldcase1 = False
        if foldcase2:
            ch = chr(b)
            if ch.isupper():
                bb = ord(ch.lower())
            elif ch.islower():
                bb = ord(ch.upper())
            else:
                foldcase2 = False
        if foldcase1 and foldcase2:
            self._set_score((aa, b), val)
            self._set_score((a, bb), val)
            self._set_score((aa, bb), val)
        elif foldcase1:
            self._set_score((aa, b), val)
        elif foldcase2:
            self._set_score((a, bb), val)

    def score_alignment(self, a):
        return score_alignment(self, a)

    def score_texts(self, text1, text2):
        return score_texts(self, text1, text2)

    def __str__(self):
        is_dna1 = "".join(self.alphabet1) == "ACGT"
        is_dna2 = "".join(self.alphabet2) == "ACGT"
        label_rows = not (is_dna1 and is_dna2)
        width = 3
        for a in self.alphabet1:
            for b in self.alphabet2:
                score = self._get_score((ord(a), ord(b)))
                text = f"{score:8.6f}" if isinstance(score, float) else f"{score}"
                width = max(width, len(text) + 1)
        lines, line = [], []
        if label_rows:
            line.append(" " if is_dna1 else "  ")
        for b in self.alphabet2:
            line.append(f"{(b if is_dna2 else f'{ord(b):02X}'):>{width}}")
        lines.append("".join(line) + "\n")
        for a in self.alphabet1:
            line = []
            if label_rows:
                line.append(a if is_dna1 else f"{ord(a):02X}")
            for b in self.alphabet2:
                score = self._get_score((ord(a), ord(b)))
                text = f"{score:8.6f}" if isinstance(score, float) else f"{score}"
                line.append(f"{text:>{width}}")
            lines.append("".join(line) + "\n")
        return "".join(lines)


def _text_bytes(text):
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    try:
        return np.frombuffer(text.encode("latin-1"), dtype=np.uint8)
    except UnicodeEncodeError:
        # The C ABI deliberately accepts bytes only.  Keep Unicode inputs on
        # the Python path instead of truncating code points or indexing native
        # memory outside the scoring table.
        return None


def _native_table(scoring_scheme):
    table = np.asarray(scoring_scheme.table)
    if table.ndim != 2 or table.dtype.kind not in "biuf":
        return None
    if table.dtype.kind in "iu" and np.any(np.abs(table.astype(object)) > 2**53):
        return None
    native = np.ascontiguousarray(table, dtype=np.float64)
    # Do not silently turn large integers or extended-precision values into a
    # different scoring matrix at the ABI boundary.
    if not np.array_equal(table, native):
        return None
    return native


def _score_texts_python(scoring_scheme, text1, text2):
    total = 0
    last_gap_a = last_gap_b = False
    for ca, cb in zip(text1, text2, strict=True):
        if ca == scoring_scheme.gap1 and cb == scoring_scheme.gap2:
            continue
        if ca == scoring_scheme.gap1:
            total -= scoring_scheme.gap_extend
            if not last_gap_a:
                total -= scoring_scheme.gap_open
                last_gap_a, last_gap_b = True, False
        elif cb == scoring_scheme.gap2:
            total -= scoring_scheme.gap_extend
            if not last_gap_b:
                total -= scoring_scheme.gap_open
                last_gap_a, last_gap_b = False, True
        else:
            total += scoring_scheme.table[ord(ca), ord(cb)]
            last_gap_a = last_gap_b = False
    return total


def _can_use_native(encoded1, encoded2, table):
    if encoded1 is None or encoded2 is None or table is None:
        return False
    # Native code performs unchecked table indexing, so prove every byte is
    # representable before passing its address across the FFI boundary.
    return (not len(encoded1) or encoded1.max() < table.shape[0]) and (
        not len(encoded2) or encoded2.max() < table.shape[1]
    )


def score_texts(scoring_scheme, text1, text2):
    if len(text1) != len(text2):
        raise IndexError("string index out of range")
    a, b = _text_bytes(text1), _text_bytes(text2)
    table = _native_table(scoring_scheme)
    if not _can_use_native(a, b, table):
        return _score_texts_python(scoring_scheme, text1, text2)
    return lib().mbx_score_texts(addr(a), addr(b), len(a), addr(table), table.shape[1],
                                 ord(scoring_scheme.gap1), ord(scoring_scheme.gap2),
                                 float(scoring_scheme.gap_open), float(scoring_scheme.gap_extend))


def score_alignment(scoring_scheme, a):
    total = 0.0
    for i in range(len(a.components)):
        for j in range(i + 1, len(a.components)):
            total += score_texts(scoring_scheme, a.components[i].text, a.components[j].text)
    return total


def accumulate_scores(scoring_scheme, text1, text2, skip_ref_gaps=False):
    if len(text1) != len(text2):
        raise IndexError("string index out of range")
    a, b = _text_bytes(text1), _text_bytes(text2)
    table = _native_table(scoring_scheme)
    if not _can_use_native(a, b, table):
        scores, total = [], 0
        last_gap_a = last_gap_b = False
        for ca, cb in zip(text1, text2, strict=True):
            if ca == scoring_scheme.gap1 and cb == scoring_scheme.gap2:
                pass
            elif ca == scoring_scheme.gap1:
                total -= scoring_scheme.gap_extend
                if not last_gap_a:
                    total -= scoring_scheme.gap_open
                    last_gap_a, last_gap_b = True, False
            elif cb == scoring_scheme.gap2:
                total -= scoring_scheme.gap_extend
                if not last_gap_b:
                    total -= scoring_scheme.gap_open
                    last_gap_a, last_gap_b = False, True
            else:
                total += scoring_scheme.table[ord(ca), ord(cb)]
                last_gap_a = last_gap_b = False
            if not skip_ref_gaps or ca != scoring_scheme.gap1:
                scores.append(total)
        return np.asarray(scores, dtype=float)
    result = np.empty(len(text1) - (text1.count(scoring_scheme.gap1) if skip_ref_gaps else 0), dtype=float)
    used = lib().mbx_accumulate_scores(addr(a), addr(b), len(a), addr(table), table.shape[1],
                                       ord(scoring_scheme.gap1), ord(scoring_scheme.gap2),
                                       float(scoring_scheme.gap_open), float(scoring_scheme.gap_extend),
                                       int(skip_ref_gaps), addr(result))
    return result[:used]


def read_scoring_scheme(f, gap_open, gap_extend, gap1="-", gap2=None, **kwargs):
    close_it = False
    if isinstance(f, (str, Path)):
        f, close_it = open(f), True
    try:
        return build_scoring_scheme("".join(f), gap_open, gap_extend, gap1=gap1, gap2=gap2, **kwargs)
    finally:
        if close_it:
            f.close()


def int_or_float(s):
    try:
        return int(s)
    except ValueError:
        return float(s)


def sym_to_char(sym):
    if len(sym) == 1:
        return sym
    if len(sym) != 2:
        raise ValueError
    return chr(int(sym, base=16))


def build_scoring_scheme(s, gap_open, gap_extend, gap1="-", gap2=None, **kwargs):
    bad_matrix = "invalid scoring matrix"
    lines = s.rstrip("\n").split("\n")
    symbols2 = lines.pop(0).split()
    rows, symbols1 = [], None
    rows_have_syms, blastz = False, True
    for line in lines:
        row = line.split()
        if len(row) == len(symbols2):
            if symbols1 is None:
                if len(lines) != len(symbols2):
                    raise Exception(bad_matrix)
                symbols1 = symbols2
            elif rows_have_syms:
                raise Exception(bad_matrix)
        elif len(row) == len(symbols2) + 1:
            if symbols1 is None:
                symbols1, rows_have_syms, blastz = [], True, False
            elif not rows_have_syms:
                raise Exception(bad_matrix)
            symbols1.append(row.pop(0))
        else:
            raise Exception(bad_matrix)
        rows.append(row)
    try:
        alphabet1 = [sym_to_char(sym) for sym in symbols1]
        alphabet2 = [sym_to_char(sym) for sym in symbols2]
    except ValueError:
        raise Exception(bad_matrix) from None
    if alphabet1 != symbols1 or alphabet2 != symbols2:
        blastz = False
    if blastz:
        alphabet1, alphabet2 = [x.upper() for x in alphabet1], [x.upper() for x in alphabet2]
    foldcase1 = blastz or "".join(alphabet1) == "ACGT"
    foldcase2 = blastz or "".join(alphabet2) == "ACGT"
    range1 = 256 if ord(max(alphabet1)) >= 128 else 128
    range2 = 256 if ord(max(alphabet2)) >= 128 else 128
    typecode = np.float32 if any(isinstance(int_or_float(x), float) for row in rows for x in row) or isinstance(gap_open, float) or isinstance(gap_extend, float) else np.int32
    scheme = ScoringScheme(gap_open, gap_extend, alphabet1=alphabet1, alphabet2=alphabet2,
                           gap1=gap1, gap2=gap2, text1_range=range1, text2_range=range2,
                           typecode=typecode, **kwargs)
    for i, row in enumerate(rows):
        for j, score in enumerate(map(int_or_float, row)):
            scheme.set_score(ord(alphabet1[i]), ord(alphabet2[j]), score)
            if foldcase1 and foldcase2:
                scheme.set_score(ord(alphabet1[i].lower()), ord(alphabet2[j].upper()), score)
                scheme.set_score(ord(alphabet1[i].upper()), ord(alphabet2[j].lower()), score)
                scheme.set_score(ord(alphabet1[i].lower()), ord(alphabet2[j].lower()), score)
            elif foldcase1:
                scheme.set_score(ord(alphabet1[i].lower()), ord(alphabet2[j]), score)
            elif foldcase2:
                scheme.set_score(ord(alphabet1[i]), ord(alphabet2[j].lower()), score)
    return scheme


hox70 = build_scoring_scheme("""  A    C    G    T
                                  91 -114  -31 -123
                                -114  100 -125  -31
                                 -31 -125  100 -114
                                -123  -31 -114   91 """, 400, 30)
