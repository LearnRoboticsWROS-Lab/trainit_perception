#!/usr/bin/env python3
"""Score a candidate TERM against EXTERNAL labels. A term without this is a number, not a
measurement (ADR-0017 Decision 6).

    python3 scripts/score_term.py --term visible_fraction --labels labels.jsonl \
            --expected-radius-m 0.02095
    python3 scripts/score_term.py --term box_overlap --labels labels.jsonl
    python3 scripts/score_term.py --term edge_on --labels labels.jsonl --tool-max-tilt-deg 45
    python3 scripts/score_term.py --demo                      # self-test of this harness

WHY THIS SCRIPT EXISTS, and it is not a style preference. The first version of `visible_fraction`
divided by cos(tilt) -- the textbook way to separate occlusion from foreshortening, and correct in
a textbook. In a bin it destroyed the number: the fit's tilt runs to a median of 33 degrees, a p90
of 63, and cos(63 deg) is 0,45, so the term saturated at 1,00 on two thirds of detections
including objects the truth said were 60 % covered. The pick run that ranked by it was ranking by
a constant. **No unit test could see it**, because every unit test fed the function a tilt it had
computed itself. It took 224 detections with labels from OUTSIDE the code to find, and the number
that came back was r = +0,94 with the correction removed.

So a framework candidate term is DEFINED to include a scoring script against external labels, and
the digital twin is what supplies them: a simulator knows each object's true pose, its true face
radius and exactly how many pixels of it the renderer labelled.

THE LABEL FILE is JSON Lines, one row per detection ALREADY MATCHED to its label -- matching is
the twin's job, because only the twin knows which object a detection belongs to. Produce it with
whatever runs your detector (an Isaac script, a dataset walker, a rosbag reader); this script
never imports a model, a renderer or ROS.

    visible_fraction, per row:
        frame             str    any id; rows of one frame share it
        mask_area_px      float  pixels the DETECTOR's mask covered
        depth_m           float  the depth the face was found at (the fit's, or the detection's)
        fx, fy            float  the intrinsics those pixels were measured with
        truth_area_px     float  pixels of that face the LABELLER marked visible
        truth_radius_px   float  the face's full radius in pixels, from the labeller
        tilt_deg          float  optional, reported only -- the term must not use it

    box_overlap, per row:
        frame                   str
        box_px                  [u0, v0, u1, v1] the DETECTOR's box
        truth_visible_fraction  float  how much of that object the labeller says is visible

    edge_on, per row:
        frame             str
        tilt_deg          float  the tilt the FIT reported -- what the term reads
        truth_tilt_deg    float  the angle between the object's TRUE grasp-face normal and the
                                 camera axis, from the labeller
        truth_visible_fraction  float  optional; used only to check the two terms are not
                                 redundant, which is the difference between a TERM and the
                                 CORRECTION this one replaced (ADR-0018 Decision 5)

Exit code 0 when every check passed, 1 when any failed or the sample is too small to say.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trainit_perception.ranking import (box_overlap, edge_on, make_face,  # noqa: E402
                                        visible_fraction)

MIN_ROWS = 50               # under this a correlation is an anecdote


def _corr(a, b) -> float:
    """Pearson's r, or NaN when either side is CONSTANT -- which is an answer, not an error.

    A constant term is the failure this harness exists to catch twice over: ranking by it is
    ranking by nothing. `np.corrcoef` would return NaN with a warning and a reader could mistake
    it for a missing label, so it is named here instead.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() <= 1e-12 or b.std() <= 1e-12:
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1])


class Report:
    def __init__(self) -> None:
        self.ok = True

    def verdict(self, good: bool, text: str) -> None:
        self.ok &= bool(good)
        print(f'  {"OK " if good else "BAD"} {text}')


def read_labels(path: Path) -> List[dict]:
    rows = []
    with open(path) as fh:
        for n, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f'{path}:{n}: not JSON ({exc})')
    return rows


def require(rows: List[dict], keys) -> None:
    missing = sorted({k for k in keys for r in rows if k not in r})
    if missing:
        raise SystemExit(f'the label rows are missing {", ".join(missing)} -- see the module '
                         f'docstring for the schema this term needs')


# --- visible_fraction ------------------------------------------------------------------------

def score_visible_fraction(rows: List[dict], face, rep: Report, cos_bug: bool = False) -> None:
    require(rows, ('mask_area_px', 'depth_m', 'fx', 'fy', 'truth_area_px', 'truth_radius_px'))
    mine, truth, m_area, t_area, m_circle, t_circle, tilts = [], [], [], [], [], [], []
    for r in rows:
        K = np.array([[float(r['fx']), 0.0, 0.0], [0.0, float(r['fy']), 0.0], [0.0, 0.0, 1.0]])
        full = face.expected_area_px(float(r['depth_m']), K)
        if full is None:
            raise SystemExit('the face has no configured size, so `visible_fraction` cannot be '
                             'computed at all -- pass --expected-radius-m')
        v = visible_fraction(float(r['mask_area_px']), full)
        if cos_bug:
            # THE HISTORICAL BUG, on purpose (--demo-cos-bug): divide the occlusion out by the
            # cosine of the fit's tilt. Correct in a textbook, and it is what this harness exists
            # to catch -- no unit test could, because each one fed the function its own tilt.
            v = min(1.0, v / max(math.cos(math.radians(float(r.get('tilt_deg', 0.0)))), 1e-3))
        mine.append(v)
        truth.append(min(float(r['truth_area_px']) /
                         (math.pi * float(r['truth_radius_px']) ** 2), 1.0))
        m_area.append(float(r['mask_area_px']))
        t_area.append(float(r['truth_area_px']))
        m_circle.append(full)
        t_circle.append(math.pi * float(r['truth_radius_px']) ** 2)
        tilts.append(float(r.get('tilt_deg', 0.0)))

    mine, truth = np.array(mine), np.array(truth)
    m_area, t_area = np.array(m_area), np.array(t_area)
    m_circle, t_circle = np.array(m_circle), np.array(t_circle)
    tilts = np.array(tilts)
    print(f'  {len(mine)} labelled detections over {len({r["frame"] for r in rows})} frames; '
          f'the labels report a tilt of {np.median(tilts):.0f} deg median, '
          f'{np.percentile(tilts, 90):.0f} p90')

    print('\n1/4 the MASK the detector produced is the area the labeller marked')
    r1 = float(np.corrcoef(m_area, t_area)[0, 1])
    rat = float(np.median(m_area / np.maximum(t_area, 1e-9)))
    rep.verdict(r1 > 0.85 and abs(rat - 1.0) < 0.10,
                f'correlation {r1:+.3f}, ratio {rat:.3f} -- if this failed, nothing downstream '
                f'of the mask could be trusted')

    print('2/4 the full-face area the term divides by is the one the object really has')
    rat2 = float(np.median(m_circle / np.maximum(t_circle, 1e-9)))
    rep.verdict(abs(rat2 - 1.0) < 0.05,
                f'ratio {rat2:.3f}, from the depth and the configured face radius against the '
                f"labeller's own disc -- this is where a unit mix-up shows")

    print('3/4 and so the fraction agrees with the truth')
    r3 = float(np.corrcoef(mine, truth)[0, 1])
    err = float(np.median(np.abs(mine - truth)))
    rep.verdict(r3 > 0.85 and err < 0.08,
                f'correlation {r3:+.2f}, median error {err:.3f}, '
                f'p90 {np.percentile(np.abs(mine - truth), 90):.3f}')

    print('4/4 it does NOT saturate -- the failure the cos(tilt) version had')
    lo = truth < 0.8
    sat = float(np.mean(mine > 0.99))
    med_lo = float(np.median(mine[lo])) if lo.any() else float('nan')
    rep.verdict(bool(sat < 0.45 and lo.any() and med_lo < 0.8),
                f'{100*sat:.0f} % of detections read 1,00 against the truth\'s '
                f'{100*np.mean(truth > 0.99):.0f} %, and on the {int(lo.sum())} objects the truth '
                f'calls less than 80 % visible this says {med_lo:.2f} '
                f'(the cos(tilt) version said 0,96)')


# --- box_overlap -----------------------------------------------------------------------------

def score_box_overlap(rows: List[dict], rep: Report) -> None:
    require(rows, ('box_px', 'truth_visible_fraction'))
    by_frame: Dict[str, List[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        by_frame[str(r.get('frame', ''))].append(i)
    mine = np.zeros(len(rows))
    for idxs in by_frame.values():
        ov = box_overlap([rows[i]['box_px'] for i in idxs])
        for i, v in zip(idxs, ov):
            mine[i] = v
    truth = np.array([float(r['truth_visible_fraction']) for r in rows])
    areas = np.array([(float(r['box_px'][2]) - float(r['box_px'][0])) *
                      (float(r['box_px'][3]) - float(r['box_px'][1])) for r in rows])
    print(f'  {len(rows)} labelled detections over {len(by_frame)} frames; overlap median '
          f'{np.median(mine):.2f}, p90 {np.percentile(mine, 90):.2f}')

    print('\n1/3 every value is a fraction of its OWN box')
    rep.verdict(bool(np.all((mine >= 0.0) & (mine <= 1.0))),
                f'range {mine.min():.3f} .. {mine.max():.3f}')

    print('2/3 a box alone in its frame is not overlapped')
    alone = [i for idxs in by_frame.values() if len(idxs) == 1 for i in idxs]
    rep.verdict(not alone or bool(np.all(mine[alone] == 0.0)),
                f'{len(alone)} frames hold one detection' if alone else
                'no single-detection frame in this set -- nothing to check')

    print('3/3 it tracks the labelled occlusion NEGATIVELY, and only partly -- which is why the '
          'face term exists beside it')
    r3 = float(np.corrcoef(1.0 - mine, truth)[0, 1]) if mine.std() > 0 else float('nan')
    rep.verdict(np.isfinite(r3) and r3 > 0.2,
                f'(1 - overlap) against the labelled visible fraction: r = {r3:+.2f}. MEASURED on '
                f'a real bin: +0,54 -- half the information is in one term and not the other, so '
                f'a correlation near 1 here would mean the two terms were redundant')
    small = areas < np.median(areas)
    if small.any() and (~small).any():
        print('    note: the smaller half of the boxes reports '
              f'{np.median(mine[small]):.2f} median overlap against '
              f'{np.median(mine[~small]):.2f} for the larger half -- the asymmetry is deliberate: '
              'what matters is how much of THIS object is involved with another')


# --- edge_on -----------------------------------------------------------------------------------

def score_edge_on(rows: List[dict], max_tilt_deg: float, rep: Report,
                  constant_tilt: bool = False) -> None:
    """Score the EDGE-ON term against labelled tilts (ADR-0018 Decision 5).

    Two failures are specific to this term and neither is visible to a unit test, because a unit
    test feeds the function a tilt it computed itself -- the same blindness that let the cos(tilt)
    correction through:

      * THE CONSTANT. A 3D method that claims no orientation reports a tilt of 0 for every
        candidate, so the term reads 0 everywhere and the rule multiplying by it is a rule
        multiplying by one. `--demo-constant-tilt` reproduces it, and check 3 is what catches it.
      * THE SATURATION. Set `tool_max_tilt_deg` tighter than the bin's own tilt distribution and
        every candidate reads 1,00: the rule is then 0 for all of them, ties publish the
        detector's own order, and the ranking has quietly stopped being one. Measured on a real
        bin the fit's tilt runs to a median of 33 degrees and a p90 of 63, so a 20-degree tool
        limit saturates most of the frame. Check 4.
    """
    require(rows, ('tilt_deg', 'truth_tilt_deg'))
    fit_tilt = np.array([float(r['tilt_deg']) for r in rows])
    truth = np.array([float(r['truth_tilt_deg']) for r in rows])
    read = np.zeros(len(rows)) if constant_tilt else fit_tilt
    mine = np.array([edge_on(t, max_tilt_deg=max_tilt_deg) for t in read])
    print(f'  {len(mine)} labelled detections over {len({r["frame"] for r in rows})} frames; the '
          f'labels report a TRUE tilt of {np.median(truth):.0f} deg median, '
          f'{np.percentile(truth, 90):.0f} p90, against a tool limit of {max_tilt_deg:.0f} deg')

    print('\n1/5 the tilt the term READS is the tilt the labeller MEASURED')
    r1 = _corr(read, truth)
    err = float(np.median(np.abs(read - truth)))
    rep.verdict(np.isfinite(r1) and r1 > 0.85 and err < 10.0,
                f'correlation {r1:+.3f}, median error {err:.1f} deg -- the term is only ever as '
                f'good as the fit\'s normal, so this is the check that matters most')

    print('2/5 it is LINEAR IN THE ANGLE: no cosine anywhere, which is the whole decision')
    unsat = read < max_tilt_deg
    if unsat.sum() < 3:
        rep.verdict(False, f'only {int(unsat.sum())} of {len(read)} candidates are below the '
                           f'{max_tilt_deg:.0f} deg tool limit, so the SHAPE of the term cannot '
                           f'be checked at all -- raise --tool-max-tilt-deg')
    elif read[unsat].std() <= 1e-9:
        rep.verdict(False, 'every candidate reads the SAME tilt, so the shape of the term cannot '
                           'be checked -- and check 3 says why that is the real problem')
    else:
        fitted = np.polyfit(read[unsat], mine[unsat], 1)
        resid = float(np.max(np.abs(np.polyval(fitted, read[unsat]) - mine[unsat])))
        cosine = 1.0 - np.cos(np.radians(read[unsat]))
        gap = float(np.max(np.abs(cosine - mine[unsat])))
        rep.verdict(resid < 1e-9,
                    f'largest deviation from a straight line {resid:.2e}; a 1 - cos(tilt) shape '
                    f'would have differed from these values by up to {gap:.3f}')

    print('3/5 it DISCRIMINATES -- the failure a method that claims no orientation would have')
    rep.verdict(bool(mine.std() > 0.05 and len(np.unique(np.round(mine, 3))) >= 3),
                f'spread {mine.std():.3f} over {len(np.unique(np.round(mine, 3)))} distinct '
                f'values; a constant term means the rule using it is multiplying by a constant '
                f'and the published order is whatever the model returned')

    print('4/5 it is not SATURATED over this bin -- the failure `clearance` was measured to have')
    sat = float(np.mean(mine >= 1.0))
    rep.verdict(sat < 0.45,
                f'{100*sat:.0f} % of candidates are at or past the tool limit ({max_tilt_deg:.0f} '
                f'deg). Past 45 % the rule is 0 on most of the frame, ties publish the '
                f"detector's own order, and the ranking has stopped being one")

    print('5/5 it is a TERM and not a CORRECTION: it carries what the area term does not')
    if any('truth_visible_fraction' in r for r in rows):
        vis = np.array([float(r.get('truth_visible_fraction', float('nan'))) for r in rows])
        good = np.isfinite(vis)
        r5 = _corr(1.0 - mine[good], vis[good])
        rep.verdict(not np.isfinite(r5) or abs(r5) < 0.9,
                    f'(1 - edge_on) against the labelled visible fraction: r = {r5:+.2f}. A '
                    f'correlation near 1 would mean the tilt was already inside the area term '
                    f'and this factor only counted it twice -- which is what dividing '
                    f'visible_fraction by cos(tilt) did, saturating it at 1,00 on 65 % of '
                    f"detections where the truth said 28 %")
    else:
        print('    skipped: these labels carry no truth_visible_fraction, so whether the two '
              'terms are redundant cannot be answered here')


# --- the harness self-test -------------------------------------------------------------------

def write_demo_labels(path: Path, n_frames: int = 12, per_frame: int = 7, seed: int = 7) -> Path:
    """Synthetic labels, to prove THIS SCRIPT runs. It validates nothing about a real camera.

    One scene, scored by both terms, built so that nothing in it is computed the way a term
    computes it:

      * a pile of overlapping discs with a random depth order, each disc's TRUE visible fraction
        measured by sampling points inside it and asking whether a NEARER disc covers them --
        circle geometry with occlusion, which is not the axis-aligned box arithmetic
        `box_overlap` does nor the area ratio `visible_fraction` does;
      * the truth in the labeller's own units (pixels), while the term works from metres and
        intrinsics -- so check 2/4 still cross-checks the unit handling, a real class of error;
      * a TRUE tilt independent of the occlusion, and the tilt a fit would report for it. The
        independence is deliberate: it is what lets the edge-on term's fifth check show that the
        tilt carries something the area term does not.

    What a synthetic scene cannot check is the thing only a renderer can: whether a real mask is
    the area a real label marked, and whether a real fit's depth is the depth the object was at.
    """
    rng = np.random.default_rng(seed)
    fx = fy = 424.0
    radius_m = 0.02095
    rows = []
    for f in range(n_frames):
        z = float(rng.uniform(0.50, 0.75))
        r_px = fx * radius_m / z
        # a pile: centres within a few radii of each other, so faces really do cover each other
        cx = rng.uniform(0.0, 5.0 * r_px, per_frame) + 120.0
        cy = rng.uniform(0.0, 5.0 * r_px, per_frame) + 120.0
        order = rng.permutation(per_frame)           # order[i] small = nearer the camera
        for i in range(per_frame):
            # the TRUE visible fraction: sample the disc, drop what a nearer disc covers
            ang = rng.uniform(0.0, 2.0 * math.pi, 3000)
            rad = r_px * np.sqrt(rng.uniform(0.0, 1.0, 3000))
            px, py = cx[i] + rad * np.cos(ang), cy[i] + rad * np.sin(ang)
            covered = np.zeros(px.shape, bool)
            for j in range(per_frame):
                if j == i or order[j] >= order[i]:
                    continue                          # only a NEARER disc can cover this one
                covered |= (px - cx[j]) ** 2 + (py - cy[j]) ** 2 <= r_px ** 2
            v = float(1.0 - covered.mean())
            truth_area = v * math.pi * r_px * r_px
            # a detector that measures the visible area to a few per cent
            mask_area = max(1.0, truth_area * float(rng.normal(1.0, 0.03)))
            # a TRUE tilt drawn over the range a real bin shows (median 33 deg, p90 63), and the
            # tilt a fit would have reported for it -- the plane normal is good to a degree or
            # two, which is what the edge-on term is fed
            truth_tilt = float(rng.uniform(0, 70))
            fit_tilt = float(np.clip(truth_tilt + rng.normal(0.0, 2.5), 0.0, 90.0))
            rows.append(dict(frame=f'demo_{f:03d}', mask_area_px=mask_area, depth_m=z,
                             fx=fx, fy=fy, truth_area_px=truth_area, truth_radius_px=r_px,
                             tilt_deg=fit_tilt, truth_tilt_deg=truth_tilt,
                             box_px=[float(cx[i] - r_px), float(cy[i] - r_px),
                                     float(cx[i] + r_px), float(cy[i] + r_px)],
                             truth_visible_fraction=v))
    with open(path, 'w') as fh:
        for r in rows:
            fh.write(json.dumps(r) + '\n')
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--term', default='visible_fraction',
                    choices=['visible_fraction', 'box_overlap', 'edge_on'])
    ap.add_argument('--labels', type=Path, help='JSON Lines label file (see the module docstring)')
    ap.add_argument('--face', default='disc', help='grasp face the term divides by')
    ap.add_argument('--expected-radius-m', type=float, default=0.0,
                    help="the face's configured radius, in metres")
    ap.add_argument('--tool-max-tilt-deg', type=float, default=45.0,
                    help='the tilt the TOOL can still hold -- where `edge_on` saturates. A tool '
                         'property, which is why it has no framework-chosen value')
    ap.add_argument('--demo-cos-bug', action='store_true',
                    help='reproduce the cos(tilt) correction this term was first written with, '
                         'to show the harness catching it')
    ap.add_argument('--demo-constant-tilt', action='store_true',
                    help='reproduce the OTHER way the edge-on term fails: computed from a 3D '
                         'method that claims no orientation, so the tilt is 0 for every '
                         'candidate and the rule multiplies by a constant')
    ap.add_argument('--demo', action='store_true',
                    help='write synthetic labels and score them: a self-test of this harness, '
                         'NOT a validation of the term')
    ap.add_argument('--min-rows', type=int, default=MIN_ROWS)
    args = ap.parse_args()

    if args.demo:
        tmp = Path(os.environ.get('TMPDIR', '/tmp')) / 'trainit_score_term_demo.jsonl'
        args.labels = write_demo_labels(tmp)
        if args.expected_radius_m <= 0:
            args.expected_radius_m = 0.02095
        print(f'DEMO: synthetic labels in {args.labels}. This exercises the harness; it says '
              f'NOTHING about a real camera.\n')
    if not args.labels:
        ap.error('--labels is required (or --demo)')

    rows = read_labels(args.labels)
    if len(rows) < args.min_rows:
        print(f'  only {len(rows)} labelled detections -- not enough to score a term '
              f'(need {args.min_rows})')
        return 1
    print(f'scoring `{args.term}` against {args.labels}')
    rep = Report()
    if args.term == 'visible_fraction':
        score_visible_fraction(rows, make_face(args.face,
                                               {'expected_radius_m': args.expected_radius_m}),
                               rep, cos_bug=args.demo_cos_bug)
    elif args.term == 'edge_on':
        if args.tool_max_tilt_deg <= 0:
            raise SystemExit('--tool-max-tilt-deg must be positive: at 0 the term is switched '
                             'off, and `rules.computable_terms` then leaves `edge_on` out '
                             'altogether rather than scoring a constant')
        score_edge_on(rows, args.tool_max_tilt_deg, rep,
                      constant_tilt=args.demo_constant_tilt)
    else:
        score_box_overlap(rows, rep)
    print(f'\n{args.term.upper()} MEANS WHAT IT SAYS' if rep.ok else
          f'\n{args.term.upper()} DOES NOT (see BAD above)')
    return 0 if rep.ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
