"""The scorer ships WITH the term, and it has teeth (ADR-0017 Decision 6).

A candidate term that nobody scores against something external is a number, not a measurement.
These tests do not score the term -- external labels do that, and only a twin or a real cell can
supply them. They check that the harness runs and that it FAILS on the bug it exists for, because
a check that cannot fail is decoration.
"""
import os
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / 'scripts' / 'score_term.py'


def run(tmp_path, *args):
    env = dict(os.environ, TMPDIR=str(tmp_path), PYTHONPATH=str(SCRIPT.parent.parent))
    return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True,
                          env=env, timeout=300)


def test_the_scorer_is_shipped_beside_the_terms():
    assert SCRIPT.is_file(), 'a term without its scorer silently stops being measured'


def test_it_runs_and_passes_on_labels_the_term_should_satisfy(tmp_path):
    r = run(tmp_path, '--demo', '--term', 'visible_fraction')
    assert r.returncode == 0, r.stdout + r.stderr
    assert '4/4' in r.stdout and 'BAD' not in r.stdout


def test_it_CATCHES_THE_COS_TILT_BUG_which_no_unit_test_could(tmp_path):
    """The historical failure: dividing the occlusion out by cos(tilt) saturated the term at 1,00
    on two thirds of detections. Every unit test passed, because each fed the function a tilt it
    had computed itself."""
    r = run(tmp_path, '--demo', '--demo-cos-bug')
    assert r.returncode == 1, r.stdout + r.stderr
    assert 'DOES NOT' in r.stdout
    assert 'does NOT saturate' in r.stdout


def test_it_catches_a_face_radius_that_is_simply_WRONG(tmp_path):
    r = run(tmp_path, '--demo', '--expected-radius-m', '0.030')
    assert r.returncode == 1, r.stdout + r.stderr
    assert 'ratio' in r.stdout


def test_box_overlap_is_scored_too_and_only_PARTLY_tracks_the_truth(tmp_path):
    r = run(tmp_path, '--demo', '--term', 'box_overlap')
    assert r.returncode == 0, r.stdout + r.stderr
    assert 'MEASURED on a real bin: +0,54' in r.stdout


def test_the_EDGE_ON_term_is_scored_against_LABELLED_TILTS(tmp_path):
    """ADR-0018 Decision 5 ships the term with its scorer, like every other one."""
    r = run(tmp_path, '--demo', '--term', 'edge_on')
    assert r.returncode == 0, r.stdout + r.stderr
    assert '5/5' in r.stdout and 'BAD' not in r.stdout
    assert 'LINEAR IN THE ANGLE' in r.stdout


def test_it_CATCHES_A_CONSTANT_TILT_which_is_ranking_by_NOTHING(tmp_path):
    """The other way this term fails, and the one no unit test sees: computed from a 3D method
    that claims no orientation, the tilt is 0 for every candidate and the rule using it is
    multiplying by a constant."""
    r = run(tmp_path, '--demo', '--term', 'edge_on', '--demo-constant-tilt')
    assert r.returncode == 1, r.stdout + r.stderr
    assert 'DOES NOT' in r.stdout
    assert 'DISCRIMINATES' in r.stdout


def test_it_catches_a_TOOL_LIMIT_tighter_than_the_bins_own_tilts(tmp_path):
    """Set the limit under the bin's median tilt and every candidate saturates at 1,00: the rule
    is then 0 on most of the frame and ties publish the detector's own order."""
    r = run(tmp_path, '--demo', '--term', 'edge_on', '--tool-max-tilt-deg', '20')
    assert r.returncode == 1, r.stdout + r.stderr
    assert 'SATURATED' in r.stdout
    # and a limit the bin fits inside passes
    assert run(tmp_path, '--demo', '--term', 'edge_on',
               '--tool-max-tilt-deg', '70').returncode == 0


def test_a_switched_off_tool_limit_is_an_ERROR_rather_than_a_scored_constant(tmp_path):
    r = run(tmp_path, '--demo', '--term', 'edge_on', '--tool-max-tilt-deg', '0')
    assert r.returncode != 0
    assert 'switched off' in (r.stdout + r.stderr)


def test_a_sample_too_small_to_say_is_an_ERROR_and_not_a_pass(tmp_path):
    labels = tmp_path / 'few.jsonl'
    labels.write_text('{"frame": "a", "mask_area_px": 10, "depth_m": 0.6, "fx": 424, "fy": 424, '
                      '"truth_area_px": 10, "truth_radius_px": 15}\n')
    r = run(tmp_path, '--labels', str(labels), '--expected-radius-m', '0.02095')
    assert r.returncode == 1
    assert 'not enough to score' in r.stdout


def test_a_label_file_missing_a_column_says_WHICH(tmp_path):
    labels = tmp_path / 'bad.jsonl'
    labels.write_text(''.join('{"frame": "a", "mask_area_px": 10, "depth_m": 0.6}\n'
                              for _ in range(60)))
    r = run(tmp_path, '--labels', str(labels), '--expected-radius-m', '0.02095')
    assert r.returncode != 0
    assert 'truth_area_px' in (r.stdout + r.stderr)
