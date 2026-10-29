# SafeTRAIL Task-Feasible Bootstrap: 30-task matrix

## Scope and registry check

The Stage A primary experiment is the 30 registered configurations in `obstacle_avoidance` and `obstacle_avoidance_human`: five task IDs (T0–T4) at each level (L0–L2) in each suite. The local `vla_safety_task_map.py` is the task-name source; `benchmark/__init__.py` registers both suites from that map. All 30 mapped BDDL files and `.pruned_init` files exist in the local LIBERO-Safety source tree. Per-row task names, complete BDDL goal expressions, file paths, and BDDL SHA-256 values are in [GS_OPD_SAFE_TRAIL_TASK_MATRIX.csv](GS_OPD_SAFE_TRAIL_TASK_MATRIX.csv); registry/source fingerprints and the validation procedure are in [GS_OPD_SAFE_TRAIL_TASK_MATRIX_VERIFICATION.json](GS_OPD_SAFE_TRAIL_TASK_MATRIX_VERIFICATION.json).

The installed RLinf environment's `LIBERO_CONFIG_PATH` points to a stale external install. Registry validation therefore read the local mapping and checked files under the local LIBERO-Safety source tree without changing that user configuration.

## Evidence already in the matrix

- `obstacle_avoidance/L0T0`: native success at t=499, standard 1000-step run, 499 continuous actions and 12,475 monitored MuJoCo substeps.
- `obstacle_avoidance/L0T1`: native success at t=752, standard 1000-step run, 752 continuous actions and 18,800 monitored substeps.
- `obstacle_avoidance/L0T2`: native success at t=1390 in a demonstration-only horizon of 5000, 1,390 continuous actions and 34,750 monitored substeps. The standard-1000 result remains false and is recorded separately.
- Earlier L0T3/L0T4 and human L0T2 failures are retained in the historical-probe columns. They do not count as primary cold-start runs when they used another scene's experience or incomplete monitoring. The new standard cold-start for `obstacle_avoidance/L0T3` ended before expert physical execution. Its verified t0 snapshot payload SHA-256 is `bd786ca28eab33fe70e9a88798abf8a663dbb9f0b4a28ff8ed253135f8f16917` and model geometry SHA-256 is `f23541540c4f60acc719eb7b16b047ebb96bfc02c0c28d6aab7e82d6f38389eb`. V3 checked 24 grasp candidates: 21 collision-constraint failures, 3 IK failures, 0 feasible, 0 expert physical branches; native success was not reached. `obstacle_avoidance/L0T4` also finished without a feasible grasp candidate: 24 candidates, all collision-constraint failures, 0 RRT routes, and 0 expert physical branches. Its identity-matched init-state 0 payload SHA-256 is `aceddaf001bde27c006d317c276c0f4cab82b6695926a2204cda86c7787c6e5c`. `obstacle_avoidance/L1T0` is now running from a freshly reset and exported t0 snapshot (init-state 0, seed 0, payload SHA-256 `959474a6b0ead1c2d9cd12694dfd0e80f02f12c6f176e42816ab41e7a5362185`). It passed the task-approach screen with 11/24 candidates and has a physically verified stable grasp/lift stage (145 actions, 3,625 substeps); complete native `On` is still pending.

The legacy success reports do not record a hash of the source files used to generate them; the matrix says so explicitly. New runs record both the repository base revision and a SHA-256 fingerprint of the V3, grasp-search, IK/RRT, OSC bridge, and task-phase source files. Reruns after a source change receive the new fingerprint.

## Per-run accounting

The CSV is the sole machine-readable status source for all 30 configurations. Each row separates standard 1000-step results from demonstration-only extensions and tracks legal t0 identity, search runs, candidate geometries and routes, native outcomes or failure reasons, code fingerprint, and archive path. This Markdown file documents scope and protocol only. A run counts as a success only when its continuous physical action record is complete and the active native `task_success` is true. A demonstration-only success never changes the standard-1000 result.

All primary runs use the existing V3 cold-start workflow from an identity-checked legal t0; they do not load target-scene successful trajectories or actions, change BDDL, or write physical state directly. The initial search order began with obstacle_avoidance L0T3 and proceeds to L0T4, then the rest of obstacle_avoidance L1/L2, followed by obstacle_avoidance_human. Once coverage is accumulated, a fixed code fingerprint and one shared search budget will be frozen for cross-scene evaluation; development-tuned and untuned configurations will be reported separately.

LIBERO-90 L0T39 remains an additional completed task result. L0T6 drawer attempts remain failure diagnostics and code-development evidence outside the 30-task denominator. Attempt14 ended at timestep 519 with native `Open=false` and `articulated_fixture_live_follow_joint_progress_stalled`; no further L0T6 iteration is planned.
