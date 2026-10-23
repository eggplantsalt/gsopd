# Automatic task-phase execution

This L0T2 task adapter connects the generic recovery state machine to live
task predicates and object-relative expert priors. It does not change the
Student action space, BDDL, controller semantics, object state, or standard
1000-step evaluation horizon.

## What the task adapter runs

The configured phase controller advances only from Bridge observations and
native BDDL evidence:

`pregrasp → approach → close_gripper → stable_grasp → lift → transport → precise_placement → release → withdrawal → settle → second_object_recovery → verify_task`

Approach and placement targets are recomputed from the current registered
object and native BDDL region poses. The verified archived grasp template
provides an object-frame grasp transform and approach distance; it supplies no
copied action sequence. The task configuration records the order, target
region poses, support geometry, and provenance. These are curated L0T2
historical priors, not VLM Teacher output. A missing native predicate,
uncertain grasp, unavailable geometry, untrusted fixture activation, or
unknown physical-safety sample stops the run.

RRT-Connect handles the pregrasp, object-to-object transport, second-object
recovery, and withdrawal routes. It starts from the current live 7D qpos and
checks complete robot and carried-object collision geometry. Near-object
grasp, descent, release, and settle use the native 7D OSC_POSE interface. Each
real control transition is checked across all 25 physics substeps before it
can count as safe. Only bilateral live target-finger grasp evidence advances
the grasp phase; release requires the native `On` predicate and a stable
object. Final task success requires the native task checker. The L0T2
configuration has no trusted stove-activation controller, so an unexpected
false `Turnon` predicate fails closed.

## Reproducing runs

Use the LIBERO-Safety environment and an unused local port. The CLI checks
ownership with `ss` before starting a Bridge and refuses an occupied port.
Every run needs a new output directory.

```bash
/workspace/zy_workspace/LIBERO-Safety/.venv/bin/python \
  -m research.gs_opd.repair.automatic_recovery_cli \
  --mode t10-snapshot \
  --config research/gs_opd/configs/l0t2_automatic_task_recovery_t10_seed.json \
  --output-dir /opt/data/private/gs_opd/rollouts/<new-run-directory> \
  --bridge-port <unused-port>
```

This mode restores the verified timestep-10 runtime Snapshot, replays only the
archived Student prefix through the risk predictor, and hands control to the
task expert after takeover. It loads no Student checkpoint.

```bash
/workspace/zy_workspace/LIBERO-Safety/.venv/bin/python \
  -m research.gs_opd.repair.automatic_recovery_cli \
  --mode episode \
  --config research/gs_opd/configs/l0t2_automatic_task_recovery.json \
  --output-dir /opt/data/private/gs_opd/rollouts/<new-run-directory> \
  --bridge-port <unused-port>
```

Episode mode starts from native t=0 and loads the existing Pi0.5 checkpoint.
The configured task expert takes over only after the Student risk monitor or
native task-phase routing requests it. Values above 1000 steps are explicitly
demonstration-only and are limited to 1800; benchmark evaluation remains at
1000.

## Current validation boundary

The archived timestep-238 state previously labeled pregrasp is not a safe
start under the full target-object geometry rule: the closest non-finger robot
geom was 0.419 mm from Pot 2 against a 10 mm clearance requirement. The engine
rejected it before executing any action. The pregrasp goal was corrected to
use the verified object-relative approach distance and live Pot pose.

From the verified timestep-10 state, predictive risk takeover occurred before
executing the archived Student prefix. The engine then executed 118 real
OSC_POSE transitions, with complete 25-substep safety records and no unsafe
transition; the measured minimum forbidden-object clearance was 36.534 mm.
It selected three RRT-Connect routes over three planning attempts
and replanned from live state after tracking deviation, a risk boundary, and
narrow-channel entry. The third replan request exhausted the configured
pregrasp planning-attempt budget while the EEF was still 232.6 mm from the
pregrasp goal. No grasp phase or placement phase was reached.

This is a local, safe executed prefix only. It is not a local-recovery success,
task success, complete safety success, standard-horizon success, or safe
demonstration. Since pregrasp continuation did not complete, the dependent
native t=0 full-task run was not launched. The traces and Snapshots remain in
their unique run directories and are not demonstration candidates.

Each run used a separately spawned and closed Bridge process. Port 38731 was
used for the archived t=238 pregrasp audit, 38732 for read-only geometry
diagnostics, and 38733 for the t=10 continuation runs. Port ownership was
checked before each Bridge start. The second t=10 run reused 38733 only after
the preceding owned Bridge had exited and the port was confirmed free; no
pre-existing user process was stopped or modified.
