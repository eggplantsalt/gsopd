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

The old run9 pregrasp pose and the new phase-controller pregrasp pose are
different by about 53.2 mm. The old goal allowed Pot 2 contact, while the new
pregrasp target allows no target-object contact. Its source is the archived
object-frame grasp/approach transform retargeted to the live Pot pose; this is
an expert prior, not a VLM Teacher output.

The t=10 comparison run accepted three rolling RRT routes with minimum
forbidden clearances 19.571, 18.485, and 18.448 mm. Tracking deviation,
risk-margin approach, and narrow-channel entry triggered replans after
separate control segments. The former three planning-call phase cap stopped
the run at t=128, 232.6 mm from the goal. That hard call-count cap is removed:
each RRT invocation still has at most three routes; productive replans are
counted only after real physical progress and capped at 12 per phase / 72
overall. Repeated identical reasons and physical stagnation still fail closed.
Risk-boundary rearming uses measured geometry: enter at 50 mm and rearm only
after measured clearance reaches 60 mm. Every committed transition still has
complete 25-substep checks and the hard 10 mm margin.

The archived run9 t=238 and t=258 states fail the current Pot 2 full-robot
clearance rule at 0.419 mm and 2.277 mm. A scan of the archived observations
and contact records selected t=240 as a safe resumable state: Pot 2 clearance
30.9 mm and Ginger Beer / Place Box clearance at least 24.4 mm. Strict
Snapshot restore and current-policy checks passed. From t=240 the new
controller reached pregrasp in 30 real OSC_POSE controls, then executed 18
safe approach controls through t=287.

At t=287, isolated Snapshot preview predicted a 9.83 mm robot link5–Ginger
Beer clearance for the next control prefix. That prefix was rejected before
commit. RRT then checked three provenance-backed approach/grasp variants:
the archived source pose, a 1.634 mm live-geometry-directed lateral shift,
and a 0.04 rad archived approach-axis rotation. All three goal configurations
were rejected by the full-robot endpoint check as unsafe. The run stopped
after that bounded candidate set. It did not close the gripper, verify a stable
bilateral grasp, or lift the pot by 30 mm.

Historical phase evidence was also reviewed: source π₀.₅ Pot 2 grasp at
t=118–141; a 425-control Pot 2 placement correction verifying only the Pot 2
On subgoal; a 135-control Grasp Retargeting grasp/lift trial; and a 239-control
final placement continuation reaching t=1093 with both On predicates,
Turnon, and native task success. That final continuation used a 1300-step
demonstration-only horizon and its source prefix contains Ginger Beer contact,
so it is not a safety-success demonstration. These records provide grasp,
object-relative pose, and stage-control priors only. New actions are generated
from live observations and object poses.

Because the isolated grasp/lift gate did not pass, dependent t=10 task
continuation and native t=0 full Episode execution were not started. Pot 2
placement, Pot 1 pick/place, fixture activation, and complete native predicate
acceptance were not run. Task success, complete safety success, standard
1000-step horizon success, and extended-horizon demonstration success are all
false. No safety-success candidate was created; existing demonstrations and
LeRobot data were preserved; LoRA was not run.

All 174 repair unit tests passed. The changed Python modules compile. Detailed
evidence remains in the independent run directories:
- run9 source trajectory and per-step Snapshots:
  /opt/data/private/gs_opd/rollouts/automatic_safety_recovery_l0t2_episode_20261023_run9/
- t=10 three-call-cap regression comparison:
  /opt/data/private/gs_opd/rollouts/automatic_task_execution_l0t2_t10_seed_20261023_attempt2/
- strict t=238/t=258 rejection, safe t=240 restore, and t=287 stage continuation:
  /opt/data/private/gs_opd/rollouts/automatic_task_execution_l0t2_run9_grasp_validation_20261023_attempt1/ through attempt5/
- three bounded unsafe grasp endpoints:
  /opt/data/private/gs_opd/rollouts/automatic_task_execution_l0t2_approach_replan_20261023_attempt2/

Each experiment used an independent Bridge after checking that its port was
free, and closed only the Bridge process it owned. Earlier validations used
ports 38731–38733; the current bounded recovery runs used independent ports
38741–38747. No pre-existing user process was stopped or modified.
