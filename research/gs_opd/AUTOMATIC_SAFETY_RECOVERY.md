# Automatic Safety Recovery Engine

This engine runs the Student, risk monitor, recovery planner, native OSC_POSE
controller, and task handoff as one bounded state machine. Its reusable logic is
in `repair/automatic_recovery.py`; the L0T2 LIBERO adapter is in
`repair/automatic_recovery_runtime.py`, and the Safety Bridge owns MuJoCo
geometry, snapshots, isolated prefix checks, and physical execution.

## Run

Use a fresh output directory for every run. The archived t=10 run resumes from
the verified t=10 runtime snapshot; the episode mode starts the unchanged
Student in native L0T2 state 0. `runtime-snapshot` resumes the path configured
by `artifacts.t10_runtime_snapshot` in the JSON config.

```bash
cd /workspace/zy_workspace/RLinf-gs-opd-overfit-recovery
export PYTHONPATH="$PWD"
/workspace/zy_workspace/RLinf/.venv/bin/python \
  -m research.gs_opd.repair.automatic_recovery_cli \
  --mode t10-snapshot
```

```bash
/workspace/zy_workspace/RLinf/.venv/bin/python \
  -m research.gs_opd.repair.automatic_recovery_cli \
  --mode episode
```

The CLI creates a timestamped fresh output directory under
`/opt/data/private/gs_opd/rollouts/` by default. It uses the unchanged π₀.₅
checkpoint, CUDA device 0, the L0T2 config below, and isolated Safety Bridge
port 38569. Override these with
`--student-checkpoint`, `--student-device`, `--config`, `--python`, or
`--bridge-port` when needed. A completed run writes a run config, event log,
Observation–Action NPZ, task/safety summary, per-timestep restorable MuJoCo
snapshots, and `agentview.mp4` / `wrist.mp4` videos. No run directory is reused.

## State machine and safety contract

The engine validates a task context before control: task identity, BDDL-derived
forbidden contacts, permitted target contacts, live grasp state, phase goals,
and the source of each goal. A missing binding, unknown grasp, missing goal,
unknown collision result, exhausted budget, or unimplemented phase ends in a
fail-closed state. It does not infer a goal from an image or claim that a VLM
Teacher supplied one.

For each Student 7D action chunk, the Bridge previews a configured control
prefix on an isolated runtime snapshot. The prefix advances native MuJoCo
physics and checks all robot collision Geoms, contact state, OSC tracking,
joint state, and 25 physical substeps per control step. It commits no preview
state. If risk reaches the configured threshold, the engine obtains a phase
goal, solves bounded IK from current state, plans 7D qpos with RRT-Connect, and
checks full robot geometry at dense configurations. A candidate route must
preserve the hard 10 mm margin for forbidden objects; the planner can prefer
routes with additional clearance but cannot lower the floor.

Native 7D actions remain unchanged. Recovery segments use OSC_POSE and are
checked against the active RRT qpos reference, EEF pose, task progress, and
actual complete geometry. Unsafe or incomplete prefixes are discarded. The
engine stops and replans from the current real state on risk, narrow-channel
entry, excessive path deviation, stagnation, or control error, subject to the
replan and time budgets. Successful segments continue the same route. At a
verified goal it hands control back to the Student or a declared task expert;
it then checks native task success and complete safety evidence separately.

When a registered object is verified as held by opposing Panda finger
contacts, the Bridge measures the current EEF-to-object transform. Scratch RRT
configurations move the object's free root with the candidate EEF pose and
check its full collision geometry against classified scene objects. Isolated
OSC previews and committed execution check the object's actual simulated pose
at every physical substep. A task can add exact carried-object/target contact
pairs only when those target contacts are explicitly allowed. Missing or
incomplete carried-object geometry is an error. This generic path supports
carried-object obstacle avoidance; a restored L0T2 held-Pot-2 runtime state
returned complete carried-object geometry for every classified scene group
with 204.2 mm minimum clearance to its forbidden objects. It has not yet been
validated on L0T0.

## L0T2 task inputs and current limitation

L0T2 constraints come from its active native BDDL predicates and verified
compiled geometry. Ginger Beer and Place Box keep a 10 mm minimum clearance.
The initial recovery goal is Pot 2 pregrasp, reconstructed from a historical
successful grasp record and the live Pot 2 pose. That is an existing expert
prior, not VLM Teacher output. Current configuration marks `grasp`,
`transport`, `placement`, and `second_object_recovery` as unresolved goal
phases. Therefore the engine can finish the automatic obstacle-avoidance
recovery and Student handoff, but it stops explicitly at `grasp` until a
verified current-state grasp goal/controller is supplied. It never pastes the
historical action sequence into the current episode.

`configs/l0t2_automatic_recovery.json` holds the segment size, risk distance,
10 mm hard floor, maximum replans, planning and execution budgets, tracking
limits, task phases, goal provenance, and source artifact fingerprints. The
standard horizon is 1000 and extension is disabled. A task-specific config
must identify every phase goal and allowed target contact; unresolved stages
remain fail-closed.

## Verification results (2026-10-23)

- Archived t=10 Snapshot run: 228 recovery controls reached the historical Pot
  2 pregrasp goal at episode t=238 after three replans. Local recovery passed.
  After handoff, its previous implementation retried Student risk and stopped
  on an unknown forecast; that handoff loop was fixed. A resumed t=238 check
  now detects the reached goal before executing Student actions and fails
  explicitly at unresolved phase `grasp` with zero new controls.
- Full Episode run: started at native t=0, Student risk triggered recovery at
  t=10, then the engine autonomously replanned twice as it approached narrow
  geometry. It reached the Pot 2 pregrasp goal after 258 controls at t=258.
  `local_recovery_success=true`; full task and safety success are false because
  phase `grasp` has no verified goal/controller. Standard 1000-step horizon
  success is false; no horizon extension was used.
- Regression suite: 19 unit tests pass, including engine budgets/handoff,
  route safety, isolated-branch coverage, carried-object MuJoCo scratch motion,
  trace writing, and Snapshot restore.
- Held-object Bridge integration check: restored an archived stable Pot 2
  grasp, verified live opposing-finger grasp state, and measured the carried
  object against all classified scene groups without advancing physics. The
  geometry query was complete; this validates the L0T2 carried binding path,
  not a L0T0 rollout.

Run directories:

- t=10 recovery: `/opt/data/private/gs_opd/rollouts/automatic_safety_recovery_l0t2_t10_20261023_run6/`
- resumed post-handoff fail-closed check: `/opt/data/private/gs_opd/rollouts/automatic_safety_recovery_l0t2_posthandoff_20261023_run8/`
- complete Episode: `/opt/data/private/gs_opd/rollouts/automatic_safety_recovery_l0t2_episode_20261023_run9/`
- held-object geometry check: `/opt/data/private/gs_opd/rollouts/automatic_safety_recovery_carried_geometry_smoke_20261023_attempt2/`

The episode trace, videos, events, and snapshots are retained in those
directories. They are recovery evidence, not safety-success demonstrations;
none is eligible for safety-success training. Existing 1093-step data and
historical snapshots were not overwritten, and no LoRA training was started.

Re-run the core regression tests with:

```bash
PYTHONPATH="$PWD" /workspace/zy_workspace/RLinf/.venv/bin/python -m unittest \
  research.gs_opd.repair.test_automatic_recovery \
  research.gs_opd.repair.test_automatic_recovery_runtime \
  research.gs_opd.repair.test_jointspace_rrt \
  research.gs_opd.repair.test_carried_object_geometry \
  research.gs_opd.data.test_trajectory_writer \
  research.gs_opd.smoke.test_safety_bridge_snapshot -v
```
