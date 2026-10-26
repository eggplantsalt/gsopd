# GS-OPD route-cache handoff and bounded capture report (2026-10-26)

## Result

The reusable RRT route handoff is implemented and its exact-match path was exercised in L2T2. The cached route bypassed a duplicate planning call, but real OSC execution stagnated before preplace. Per the bounded plan, capture then moved to `obstacle_avoidance_human/L0T2` using GS-OPD's own L0T2/L1T2 success experience. That branch achieved a stable bilateral grasp and lift, then advanced a real placement approach by about 0.51 m; it stopped about 85 mm from preplace. Neither scene reached native BDDL success, full release, or stable placement. **New complete Task-Success Episodes: 0.**

All totals below come from archived action records and run reports. Total wall-clock duration was not persisted reliably by the executor, so it is marked unavailable rather than inferred from inconsistent filesystem timestamps. The human L0T2 grasp report does persist time-to-first stable grasp: 454.416 s.

## Per-scene results

| Scene | Native success / predicates | Physical actions | Physical branches | Student / expert actions | Experience reuse | Elapsed time and outcome |
|---|---|---:|---:|---:|---|---|
| `obstacle_avoidance/L2T2` | `task_success=false`; Pot1 On=true, Pot2 On=false, Stove Turnon=true | 42 | 1 | 0 / 42 | Reused the GS-OPD L1T2 object/Region-relative keypose route; exact cached 7D qpos + EEF paths. Historical source actions were not replayed. | Total duration unavailable; cached route planning was 11.427 s. Followed 65 aligned qpos/EEF samples with exact identity match, then stagnated after 42 OSC actions. |
| `obstacle_avoidance_human/L0T2` | `task_success=false`; Pot1 On=false, Pot2 On=false, Stove Turnon=true | 575 (215 grasp/lift + 360 transport) | 2 (one grasp, one placement) | 0 / 575 | Reused GS-OPD L0T2 and L1T2 complete-success keypose experience plus the existing moka-pot grasp/transfer skill. L1T2-derived placement candidate; no source actions replayed. | Total duration unavailable; first verified stable grasp at 454.416 s. Real placement approach advanced 0.508 m and ended about 85 mm from preplace. One follow-up RRT from the failed state ran no physical actions because current qpos was outside controller joint limits. |

No task-specific scripts, fixed scene poses, or manual pose adjustments were added. Placement candidates remained task-equivalent and were derived through the generic target-region interface. Both runs used real MuJoCo dynamics and controls. No robot/object state was written directly, no BDDL predicate was altered, no official demonstrations were used, and training was not started.

## Reusable route handoff

The route-cache record uses schema `gs_opd_precomputed_rrt_route.v1` and stores the aligned 7D joint path and world EEF path. Reuse requires exact match of model geometry SHA, active BDDL SHA, exact start-state SHA, goal pose, safety-geometry policy SHA, planner configuration SHA, and path integrity hash. A changed identity invalidates the relevant route and requests replanning only for that segment. The cache is a planning artifact, not a safety certificate.

For the L2T2 physical attempt, the reused route identity SHA was `943fc33111f23eaea648ed158245a9a77938a8387aeb4a80696ce592ea022e9c`; path SHA was `8ce8a9bdc3a529687cebeb574385842ca8127cd46cd82544a41ea534210a5f24`. All inputs matched the restored starting state, model, BDDL, goal, policy, and planner. The route archive retained 65 qpos rows and 65 EEF rows. The live closed-loop controller physically executed 42 actions (1,050 expected and captured contact-monitoring substeps) before `transport_preplace_dense_RRT_path_waypoint_stagnated_for_24_actions`. Native predicates remained 2/3. The branch archive includes the failed Runtime Snapshot, dual-camera video, compressed per-step records, NPZ actions/states, and durable records/frames.

This validates the data handoff and strict cache invalidation path. It does **not** establish that a cached route is dynamically trackable to its endpoint: exact route reuse removed repeat planning, while OSC still failed to reach preplace.

## Human L0T2 evidence and limits

The reset Snapshot was checked against model SHA `633b8b0956f2f38e7f1c62067608f42743d0bccdc6bfeaca11976b5c7f301606` and BDDL SHA `2c4010e44700b0d2069dcd47f489a7411ec364f31cfd69141801b49db3398af7`. The first expert branch established bilateral contact on Pot1 and lifted it 34.404 mm in 215 actions. It was resumed from that exact saved Snapshot for placement. The 360-action physical approach retained the grasp and moved the EEF from `[-0.1848, 0.2175, 1.0474]` m to `[-0.2018, -0.2335, 1.2794]` m, while the selected preplace target was `[-0.1606, -0.2010, 1.3461]` m. The branch stopped at its action budget; it had not released Pot1, and the native task remained false.

One bounded continuation attempt from that failed Runtime Snapshot made no control actions: the live start qpos failed the existing controller joint-limit check. Joint limits were preserved. The final failure is therefore two distinct, evidenced gates: the chosen OSC waypoint did not reach preplace within 360 actions, then the changed failed-state configuration could not enter the current RRT controller. It is not evidence that the task is physically impossible.

## Monitoring, safety, and archive

The L2T2 transport branch records 42/42 actions and 1,050/1,050 expected contact samples. Human L0T2 records 215 grasp/lift actions with 5,375 expected samples and 360 transport actions with 9,000 expected samples. The grasp archive flags 49 actions as incomplete substep monitoring; all 360 placement records also carry `substep_monitoring_complete=false` despite recording the expected sample count. These runs are **not** fully monitoring-complete and none is safety-certified. They remain useful real-physics task-search evidence, not validated training Episodes.

Machine-readable report: [GS_OPD_CROSS_SCENE_CAPTURE_REPORT_20261026.json](GS_OPD_CROSS_SCENE_CAPTURE_REPORT_20261026.json).

Full external artifacts are under `/opt/data/private/gs_opd/rollouts/gs_opd_route_cache_handoff_20261026/`:

- L2T2 reused-route attempt: `task_success_first_executor_obstacle_avoidance_L2T2_20261026_200327_567640623/`.
- Human L0T2 grasp/lift and its handoff Snapshot: `human_L0T2/task_success_first_executor_obstacle_avoidance_human_L0T2_20261026_200726_204334329/`.
- Human L0T2 real transport branch and failed Snapshot: `human_L0T2/task_success_first_executor_obstacle_avoidance_human_L0T2_20261026_202154_293951476/`.
- Human L0T2 zero-action bounded continuation: `human_L0T2/task_success_first_executor_obstacle_avoidance_human_L0T2_20261026_203410_126057230/`.

## Shared capability gap

Route identity and full-path reuse now work, but the current capture chain lacks a robust generic bridge from a geometrically planned path to a long physical move under OSC feedback. In L2T2, a reused 65-state route still stagnated under closed-loop OSC; in human L0T2, a bounded live OSC fallback made substantial progress but did not finish, and the resulting posture was not admissible to the current RRT controller. The next general improvement should preserve task-equivalent placement choices while making the planned route executable under feedback and recoverable after partial physical progress. Do not mask this gap by increasing a stagnation threshold or changing joint limits.
