# ft-diloco — Build Log

> The chronological development narrative + lessons learned. This is the spine the
> blog post is drafted from: each milestone records the goal, what we built, what we
> measured, and — most valuably — what *surprised* us. Negative results are kept on
> purpose; they're the honest core of the story. Living document, appended as we go.

**One-line thesis:** train a real language model across cheap, unreliable, geographically
scattered machines that sync only occasionally (DiLoCo), and show that killing /
partitioning / re-adding machines mid-run doesn't break convergence — built on
[torchft](https://github.com/meta-pytorch/torchft), demonstrated on a home gaming PC, a
spare desktop, and ~$3 of rented cloud GPU.

**The cluster:** worker4 (Ryzen 9 5950X + RTX 3060 12GB, the GPU trainer) · worker1
(8-core CPU box, the lighthouse coordinator) · gigabit home ethernet · + Vast.ai cloud
GPUs over a Tailscale mesh for the real-WAN milestones. Dev on a Mac, executed over SSH.

---

## M0 — Baseline & harness

**Goal:** plain single-GPU training of a small model, plus the eval/plotting harness
everything else hangs off.

**Built:** a minimal nanoGPT-class decoder (`src/ftdiloco/model.py`), a uint16-memmap
data pipeline (`data.py`, TinyStories tokenized with the GPT-2 BPE), a JSONL telemetry
schema (`metrics.py`) that *every* downstream analysis reads, and a single training
entrypoint (`train.py`) used for both baseline and DiLoCo modes.

**Result:** tiny50m (51M params) converged to eval loss **1.677 / ppl 5.35** on
TinyStories, ~43k tok/s on the 3060, tight 3-seed noise band (±0.001). This became the
reference bar for every later parity claim.

**Surprise / lesson:** the first GPU run OOM'd at batch 32×512. The culprit wasn't the
model — it was the **cross-entropy transient over a 50k-token vocab** (logits tensor
balloons). Fix: same effective batch via micro-batch 8 × grad-accum 4, which shrinks the
logits 4×. *Lesson: at small model scale the vocab/CE memory dominates, not the weights.*

---

## M0.5 — Does torchft actually work on our rig?

**Goal:** de-risk the foundation before building on it. Install torchft, run its own
example, kill a node by hand.

**Result:** the kill/rejoin loop works. `kill -9` a replica → the survivor commits its
next sync solo within one sync period, quorum shrinks 2→1 with zero stall; relaunch it →
torchft P2P-recovers it from the survivor and it rejoins. **30/30 state digests (model
params + outer Nesterov momentum) bit-identical post-rejoin** — the single most important
early result, because it answers the project's central open question (*"what happens to
the outer optimizer's momentum when a worker leaves and rejoins?"*) empirically: it's
recovered, exactly, with no checkpoint.

**Surprises / lessons (the torchft integration was full of undocumented sharp edges):**
- `Manager` hard-requires torchrun-style env (`MASTER_ADDR`/`MASTER_PORT`) *and a
  self-hosted TCPStore* — it connects as a client (`is_master=False`) and blocks forever
  with no error if nothing is hosting the store. We host it ourselves in `ft.py`.
- We steer around two live torchft bugs from day one: #316 (async-quorum SIGSEGV → we use
  sync quorum) and #323 (PGTransport timeout ineffective → we use HTTPTransport).
- *Lesson: a maintainer-blessed "experimental" API can still be a minefield; budget real
  time for integration even when the headline feature exists.*

---

## M1 — DiLoCo parity + the communication win

**Goal:** prove DiLoCo trains *as well as* the baseline while communicating far less;
sweep the sync interval H.

**Result (the parity/comm table):**

| sync every H | 25 | 50 | 100 | 200 | 500 |
|---|---|---|---|---|---|
| Δ eval loss vs baseline | +2.8% | +4.7% | +6.3% | +7.4% | +9.4% |
| comm vs per-step DP | 25× less | 50× | 100× | 200× | 500× |

477/477 syncs committed, zero failures. Comm reduction is exactly H-fold; *measured* wire
bytes (veth counters in a network-namespace harness) matched the analytic model within
+2% at H≤100.

**Surprises / lessons:**
- The measured-vs-analytic comm gap *grows* at large H — and the residual is a **constant
  ~0.5 GB/run control-plane floor** (lighthouse heartbeats + quorum gRPC), which only
  dominates once the gradient payload shrinks. A nice incidental measurement of torchft's
  coordination overhead.
- We report the parity gap *honestly*: the small-scale M≥2 DiLoCo penalty is real and
  documented (outer-lr was left untuned across H), not hidden. *Lesson: a believable
  result names its own caveats.*

---

## M2 — The money shot (chaos engineering)

**Goal:** the visceral demo — training running, a worker killed on camera, the loss
curve keeps descending.

**Built:** a Python chaos harness (`chaos/`) — a scheduled controller injecting *real*
OS-level faults (kill, SIGSTOP straggler, network partition via link-down, throttle,
relaunch), logging a ground-truth `chaos.jsonl`. Plus precise recovery-latency
definitions (`T_detect`, `T_resume`, `T_rejoin`) computed by fusing three independent log
sources.

**Result:** a 6-fault scripted run measured `T_detect = T_resume = 5.0s`, `T_rejoin =
54s` end-to-end, **84/84 post-rejoin digests bit-identical** on the GPU model, and the
run finished within **+0.6%** of fault-free loss. Shipped the kill-a-node GIF.

**Surprises / lessons:**
- The GIF took **6 takes**. Hard-won recording lessons: capture tmux pane ids explicitly
  (`-P -F`, tmux renumbers on split), size the *outer* recorder session, trim the cast
  tail, and — the big upgrade — replace hand-rolled ANSI with a **Rich + plotext live
  dashboard** (real loss chart with axes, a "hole" in the line where a worker dies, status
  badges TRAINING→DEAD→RECOVERING). asciinema records on Linux, `agg` renders to GIF on
  the Mac. *Lesson: the demo is half the deliverable; leverage Rich/plotext, don't
  hand-roll terminal graphics.*

---

## M3 — Failure storms, and the best finding in the project

**Goal:** survive not one kill but dozens; plot step-efficiency vs failure rate (our
scaled-down echo of torchft's "Llama-3 1B through 1,100 failures at 82.3% step
efficiency").

**Result:** two ~45-minute Poisson storms (kills + stragglers + partitions, a fault every
~45s). **88.2% and 85.0% step efficiency — both above torchft's 82.3% bar** — at
comparable-or-higher fault rates, on two consumer replicas, with zero manual intervention.

**The negative result we kept (and it's the strongest #171 evidence in the whole repo):**
the *first* storm round looked healthy by throughput (86.7% sync efficiency) but the model
*regressed* mid-storm (eval 2.4 → 4.0). Mechanism: under restart churn at small replica
counts, a kill landing while the only other member is alive-but-unhealed leaves a
**fresh-init worker as a singleton quorum — its random weights silently become the cluster
state**, and the victim heals *from that* (we caught heals from a donor at manager step 0).
**Live P2P recovery is necessary but not sufficient under churn.** torchft's 30-group
production setup makes this practically unreachable; few-big-member cross-datacenter DiLoCo
(exactly issue #171's regime) walks straight into it.

**Fix shipped:** commit-coupled checkpoints (each replica persists state every K commits;
restarts init from the newest checkpoint) + a healthy-donor invariant in the chaos harness.
Second storm round: monotonic convergence restored. *Lesson: measure progress, not just
throughput — they can diverge silently, and that divergence is the actual research.*

---

## M4 — WAN realism & real cross-region cloud

**Goal:** show the bandwidth story under a throttled link, then on the real internet.

**Home-lab WAN sweep (netem, 20ms RTT):** per-step sync (the DDP pattern) degrades 5.3k →
0.9k tok/s from 1 Gbps to 50 Mbps and *fails entirely* at 10 Mbps; DiLoCo H=100 holds
flat at 27–37k tok/s. Two findings: (1) even at *gigabit*, syncing every step is 7× slower
than DiLoCo on the identical link; (2) at 10 Mbps the failure is *structural* — the ~200s
allreduce **starves torchft's own control plane** (heartbeats share the pipe → quorum
times out → cascade). Plus a real two-physical-machine run (3060 GPU + CPU box over house
ethernet): 4/4 cross-machine commits, bit-identical digests on heterogeneous hardware, and
a fresh gotcha — torchft's *separate* `quorum_timeout` (60s default) silently kills
heterogeneous clusters whose workers reach their first sync minutes apart.

**Real cloud (the war story):** worker4 (home) + Vast.ai RTX 4090 VMs over a Tailscale
mesh. The **smoke test passed** — home 3060 + Virginia 4090 trained one cluster across the
open internet (~52ms RTT), 2-participant committed sync, **bit-identical digests across the
WAN** (`a573c3de`). The headline run: base124m converged ~11 → ~2.2 eval loss across
US-home + US-East cloud. **Total cloud spend: $3.12.**

**Surprises / lessons — the cloud was where infra reality bit hardest:**
- **Docker instances can't run this stack.** Vast docker containers only get Tailscale
  *userspace-netstack*, which fails at three layers (store bind, self-dial, outbound). You
  need a *real VM* (`vastai/kvm` template) with `/dev/net/tun` and systemd.
- A pile of Vast-specific gotchas now baked into `scripts/cloud/` + `docs/cloud.md`: API
  needs account 2FA; `--env` lands in `/etc/environment` not the onstart shell; SSH via the
  direct public IP + mapped port (the proxy refuses on VMs); ephemeral VMs reuse IP:port so
  `UserKnownHostsFile=/dev/null`; the VM image ships `authorized_keys` with perms sshd
  refuses (the onstart now chmods it). The fully-automated bootstrap *does* work end-to-end
  once those are handled.
- **The sync-alignment finding** (directly relevant to #171): with `min_replica_size=1` and
  nodes that start at different times *or run at different speeds* (4090 ~0.5s/step vs 3060
  ~2s/step), each worker hits its H=100 boundary at a different wall-clock moment, finds no
  peer, and commits **solo** — the "cluster" silently degrades to N independent runs that
  each converge to *different* optima (digests don't match, unlike the smoke test). True
  collaborative averaging needs aligned starts + homogeneous speed, *or* a barrier
  (`min_replica_size≥2`). *Lesson: connectivity ≠ coordination. Semi-synchronous training
  across heterogeneous commodity nodes needs explicit sync-point alignment — which is
  exactly the question the upstream RFC is asking.*

---

## Meta / process lessons (the stuff between the milestones)

These are the operational lessons that aren't in any milestone but cost real time — the
"how we actually worked" that makes a build-log honest.

- **The flaky-link tax.** The Mac↔worker4 Tailscale path flaps under load and eventually
  died entirely; everything long-running lives in **tmux on the worker**, big transfers are
  `rsync --partial` in a retry loop, and we built `scripts/w4.sh` — a retry-until-sentinel
  runner with a **ProxyJump fallback through worker1's LAN** when the direct path is down.
  *Lesson: assume the control channel is unreliable; make every operation idempotent and
  resumable, and never hand-drive a long job over a bare SSH session.*
- **`pkill -f` is a footgun.** A pattern matching the remote command line happily kills the
  SSH/bash running it — and, embarrassingly, during the cloud pass I repeatedly killed *my
  own running workers* with diagnostic `pkill -f ftdiloco.train`. Use run-scoped patterns
  and dedicated kill scripts.
- **Nested quoting through ssh→tmux→python is the other footgun.** Anything non-trivial
  goes in a *script file* that tmux runs, not an inline triple-nested string.
- **The cloud juggling doesn't scale by hand.** The context-expensive part of M4 was
  hand-driving individual nodes. The discipline for the upcoming N=32 storm: **monitor the
  lighthouse (one aggregate query), not N nodes; fully script the fleet lifecycle;
  over-provision and let bad-boot nodes count as faults.**
- **Cost discipline works.** Across every cloud experiment — including a dozen false-start
  instances — total spend was $3.12 of a $50 tranche, because we destroy (never stop)
  instances immediately and keep a ledger in `docs/cloud.md`.

---

## What torchft owes us back (parked until reviewed)

`docs/findings-171.md` accumulates every reproduced friction point as candidate
upstream contributions — headlined by the `HTTPTransport.address()` hardcoded-hostname bug
(breaks P2P recovery behind NAT, which is *the* #171 scenario) for which we run a clean
subclass workaround. Per the owner's instruction, **no upstream contact happens until after
the blog post and an explicit review** — this section is inventory, not action.

---

## M5 — the N=32 failure storm (one desktop, 32 replicas)

**Goal:** match torchft's ~30-group framing — show the *coordination* machinery (a
many-manager lighthouse quorum, P2P recovery, commit/rollback, the barrier) survives chaos
at a dozens-of-replicas scale.

**The first decision was where to run it.** The instinct (and the earlier plan) was cheap
cloud nodes. We didn't: M4 already proved the WAN *transport* works across NAT'd, heterogeneous,
geographically-split nodes (bit-identical digests over the open internet). The N=32 *unknowns*
were all coordination questions — does the lighthouse scale to 32 managers, does a 32-way gloo
ring form and commit, does the `min_replica_size≥2` barrier hold without OOM — and those answer
on **one commodity desktop** for **$0**, with perfectly controllable chaos and zero cloud
juggling. (That juggling was the context-expensive part of M4; not repeating it was an explicit
goal.) So: 32 replica groups as CPU/gloo processes in per-replica netns on one Ryzen 9 5950X,
lighthouse on the same host, a micro (~3.3M-param) model. *Micro because 32 full models don't fit
commodity RAM and the storm tests coordination, which is model-size-agnostic — itself the honest
commodity-hardware point: at small scale you co-locate, so the model must be small.*

**Built:** a generalized N-replica harness — `run_storm.sh` (was hardcoded to 2 GPU replicas),
`launch_storm_replica.sh`, a single `supervisor_n.sh` watching all N, the `storm_micro` config —
plus an **aggregate** monitor (`storm_status.py`) that prints one fixed-size summary regardless of
N. Fixed a real bug surfaced by scaling: `kill_safe`'s healthy-donor search defaulted to
`n_replicas=2`, so at N=32 it only ever checked replicas {0,1}; plumbed `--replicas` through the
chaos controller.

**De-risk ladder (N=4 → 8 → 32), all free, all on home hardware:**
- **N=4 settled the OOM.** The M4 `min_replica_size≥2` OOM that "needed investigation" was an
  **artifact** — six orphaned relaunches piling up ~30 GB on a dirty box. On a clean box the barrier
  runs at ~6 GB: quorum forms at 4, `outer_step` spread 0 (perfect lockstep), a kill recovers in 42 s.
- **N=8** confirmed the lighthouse coordinates 8 managers cleanly (the quorum log lists all 8 with
  distinct store addresses) and an 8-way ring commits **133/133** through 8 kills, all recovered.
- **N=32** is feasible at ~25 GB working set; the 28 GB we first saw was the *simultaneous* torch-import
  peak (32 processes importing at once), which settles back down. One memory scare — a 31.5 GB transient
  when a relaunch cluster and eval coincided — but **0 OOM the entire project**.

**Then the real surprise: liveness ≠ participation.** The first N=32 storm showed median quorum
**16/32**. The tell: the *fault-free* reference showed median **17** too. So the low participation
wasn't chaos — it was **CPU oversubscription**. 32 compute-heavy processes on 16 physical cores get
timesliced unevenly by the scheduler, so their step times drift and only a subset reach each H-step
barrier together; the soft barrier (`min_replica_size=2`) commits with whoever's there and the fast
subset races on.

**The fix is CPU pinning — and it taught us something.** `taskset -c <core>` *looked* applied but the
python leaf still showed affinity `0-31`: **torch/MKL reset CPU affinity on import**, silently
overriding it. So we pin **in-process** (`os.sched_setaffinity` after torch loads) and **re-assert it
every sync** (torch resets again when it spawns gloo collective threads). With 1:1 pinning, step times
go uniform and **fault-free participation jumps 17 → 32** (the full cluster reaches every barrier
together).

**But under the storm, participation stays ~16 even pinned** — and that's the honest, #171-relevant
finding. Each fault triggers a quorum reconfiguration that knocks the survivors' barrier timing back
out of phase, and at **125 faults/hr** the cluster never re-aligns. (Dense eval did the same thing —
each replica's eval steals variable CPU and desyncs it — so the headline run uses light eval.) Meanwhile
**~30/32 replicas stay alive and training the whole time**: liveness is high, participation is gated by
sync-alignment under churn, and those are *different numbers*. *Lesson: at a dozens-of-replicas scale a
low `min_replica_size` plus a steady fault rate means "alive" and "in this sync's quorum" diverge — the
soft-barrier alignment tax. The cluster size is still 32; the per-sync averaging breadth is a separate,
honestly-reported property.*

**Result (CPU-pinned, 125 faults/hr, 30 min):** committed-sync throughput holds at **97.7% of fault-free**,
**97% of sync attempts commit**, **all 27 kills recover** (T_back median 149 s, T_resume 57 s), the global
eval loss descends **10.8 → 4.3 monotonically through 28 kills**, and **0 OOM**. Pinning also lifted
commit *reliability* 93.7% → 97% (the stable aligned subset never fails a barrier).

**Dashboards had to be rethought for scale.** The old `dashboard.py` rendered one panel *per worker* —
fine at 2, unreadable at 32. Two replacements, both fixed-size at any N and sharing one event-derivation
module (`storm_events.py`): a live aggregate TUI (`storm_dash.py` — a colored replica-state grid + quorum
sparkline + fault feed) and a **telemetry-reconstructed** GIF (`storm_gif.py`). The GIF needs *no live
recording*: every fault and commit is timestamped, so the time axis is ours to compress (30 min → 26 s)
and the encoding ours to choose (a 32-cell swarm where you can literally watch the purple healing-tail
that explains why participation trails liveness). *Lesson: at scale, reconstruct visualizations from
ground-truth telemetry rather than screen-recording a live dashboard — more legible and fully reproducible.*

---

## Next

M5 polish and the blog write-up (`nicholicaron.github.io`, ternfpga-post quality bar). Per the owner's
instruction, the upstream-to-Meta step waits until after the blog and an explicit review. This log is the
spine; `docs/findings-171.md` is the torchft-facing evidence.
