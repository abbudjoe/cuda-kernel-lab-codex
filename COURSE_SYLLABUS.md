# Course Syllabus: CUDA Kernel Development for AI Inference and Robotics

## Course purpose

This is a 24-week self-directed course for learning CUDA kernel development with Codex as tutor and reviewer. The course emphasizes practical skill: correctness, measurement, profiling, optimization, and technical communication.

The goal is not to become a CUDA expert overnight. The goal is to build a visible, credible body of work that supports a transition toward scientific software, ML systems, inference engineering, GPU performance, and robotics-oriented compute.

## Learning outcomes

By the end, you should be able to:

1. Explain the CUDA execution model: grids, blocks, threads, warps, memory hierarchy, synchronization, and streams.
2. Write simple CUDA kernels and reason about correctness.
3. Benchmark kernels using CUDA events and input-size sweeps.
4. Use profiler output to form performance hypotheses.
5. Identify memory-bound, compute-bound, launch-bound, synchronization-bound, and occupancy-limited kernels.
6. Apply common optimization patterns: coalescing, tiling, shared memory, reduction patterns, warp-level primitives, loop unrolling, and fusion.
7. Implement inference-relevant kernels such as softmax, LayerNorm/RMSNorm, activation fusion, and simple attention components.
8. Integrate a custom CUDA op with PyTorch at a basic level.
9. Implement robotics-oriented GPU kernels such as depth-to-pointcloud, voxel filtering, collision scoring, and trajectory rollout scoring.
10. Produce portfolio-quality technical writeups without exaggerating results.

## Weekly rhythm

A sustainable cadence:

- Session A: concept and naive implementation;
- Session B: tests and benchmark harness;
- Session C: profiling and optimization experiment;
- Session D: notes, review, and quiz.

If you are burned out, do three short sessions instead of four. Consistency beats intensity.

## Course map

| Stage | Weeks | Theme | Output |
|---|---:|---|---|
| 0 | 0-1 | Setup, mental model, workflow | environment report, first build, learning contract |
| 1 | 1-4 | CUDA fundamentals | vector add, reduction, transpose, scan/histogram basics |
| 2 | 5-8 | Profiling and performance patterns | matmul, memory coalescing, shared memory, Nsight notes |
| 3 | 9-12 | AI inference kernels | softmax, LayerNorm/RMSNorm, fused ops |
| 4 | 13-16 | Framework integration | PyTorch custom CUDA op, Triton comparison, inference benchmark |
| 5 | 17-20 | Robotics GPU kernels | point clouds, voxelization, collision checking, trajectory rollout |
| 6 | 21-24 | Capstone and portfolio | final demo, report, README, interview prep |

---

# Stage 0 — Setup and learning contract

## Weeks 0-1

### Goals

- Confirm hardware/software environment.
- Build and run the starter vector-add lesson.
- Learn the course loop.
- Establish benchmark and note-taking discipline.

### Deliverables

- `profiler_reports/environment.json`
- successful or documented failed build attempt;
- `notes/week_00_setup.md`
- first Codex quiz results.

### Codex seed prompt

```text
Read AGENTS.md, COURSE_SYLLABUS.md, RUBRIC.md, and docs/hardware_and_setup.md.
Act as my tutor for Stage 0.
Do not write CUDA code yet.
First summarize the course workflow, inspect the repo, and give me a setup checklist.
Then ask me to run the first environment command and paste the result.
```

### Stage gate

You pass Stage 0 when you can explain:

- what `nvcc` does;
- what host code vs device code means;
- why correctness checks must run before benchmarks;
- why CUDA event timing is preferable to naive wall-clock timing for kernel timing.

---

# Stage 1 — CUDA fundamentals

## Week 1: Vector add and timing

### Concepts

- host vs device;
- kernel launch syntax;
- memory allocation and copies;
- grid/block indexing;
- bounds checks;
- CUDA event timing.

### Deliverables

- run `lessons/01_vector_add`;
- modify input sizes;
- record timings in `lessons/01_vector_add/BENCHMARKS.md`;
- explain whether the kernel is likely memory-bound.

### Codex prompt

```text
Tutor me through vector add.
Explain the host/device flow and kernel indexing in lessons/01_vector_add/vector_add.cu.
Then ask me to predict the bottleneck before I benchmark it.
Do not optimize yet.
```

## Week 2: Reductions

### Concepts

- parallel reduction;
- shared memory;
- `__syncthreads()`;
- race conditions;
- tree reductions;
- warp divergence.

### Deliverables

- naive reduction;
- shared-memory reduction;
- benchmark comparison;
- notes on synchronization.

### Codex prompt

```text
Create a step-by-step reduction lesson.
First explain why reductions are harder than vector add.
Then scaffold the files if needed, but leave the kernel body as TODOs.
I will implement the first naive version before you review it.
```

## Week 3: Matrix transpose

### Concepts

- coalesced reads vs coalesced writes;
- shared-memory tiling;
- bank conflicts;
- input-size sweeps.

### Deliverables

- naive transpose;
- tiled transpose;
- benchmark table;
- profiler notes on memory throughput.

### Codex prompt

```text
Plan a matrix transpose lesson focused on memory coalescing.
Ask me to predict why the naive transpose will be slow.
Do not implement the tiled version until after I benchmark the naive version.
```

## Week 4: Scan or histogram

Choose one.

### Option A: prefix sum / scan

Good for learning parallel dependencies.

### Option B: histogram

Good for learning atomics and contention.

### Deliverables

- CPU reference;
- naive CUDA;
- one optimization or contention analysis;
- writeup explaining limitations.

### Codex prompt

```text
Help me choose between scan and histogram for Week 4 based on what I struggled with in Weeks 1-3.
Then create a lesson plan with correctness tests and a profiling question.
```

---

# Stage 2 — Profiling and performance patterns

## Week 5: Benchmark discipline

### Concepts

- warmups;
- CUDA event timing;
- synchronization;
- input-size sweeps;
- throughput metrics;
- benchmark reproducibility.

### Deliverables

- reusable benchmark harness improvements;
- `docs/benchmarking_methodology.md` updated with your own notes;
- environment metadata captured.

### Codex prompt

```text
Review my benchmark methodology across lessons so far.
Find timing pitfalls, missing metadata, and cases where my benchmark may be misleading.
Suggest a minimal standard benchmark template for the rest of the course.
```

## Week 6: Nsight Compute basics

### Concepts

- profiling workflow;
- achieved occupancy;
- memory throughput;
- warp execution efficiency;
- register pressure;
- roofline-style reasoning at a basic level.

### Deliverables

- one Nsight Compute report or CLI output for a prior kernel;
- notes interpreting at least three metrics.

### Codex prompt

```text
Act as my profiler coach.
I will paste Nsight Compute output for one kernel.
Help me interpret it and classify the bottleneck.
Then propose two optimization experiments with expected metric changes.
```

## Week 7: Tiled matrix multiplication

### Concepts

- arithmetic intensity;
- shared-memory tiling;
- block dimensions;
- register reuse;
- baseline comparison.

### Deliverables

- naive matmul;
- tiled matmul;
- benchmark vs CPU and maybe cuBLAS only as a reference;
- limitations writeup.

### Codex prompt

```text
Plan a tiled matrix multiplication lesson.
The goal is not to beat cuBLAS; the goal is to understand tiling, data reuse, and arithmetic intensity.
Create a staged plan and wait for me before implementing.
```

## Week 8: Optimization review week

### Goals

- Revisit one earlier lesson.
- Improve one benchmark or optimization.
- Write a portfolio-quality explanation.

### Deliverables

- revised README for one lesson;
- review checklist completed;
- quiz passed.

### Codex prompt

```text
Review Stage 1 and Stage 2 work as a portfolio reviewer.
Pick the weakest lesson and explain what would make it credible to a GPU systems hiring manager.
Then give me a prioritized fix list.
```

---

# Stage 3 — AI inference kernels

## Week 9: Numerically stable softmax

### Concepts

- row-wise reductions;
- max-subtraction stability;
- exponentials;
- memory traffic;
- transformer-relevant shapes.

### Deliverables

- CPU softmax;
- naive CUDA softmax;
- optimized row-wise softmax;
- error tolerance analysis.

### Codex prompt

```text
Plan a softmax lesson for inference.
Include numerical stability, row-wise reductions, benchmark shapes, and correctness tolerances.
Do not write the optimized kernel until I implement and benchmark the naive version.
```

## Week 10: LayerNorm or RMSNorm

### Concepts

- mean/variance reductions;
- epsilon and numerical stability;
- vectorized loads where appropriate;
- inference shapes.

### Deliverables

- CPU reference;
- CUDA implementation;
- benchmark vs PyTorch if available;
- notes on memory bandwidth.

### Codex prompt

```text
Tutor me through RMSNorm or LayerNorm as an inference kernel.
Explain the math, then ask me to identify which reductions are needed per row.
Scaffold tests before implementation.
```

## Week 11: Fused bias + activation

### Concepts

- kernel fusion;
- launch overhead;
- memory traffic reduction;
- activation functions;
- benchmarking composed vs fused operations.

### Deliverables

- separate kernels;
- fused kernel;
- benchmark comparison;
- explanation of when fusion helps.

### Codex prompt

```text
Create a lesson on kernel fusion using bias + activation.
Force me to estimate memory reads/writes for separate vs fused kernels before coding.
Then help me benchmark whether the estimate matches reality.
```

## Week 12: Inference kernel mini-project

Choose one:

- optimized softmax writeup;
- RMSNorm custom kernel;
- fused operation benchmark;
- small attention component.

### Deliverables

- portfolio-quality README;
- benchmark CSV;
- profiler interpretation;
- known limitations.

### Codex prompt

```text
Help me turn my best Stage 3 kernel into a mini portfolio project.
Review the code, benchmarks, notes, and README.
Flag anything that sounds exaggerated or insufficiently supported.
```

---

# Stage 4 — Framework integration

## Week 13: PyTorch custom op architecture

### Concepts

- Python wrapper;
- C++ extension;
- CUDA implementation;
- operator registration;
- build system;
- tensor shape and dtype checks.

### Deliverables

- working or scaffolded PyTorch extension;
- tests comparing against PyTorch reference;
- benchmark script.

### Codex prompt

```text
Explain the architecture of a PyTorch custom CUDA op.
Then scaffold a minimal extension around my best inference kernel.
Leave the core kernel implementation as my responsibility unless I ask for review.
```

## Week 14: Autograd or inference-only boundary

### Concepts

- inference-only vs training op;
- backward pass complexity;
- validation;
- packaging.

### Deliverables

- explicit decision: inference-only or autograd support;
- tests documenting behavior;
- README limitations.

### Codex prompt

```text
Help me decide whether this custom op should support autograd or remain inference-only.
Explain the tradeoff in terms of learning value, correctness risk, and career portfolio value.
```

## Week 15: Triton comparison

### Concepts

- when higher-level kernel languages help;
- comparing CUDA C++ vs Triton;
- developer productivity vs control.

### Deliverables

- optional Triton version of one kernel;
- benchmark comparison;
- notes on readability and performance.

### Codex prompt

```text
Create a fair comparison plan for my CUDA kernel vs a Triton version.
The goal is to learn tradeoffs, not to prove one is always better.
Define benchmark sizes, correctness checks, and what conclusions are allowed.
```

## Week 16: Integration review week

### Deliverables

- one polished custom op or documented failed attempt;
- lessons learned;
- interview-style explanation.

### Codex prompt

```text
Act as a skeptical ML systems interviewer.
Question me about my custom CUDA op: API design, shape constraints, memory access, benchmark design, and limitations.
Ask one question at a time.
```

---

# Stage 5 — Robotics GPU kernels

## Week 17: Depth image to point cloud

### Concepts

- camera intrinsics;
- per-pixel parallelism;
- memory layout;
- invalid depth handling.

### Deliverables

- CPU reference;
- CUDA kernel;
- benchmark;
- visual sanity check if possible.

### Codex prompt

```text
Teach me depth-image to point-cloud conversion as a CUDA kernel.
First explain the math and parallel mapping.
Then give me a TODO-based implementation plan with tests.
```

## Week 18: Point cloud voxel filter

### Concepts

- spatial hashing;
- voxel grids;
- atomics or sorting tradeoffs;
- robotics perception preprocessing.

### Deliverables

- simple CPU voxel filter;
- CUDA implementation or feasibility writeup;
- bottleneck analysis.

### Codex prompt

```text
Plan a point-cloud voxel filter CUDA lesson.
Before coding, explain at least two algorithm choices and their tradeoffs: atomics, sorting, hashing, and memory footprint.
```

## Week 19: Collision checking or nearest-neighbor distance

### Concepts

- batched distance computations;
- obstacle representations;
- parallel scoring;
- memory layout for robotics workloads.

### Deliverables

- CPU reference;
- CUDA batch computation;
- benchmark vs number of trajectories/obstacles.

### Codex prompt

```text
Create a collision-checking lesson for batched robot trajectory candidates.
Keep the robot model simplified.
Focus on data layout, parallel scoring, and benchmark interpretation.
```

## Week 20: Trajectory rollout scoring

### Concepts

- batched simulation;
- cost functions;
- parallel reduction per trajectory;
- selecting best candidate.

### Deliverables

- simplified dynamics;
- CPU rollout;
- CUDA rollout;
- benchmark and visualization optional.

### Codex prompt

```text
Plan a GPU trajectory rollout scoring mini-project.
Use a simplified 2D robot model first.
Define data structures, CPU reference, CUDA kernel stages, correctness checks, and benchmark scenarios.
```

---

# Stage 6 — Capstone and portfolio

## Weeks 21-22: Capstone build

Choose one capstone:

1. **Inference capstone:** custom CUDA/PyTorch RMSNorm or softmax benchmark with profiler report.
2. **Robotics capstone:** GPU trajectory rollout or point-cloud preprocessing pipeline.
3. **Scientific software capstone:** GPU-accelerated image/QC/data pipeline relevant to lab automation.

### Deliverables

- clean repo path;
- reproducible build;
- benchmark CSV;
- profiler notes;
- technical report.

### Codex prompt

```text
Help me select a capstone based on my completed lessons and career goals.
Score each option for learning value, portfolio value, feasibility, and alignment with lab automation/robotics AI.
Then recommend one.
```

## Week 23: Portfolio packaging

### Deliverables

- final README;
- architecture diagram if useful;
- benchmark table;
- limitations section;
- 5-10 minute demo script.

### Codex prompt

```text
Turn this capstone into a portfolio-quality README and demo script.
Audience: ML systems, robotics software, or scientific software hiring manager.
Do not exaggerate. Include reproducibility details and limitations.
```

## Week 24: Interview prep and next-step plan

### Deliverables

- interview Q&A notes;
- resume bullet drafts;
- next 90-day plan.

### Codex prompt

```text
Act as an interviewer for a CUDA/ML systems-adjacent role.
Ask me questions about my capstone, one at a time.
After each answer, grade me on correctness, specificity, and honesty about limitations.
```

---

# Course completion criteria

You have completed the course when you have:

- at least 8 completed lessons;
- at least 3 lessons with profiler notes;
- at least 2 inference kernels;
- at least 2 robotics/scientific kernels;
- one custom framework integration attempt;
- one capstone report;
- a portfolio README with measured claims;
- a clear explanation of your technical trajectory.
