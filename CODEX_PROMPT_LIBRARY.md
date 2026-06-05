# Codex Prompt Library

Use these prompts from the repo root. Adapt filenames and lesson numbers as needed.

## Universal starting prompt

```text
Read AGENTS.md, COURSE_SYLLABUS.md, RUBRIC.md, and the relevant lesson README.
Act as my CUDA tutor.
Do not implement the final solution yet.
First summarize the learning objective, inspect the files, and give me the smallest next task.
```

## Tutor mode

Use when beginning a new concept.

```text
Act as my tutor for [TOPIC].
Explain the mental model in under 500 words.
Then ask me 3 questions that check whether I understand the concept.
After that, give me one implementation task small enough to complete in one session.
Do not write the final implementation yet.
```

## Socratic prediction prompt

Use before benchmarking or profiling.

```text
Before I run this benchmark, ask me to predict:
1. whether the kernel is memory-bound, compute-bound, launch-bound, synchronization-bound, or occupancy-limited;
2. which input sizes might change the behavior;
3. which metric would confirm or falsify my guess.
Wait for my answer, then critique it.
```

## Scaffold prompt

Use when starting a new lesson.

```text
Create a new lesson scaffold for [LESSON_NAME] using lessons/_template.
Include:
- CPU reference placeholder;
- naive CUDA placeholder;
- correctness test plan;
- benchmark plan;
- notes file;
- prompts file.
Do not fill in the final optimized CUDA kernel.
```

## Implementation review prompt

Use after you write code.

```text
Review my implementation in [FILES].
Check for:
- correctness bugs;
- out-of-bounds access;
- race conditions;
- missing synchronization;
- memory access pattern problems;
- numerical stability;
- benchmark flaws.
Do not rewrite the whole file. Give me the smallest prioritized fix list.
```

## Test design prompt

```text
Design correctness tests for [KERNEL].
Include:
- small hand-checkable cases;
- random cases;
- edge cases;
- tolerance rules for floating-point results;
- input sizes that are not multiples of block size.
Explain why each test matters.
```

## Benchmark design prompt

```text
Design a benchmark plan for [KERNEL].
Include:
- warmup count;
- timed iterations;
- input-size sweep;
- metrics to record;
- hardware/software metadata;
- fair baseline comparisons;
- pitfalls that could make the benchmark misleading.
```

## Profiler coach prompt

Use after running Nsight Compute or Nsight Systems.

```text
Act as my profiler coach.
Here is my profiler output for [KERNEL]:

[PASTE OUTPUT]

Help me interpret it.
Classify the likely bottleneck.
Then propose exactly two optimization experiments.
For each experiment, state:
- hypothesis;
- code change;
- metric expected to improve;
- risk or tradeoff.
```

## Optimization experiment prompt

```text
Create a plan for one optimization experiment on [KERNEL].
Do not change unrelated files.
Keep the naive version intact.
Add the optimized version side by side.
Update benchmarks only after I run them.
```

## Notes update prompt

```text
Update the lesson NOTES.md based on this session.
Include:
- what I implemented;
- bug(s) encountered;
- benchmark result;
- profiler finding if any;
- bottleneck hypothesis;
- what changed after optimization;
- one rule of thumb;
- one open question.
Use concise technical language.
```

## Portfolio README prompt

```text
Turn this lesson into a portfolio-quality README section.
Audience: ML systems, GPU performance, robotics software, or scientific software hiring manager.
Include:
- problem statement;
- why it matters;
- implementation summary;
- correctness approach;
- benchmark setup;
- results;
- bottleneck analysis;
- limitations;
- what I would improve next.
Do not exaggerate. Every performance claim must include context.
```

## Interviewer prompt

```text
Act as a skeptical interviewer for an ML systems / CUDA / robotics software role.
Ask me questions about [LESSON_OR_CAPSTONE], one at a time.
After each answer, grade it on:
- correctness;
- specificity;
- clarity;
- honesty about limitations.
Do not give hints unless I ask.
```

## Stage review prompt

```text
Review my completed work for Stage [N].
Use RUBRIC.md.
Score each lesson from 0-5 across the categories.
Identify:
- strongest evidence of skill;
- weakest gap;
- one lesson to polish for portfolio;
- one concept I should revisit;
- next stage readiness.
```

## Capstone selection prompt

```text
Help me choose a capstone based on my completed lessons and career goals.
Options:
1. inference custom CUDA op;
2. robotics trajectory rollout;
3. scientific/lab automation GPU pipeline;
4. another option you propose.
Score each on feasibility, learning value, portfolio value, and career alignment.
Recommend one and explain the tradeoffs.
```

## Resume bullet prompt

```text
Turn this project into 3 resume bullets for roles in:
1. scientific software / lab automation;
2. ML systems / inference engineering;
3. robotics software.
Each bullet must be precise, measured, and not overclaim.
Use placeholders where benchmark numbers are missing.
```
