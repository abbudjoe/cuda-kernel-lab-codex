# Codex Workflow for This Course

## Operating principle

Use Codex to accelerate feedback, not to skip learning.

Bad loop:

```text
Ask Codex for kernel → paste code → run once → move on
```

Good loop:

```text
Predict → implement → test → benchmark → profile → hypothesize → optimize → explain → review → quiz
```

## Recommended session types

### 1. Concept session

Purpose: understand the kernel pattern before coding.

Prompt:

```text
Tutor me on [TOPIC]. Keep it practical. Explain the mental model, then give me one small coding task.
```

### 2. Implementation session

Purpose: write naive code yourself.

Prompt:

```text
Review my current TODOs and tell me the next function to implement. Do not fill it in yet.
```

### 3. Debug session

Purpose: fix build/test/correctness failures.

Prompt:

```text
Here is the compiler/test output. Explain the most likely cause and suggest the smallest fix. Do not rewrite unrelated code.
```

### 4. Performance session

Purpose: turn timings into understanding.

Prompt:

```text
Here are my benchmark results. Help me interpret them and propose one measured optimization experiment.
```

### 5. Review session

Purpose: make the lesson credible.

Prompt:

```text
Review this lesson against RUBRIC.md and the definition of done in AGENTS.md.
```

## When to use Codex skills

The optional `.agents/skills/` directory contains reusable workflows:

- `cuda-lesson`: use when creating or completing a lesson;
- `cuda-profiler-coach`: use when analyzing profiler output;
- `cuda-portfolio-review`: use when polishing a README/report.

Example prompt:

```text
Use the cuda-lesson skill to create a staged plan for Week 9 softmax. Do not implement yet.
```

## What to paste back to Codex

For best review, paste or point Codex to:

- exact command run;
- full error output;
- benchmark table;
- GPU model;
- input sizes;
- changed files;
- what you expected;
- what happened instead.

## What not to ask

Avoid vague prompts like:

```text
Make this faster.
```

Prefer:

```text
The optimized reduction is slower for N <= 4096 but faster for large N.
Help me form hypotheses. Consider launch overhead, occupancy, shared memory, and memory traffic.
Suggest one experiment at a time.
```
