# Prompts: Reduction

See `CODEX_PROMPT_LIBRARY.md` and `COURSE_SYLLABUS.md` for stage-specific prompts.

## Starter prompt

```text
Act as my CUDA tutor for Lesson 02 reduction.
Review reduction.cu and README.md.
Do not fill in the TODOs yet.
Explain the CPU reference TODO, then ask me to implement only that function.
```

## Code review prompt

```text
Review my Lesson 02 reduction TODO implementation.
Check for out-of-bounds access, data races, missing synchronization, incorrect timing claims, and numerical tolerance issues.
Do not optimize yet.
```
