# MLLMquant Reproduction Operating Rules

Policy version: 1.0

Last reviewed: 2026-07-15

## Git Prohibition by Default

Unless the user's current task explicitly authorizes a Git or GitHub operation, **all Git and GitHub operations are forbidden**. This includes read-only commands such as `git status`, `git diff`, `git log`, `git show`, `git remote`, `git branch`, `git rev-parse`, `git ls-files`, `git config`, and `git ls-remote`, as well as `git add`, `git commit`, `git push`, `git pull`, `git fetch`, `git merge`, `git rebase`, `git reset`, `git restore`, `git clean`, `git checkout`, `git switch`, `git tag`, and `git stash`.

Authorization is granular: a task must explicitly name the Git operation that is allowed. Requests to check the project, read code, edit a file, run an experiment, or save results are not Git authorization. Do not change global Git configuration, proxy configuration, credentials, remotes, branches, or repository history unless the current task explicitly permits that exact change.

## Scope and Authority

These rules apply to work rooted at `/home/zhouyangchengyu/project_origin`. The current user task has priority over this file when it explicitly grants or restricts an operation. Use Codex only with project-scoped context; do not assume project facts from prior conversations without current evidence.

## Repository Identity

The independent project root is `/home/zhouyangchengyu/project_origin`. Its configured repository URL is `https://github.com/2006lqh/MLLMquant_reproduction.git`.

`/home/zhouyangchengyu/project_origin/EfficientAI` is ordinary source content within this project, not a nested Git repository. Do not add an Alibaba EfficientAI remote or recreate nested Git metadata unless the current user task explicitly authorizes it.

## Required Context Reading

Before modifying MASQuant-related code, read the applicable current-task instructions and the relevant local documentation:

- `EfficientAI/README.md`
- `EfficientAI/masquant/README.md`
- `EfficientAI/masquant/INSTALL.md`
- `EfficientAI/masquant/examples/qwen2_5_omni/README.md`
- the specific entry script and source module being changed

Read `submissions/result.md` or `submissions/experience.md` only when the user explicitly requests a result or experience update. Do not create automatic state, progress, or handoff files.

## Verified Project Paths

The following paths exist at the project root:

- `EfficientAI/`
- `EfficientAI/masquant/`
- `submissions/`
- `submissions/experiments/`
- `submissions/logs/`
- `cache/`

Do not invent project paths. For any required path not confirmed by the current task or local filesystem inspection, mark it `NOT VERIFIED` and omit it from commands until verified.

## Artifact Placement

Place reproducible experiment artifacts under:

- `submissions/experiments/`
- `submissions/logs/` for process logs
- `cache/` and its subdirectories for regenerable caches

Do not move, delete, overwrite, or silently rename completed artifacts. Update `submissions/result.md` and `submissions/experience.md` only when the user explicitly asks for that documentation update.

Do not create archived input manifests or `comparison`/`diagnostic` artifacts by default. Retain only response, scored, and summary artifacts unless the user explicitly requests one of those additional artifact types.

## File Naming

Use this artifact naming form:

`<model>__<method>__<dataset>__<subset>__<prompt>__<artifact>.<ext>`

Use lowercase fields separated by double underscores. Use hyphens only inside a field. Represent a decimal rank with `p`, for example `rank0p2`. Do not use vague names such as `output`, `final`, `new`, `old`, `test`, or `retry`. Never overwrite a nonempty artifact; a retry name must describe the run or time.

## Run and Process Safety

This is a shared server. Do not interrupt, pause, renice, preempt, or otherwise affect another user's process. Before starting a GPU or long-running task, perform a current resource check and choose a device based on evidence, not a stale observation.

Claim that a run is active only when current evidence includes its PID, assigned GPU, log location, and output path. Treat an incomplete artifact as stale only after confirming that no process still writes to it. Do not overwrite a completed result.

## Source Modification

Keep the active local implementation when preserving a divergence from upstream code. Put a concise adjacent comment around a replaced upstream block that identifies its purpose and the behavior difference. Do not retain a whole unmodified duplicate implementation, and do not create `.old`, `.orig`, or `.bak` files.

After a Python change, run `python -m py_compile` on the changed Python file. After a shell change, run `bash -n` on the changed shell file. Run broader checks only when their inputs and resource effects are understood.

## Evidence and Truthfulness

Every reported project fact must be supported by one of: the current user task, local filesystem inspection, local source or documentation, result files, logs, caches, or command output. Label uncertain information `NOT VERIFIED`. Do not fabricate run completion, WER, model behavior, GPU use, benchmark coverage, or remote repository state.

## Documentation Maintenance

When explicitly asked to update results or experience documentation, record the command, model and method identifiers, dataset and subset, input and output paths, metric definition, quantitative result, and limitations supported by evidence. Keep observations separate from hypotheses and failed diagnostics.

## Task Completion

Before reporting completion, verify the requested files and commands, report any failed checks directly, and state remaining blockers. Do not perform extra cleanup, downloads, repository operations, or experiments merely because they appear useful.
