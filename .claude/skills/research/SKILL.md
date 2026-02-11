# NLP Research Workflow

Unified command for managing the NLP research lifecycle.

## Usage

```
/research <command> [topic]
```

## Commands

### hypothesis [topic]
Define a research hypothesis with literature review support.

**Steps:**
1. Read the conventions from `.claude/templates/shared/nlp-research-conventions.md`
2. Create a hypothesis document from `.claude/templates/hypothesis.template.md`
   - Save to: `docs/01-hypothesis/features/{topic}-hypothesis.md`
   - Replace `{{TOPIC}}` with the provided topic
   - Replace `{{DATE}}` with today's date
3. Delegate to the `literature-reviewer` agent to:
   - Search for relevant papers on the topic
   - Populate the Literature Review section
   - Identify research gaps and suggest hypotheses
4. Update `docs/.research-status.json`:
   - Set `current_phase` to `"hypothesis"`
   - Set `phases.hypothesis.status` to `"in_progress"`
   - Add the document path to `phases.hypothesis.documents`
   - Set `updated_at` timestamp

### design [topic]
Design an experiment with config generation.

**Steps:**
1. Read the hypothesis document for the topic from `docs/01-hypothesis/features/`
2. Generate a new experiment ID: `EXP-YYYYMMDD-NNN`
   - Check existing experiments in `experiments/configs/` for today's sequence number
3. Delegate to the `experiment-designer` agent to:
   - Create design doc from `.claude/templates/experiment-design.template.md`
   - Save to: `docs/02-experiment-design/features/{EXP-ID}-design.md`
   - Generate config YAML at: `experiments/configs/{EXP-ID}-config.yaml`
   - Check GPU availability and feasibility
4. Update `docs/.research-status.json`:
   - Set `current_phase` to `"experiment_design"`
   - Set `phases.experiment_design.status` to `"in_progress"`
   - Add experiment entry to `experiments` array
   - Add document path to `phases.experiment_design.documents`

### run [topic]
Guide experiment execution and create environment snapshot.

**Steps:**
1. Find the latest design document and config for the topic
2. Create an experiment log from `.claude/templates/experiment-log.template.md`
   - Save to: `docs/03-experiment-log/{EXP-ID}-log.md`
3. Capture environment snapshot:
   - Run `nvidia-smi` for GPU info
   - Run `python --version`, `pip list` for software versions
   - Run `git log -1` for git state (if in git repo)
   - Record hostname, disk space
4. Populate the Environment Snapshot and Config Snapshot sections
5. Provide execution instructions:
   - Training command to run
   - Monitoring tips (GPU utilization, loss curves)
   - Checkpoint saving guidance
6. Update `docs/.research-status.json`:
   - Set `current_phase` to `"experiment_run"`
   - Set `phases.experiment_run.status` to `"in_progress"`
   - Update experiment entry status

### analyze [topic]
Analyze results with statistical verification.

**Steps:**
1. Find the experiment log and results for the topic
2. Delegate to the `result-analyzer` agent to:
   - Create analysis doc from `.claude/templates/results-analysis.template.md`
   - Save to: `docs/04-results/features/{EXP-ID}-analysis.md`
   - Generate comparison tables
   - Run statistical tests
   - Create visualizations
   - Verify hypotheses
3. Update `docs/.research-status.json`:
   - Set `current_phase` to `"result_analysis"`
   - Set `phases.result_analysis.status` to `"in_progress"`
   - Add document path

### paper [topic]
Write paper sections based on all prior research documents.

**Steps:**
1. Read all research documents for the topic:
   - Hypothesis, design, logs, analysis documents
2. Delegate to the `paper-writer` agent to:
   - Create paper draft from `.claude/templates/paper-draft.template.md`
   - Write individual sections to `docs/05-paper/sections/`
   - Include LaTeX conversion hints
3. Update `docs/.research-status.json`:
   - Set `current_phase` to `"paper_writing"`
   - Set `phases.paper_writing.status` to `"in_progress"`
   - Add section paths

### iterate [topic]
Plan the next experiment iteration based on analysis results.

**Steps:**
1. Read the most recent analysis document for the topic
2. Identify:
   - What worked well and should be kept
   - What didn't work and needs changing
   - New ideas suggested by the results
3. Create a new hypothesis document for the next iteration:
   - Reference previous experiment results
   - Refine the hypothesis based on findings
   - Update success criteria
4. Update `docs/.research-status.json`:
   - Increment `iteration_count`
   - Reset phases for new cycle
   - Set `current_phase` to `"hypothesis"`

### status
Display current research status.

**Steps:**
1. Read `docs/.research-status.json`
2. Display a formatted summary:
   ```
   === NLP Research Status ===
   Project: {name}
   Current Phase: {phase}
   Iteration: {count}

   Phases:
     [x] Hypothesis     - {status} ({n} documents)
     [ ] Exp. Design    - {status} ({n} documents)
     [ ] Exp. Run       - {status} ({n} experiments)
     [ ] Result Analysis - {status} ({n} documents)
     [ ] Paper Writing  - {status} ({n} sections)

   Experiments:
     {EXP-ID}: {status} - {description}
   ```

### next
Suggest the next action based on current state.

**Steps:**
1. Read `docs/.research-status.json`
2. Determine the current phase and its status
3. Recommend the next action:
   - If hypothesis not started → suggest `/research hypothesis [topic]`
   - If hypothesis done, design not started → suggest `/research design [topic]`
   - If design done, run not started → suggest `/research run [topic]`
   - If run done, analysis not started → suggest `/research analyze [topic]`
   - If analysis done → suggest `/research paper [topic]` or `/research iterate [topic]`
4. Provide context-specific guidance for the recommended action

## Argument Handling

- `$ARGUMENTS` contains the full argument string
- First word is the command (hypothesis, design, run, analyze, paper, iterate, status, next)
- Remaining words are the topic
- If no command is given, default to `status`
- If command requires a topic but none given, ask the user

## State File

All state tracking uses `docs/.research-status.json`. The skill reads and updates this file to maintain research workflow continuity across sessions.

## Templates Directory

All templates are in `.claude/templates/`. Conventions are in `.claude/templates/shared/nlp-research-conventions.md`.
