Detailed implementation outline for SMF and CASM
Phase 0: Stabilize the shared continual-learning foundation
This phase should happen first because both SMF and CASM depend on the same training loop, checkpointing, registry, and evaluation semantics.
0.1 Audit the current training architecture
Objective: make sure the existing system boundaries are real and explicit before adding new behavior.
Tasks
Trace the execution path from the runner entrypoint to the per-period training step.
Confirm the following call chain is stable and documented:
run_training() controls the overall continual-learning loop.
trainer.train_period() performs the actual learning for one period.
trainer.checkpoint() persists all state after each period.
train_runner.py validates checkpoint compatibility before resuming.
Identify where the current model is built, where the tokenizer is built, where the registry is instantiated, and where the contradiction detector is wired in.
Record which parts are method-agnostic and which parts branch on TrainConfig.method.
What the agent should produce
A short architecture note in the repo or an internal markdown doc.
A list of exact files and functions that will be modified later.
Acceptance criteria
The project’s continual-learning control flow can be described in one paragraph without guessing.
Every period-level operation has one owner.
Checkpoint and resume behavior are not spread across multiple hidden code paths.

0.2 Define the method contract
Objective: lock the meaning of each training mode before implementation begins.
Tasks
Define the intended behavior of each method in one place:
full_ft: all trainable parameters updated.
lora: low-rank adaptation only.
smf: frozen backbone plus one sparse shared memory module.
casm: versioned memory bank with contradiction-aware routing and branching.
Make sure each method has a distinct training and evaluation identity.
Ensure TrainConfig.method is the authoritative switch, not ad hoc flags scattered elsewhere.
Ensure the method-specific config rules are consistent with the rest of the repo.
What the agent should produce
A method matrix showing:
trainable components
checkpointed state
routing behavior
versioning behavior
expected metrics
Acceptance criteria
Each method has a clear semantic contract.
No method is implicitly overloaded to mean something else.

0.3 Define the evaluation contract
Objective: make sure training results are measured consistently across all methods.
Tasks
Confirm the repo’s current evaluation metrics and their meaning:
plasticity
stability
token_f1
routing_acc
Confirm when evaluation happens:
after each period, when eval_after_each_period is enabled
Define which metrics are mandatory for each method:
full_ft: standard task quality plus retention baseline
lora: parameter-efficient baseline
smf: retention-focused sparse memory baseline
casm: version-aware memory routing baseline
Define any extra metrics required by SMF and CASM.
Make metric names stable and structured so downstream analysis scripts do not need special casing.
What the agent should produce
A metric specification block or markdown table.
A clear rule for which metrics are reported per method.
Acceptance criteria
Every method can be compared with the same evaluation machinery.
The evaluation outputs are machine-readable and stable.

Phase 1: TrainConfig and method-specific validation
This phase makes SMF and CASM first-class configuration modes rather than implicit variants.
1.1 Extend TrainConfig for SMF
Objective: give SMF enough configuration to be useful without polluting other methods.
Tasks
Add SMF-specific fields to TrainConfig, for example:
smf_memory_size
smf_sparsity_ratio
smf_update_layers
smf_regularization_weight
smf_freeze_backbone = True
Ensure defaults are conservative and safe.
Make sure these settings appear in serialization, checkpoint metadata, and config printing.
Keep the existing method validator intact, but add method-specific validation logic for smf.
Validation rules
smf_memory_size > 0
0 < smf_sparsity_ratio <= 1
smf_update_layers is non-empty and refers to valid layer names/indices
smf_regularization_weight >= 0
smf_freeze_backbone must be true for SMF, or else the config should fail fast
Acceptance criteria
Invalid SMF settings fail before training begins.
The config object fully describes an SMF experiment.

1.2 Extend TrainConfig for CASM
Objective: define the knobs required for versioned memory and routing.
Tasks
Add CASM-specific fields such as:
casm_num_slots
casm_router_hidden_size
casm_top_k
casm_router_temperature
casm_sparsity_weight
casm_overlap_weight
casm_branch_on_contradiction = True
Add validation rules for all CASM settings.
Ensure CASM config implies the correct checkpoint contents and evaluation behavior.
Make sure the config serializes cleanly and can be restored exactly.
Validation rules
casm_num_slots >= 1
casm_top_k >= 1
casm_top_k <= casm_num_slots
Router hidden size must be positive
Loss weights must be non-negative
Acceptance criteria
CASM cannot be launched with nonsense slot or router settings.
Config round-trips through checkpoints without losing fields.

Phase 2: SMF implementation
SMF is the first method that should be fully implemented because it establishes the sparse-memory pattern without version routing complexity.
2.1 Create the SMF memory module
Objective: build the actual sparse memory layer used by SMF.
Tasks
Implement a wrapper module around the base model.
Freeze the backbone parameters.
Add one trainable sparse memory block.
Attach the memory block to the designated update layers from config.
Introduce a sparse gate or learned mask that controls which memory dimensions are active.
Keep one shared memory state across all periods.
Make sure the memory participates in the forward pass and gradients flow into it.
Implementation details
The sparse memory should be as small and local as possible.
The backbone should never be modified in SMF mode.
The memory contribution should be additive or residual, not a replacement for the base model.
The gating mechanism should be deterministic enough to debug, but flexible enough to learn useful updates.
What to avoid
No slot IDs
No version branches
No per-probe routing
No hidden overwriting of the backbone
Acceptance criteria
The wrapper can be instantiated for an arbitrary supported model.
A forward pass includes sparse memory contribution.
Only SMF parameters receive gradients.

2.2 Modify optimizer construction for SMF
Objective: make sure the trainer only updates the intended parameters.
Tasks
In CASFTrainer.__init__, branch optimizer parameter selection on method.
For SMF, build the optimizer from trainable memory parameters only.
Exclude frozen backbone parameters entirely.
Ensure parameter groups remain compatible with weight decay, learning rate scheduling, and mixed precision.
Keep optimizer behavior stable across resume.
Implementation details
The trainer currently uses something like AdamW(self.model.parameters(), ...); this must be replaced by a filtered parameter list for SMF.
The filter should be explicit and testable.
If no trainable parameters are found, fail early with a clear error.
Acceptance criteria
Optimizer state only includes SMF parameters.
Frozen parameters do not accumulate updates or optimizer state.

2.3 Adapt the forward / training step for SMF
Objective: ensure training behaves correctly with the sparse memory wrapper.
Tasks
Keep the trainer’s general step structure intact.
Make sure the model forward pass includes the sparse memory contribution.
Ensure outputs.loss still drives backprop the usual way.
Ensure no routing signals are introduced in SMF.
Ensure regularization on the sparse memory is applied if configured.
Recommended behavior
Primary task loss comes from the model output.
Optional sparsity penalty encourages compact updates.
Optional memory regularizer prevents degenerate gating behavior.
Acceptance criteria
SMF trains with the same period loop as the other methods.
The loss decreases and updates only the sparse memory.

2.4 Add SMF metrics and reporting
Objective: measure whether SMF actually behaves like a retention-preserving sparse baseline.
Tasks
Add metrics for:
retention on earlier periods
current-period performance
forgetting after each new period
average number of active sparse parameters
Integrate these metrics into the existing evaluation output.
Make sure the metric keys are stable and easy to compare across runs.
Log them per period, not only at the end of training.
Recommended reporting format
smf/retention
smf/plasticity
smf/forgetting
smf/active_params
smf/current_period_score
Acceptance criteria
SMF results can be plotted period by period.
The metrics help distinguish “sparse but useful” from “sparse and broken.”

2.5 Validate SMF in the full temporal loop
Objective: confirm SMF works in the same continual-learning flow as the existing baselines.
Tasks
Run the normal temporal plan unchanged:
aug_sep
sep_oct
oct_nov
nov_dec
Verify the same checkpoint/resume logic works.
Compare SMF against:
full_ft
lora
Confirm SMF improves retention relative to full fine-tuning.
Confirm that the runner and trainer do not need a special SMF-only script.
Acceptance criteria
SMF is fully usable as a TrainConfig.method option.
It behaves like a baseline method, not a one-off experiment.

Phase 3: CASM implementation
CASM should build on the SMF idea but add explicit versioning, contradiction awareness, and routing.
3.1 Replace one sparse memory with a slot bank
Objective: move from a single shared sparse memory to a versioned memory structure.
Tasks
Introduce a memory slot abstraction.
Each slot should contain:
trainable parameters
subject metadata
relation metadata
valid_from
valid_until
usage counts
parent link
contradiction link
Implement slot creation and closure semantics.
Make sure slots are never silently overwritten.
Ensure old slots remain queryable even after newer versions are added.
Important design rule
CASM is a version chain, not a rewrite buffer.
Acceptance criteria
Multiple versions of the same fact can coexist.
The registry remembers historical fact states.
Closed slots remain in storage.

3.2 Add the router
Objective: select the correct slot(s) for a query and time context.
Tasks
Implement a router module that accepts:
query representation
optional temporal signal
Output:
top-1 or top-k slot IDs
routing weights
Train the router jointly or semi-jointly with memory updates.
Ensure routing output is stable enough to support routing_acc.
Implementation details
Router inputs should be compact; do not feed raw long context when a summary representation will do.
Start with a simple version before adding complexity.
Keep top-k behavior configurable.
Ensure router decisions are logged for later debugging.
Acceptance criteria
The router returns a valid subset of slots.
Routing decisions can be evaluated independently.

3.3 Make contradiction detection part of the training protocol
Objective: use contradiction detection to decide when to branch memory.
Tasks
Ensure each period processes changed probes through detector.check(changed, registry).
Make the detector output actionable training signals, not just logging.
Use detector results to decide:
whether to reuse a slot
whether to branch a new slot
whether to mark an old slot as closed
Feed contradictory examples into the update path in a structured way.
Protocol
Load the current period’s passages.
Identify changed probes.
Run contradiction detection against the registry.
For each contradiction:
preserve the old version
create a new versioned slot
record linkage between old and new
Continue training with both passages and conflict-aware examples.
Acceptance criteria
Contradictions lead to branching behavior.
The detector influences the update path, not just post-hoc metrics.

3.4 Modify train_period() for CASM
Objective: make the per-period trainer branch correctly for versioned memory.
Tasks
Add a CASM-specific branch in CASFTrainer.train_period().
In that branch:
load passages for the period
collect changed probes
run contradiction detection
route examples to the correct slot(s)
create new slots when contradictions are detected
train selected slots and the router
write the updated facts back into the registry
Preserve the existing period boundary and checkpoint cadence.
Implementation details
Keep the general dataset loading and batching shape intact if possible.
Do not add hidden side effects outside the trainer.
Keep registry updates deterministic and well ordered.
Acceptance criteria
CASM can train one period at a time without breaking the loop.
The update path is clearly different from SMF and full fine-tuning.

3.5 Add CASM losses
Objective: optimize for correctness, sparsity, and non-overlap.
Tasks
Compose a CASM training loss from:
task loss
router loss
sparsity penalty
anti-overlap penalty
Ensure each term is individually configurable.
Ensure none of the terms dominate unexpectedly at default settings.
Log each loss component separately.
Intended effect
Task loss keeps the model useful.
Router loss teaches slot selection.
Sparsity penalty keeps slot usage focused.
Anti-overlap penalty prevents different slots from collapsing to the same behavior.
Acceptance criteria
Loss components are visible in logs.
Each term can be tuned independently.
Slot collapse is reduced relative to a naive multi-slot design.

3.6 Persist versioning in checkpoints
Objective: make checkpointing preserve the full memory history.
Tasks
Extend checkpoint saving so it includes:
model weights
tokenizer
config
registry state
slot metadata
router state
contradiction links
current period marker
Extend loading so all of the above restore cleanly.
Keep checkpoint compatibility validation strict.
Make sure partial or stale checkpoints are rejected clearly.
Important
CASM cannot resume correctly if only backbone weights are saved.
The registry must be treated as first-class state.
Acceptance criteria
A checkpoint round-trip preserves version history.
Resume after any period boundary works.
Slot IDs and links remain stable across save/load.

3.7 Keep the runner simple
Objective: avoid forking orchestration logic unnecessarily.
Tasks
Leave the main period-by-period loop unchanged.
Add a CASM model factory path alongside existing real/synthetic model factories.
Route cfg.method == "casm" to the CASM wrapper.
Keep checkpoint validation and resume behavior in the runner.
Avoid special-case branches outside the model factory and trainer where possible.
Acceptance criteria
The runner still looks like one runner.
No CASM-specific scripts are required.

3.8 Add CASM-specific metrics
Objective: measure version-aware behavior, not just generic accuracy.
Tasks
Report:
plasticity on changed probes
stability on unchanged probes
contradiction accuracy
routing accuracy
Make these metrics available after each period.
Tie them directly to the CASM registry and router behavior.
Ensure evaluation uses the correct temporal context.
Recommended metric keys
casm/plasticity
casm/stability
casm/contradiction_acc
casm/routing_acc
Acceptance criteria
CASM can be judged on whether it preserved historical facts correctly.
Routing quality is measurable, not just inferred.

Phase 4: File-level implementation structure for coding agents
This section is useful if you want to assign the work across multiple agents or PRs.
Agent A: Config and validation
Owns
TrainConfig
method validation
serialization/deserialization of method-specific settings
Tasks
Add SMF and CASM fields.
Add validation logic.
Ensure config persistence and checkpoint compatibility.
Done when
full_ft, lora, smf, and casm all validate correctly.
Invalid settings fail fast.

Agent B: SMF model and optimizer
Owns
SMF wrapper module
parameter freezing
sparse memory gating
optimizer filtering
Tasks
Implement memory block.
Integrate it into forward pass.
Filter trainable parameters for optimizer.
Add SMF regularization hooks.
Done when
Only sparse memory updates.
Backbone stays frozen.
Training works across periods.

Agent C: CASM memory bank and router
Owns
slot bank
slot metadata
router logic
slot selection
routing loss
Tasks
Build slot abstraction.
Build router.
Train and evaluate routing.
Support top-k routing.
Done when
Queries map to versioned slots correctly.
Router metrics are observable.

Agent D: Trainer and period orchestration
Owns
CASFTrainer.train_period()
training step branching
data flow for contradictions and registry updates
Tasks
Add SMF and CASM branches.
Wire contradiction detection into update flow.
Keep the training loop stable.
Done when
Each method behaves correctly inside the same loop.
No hidden logic bypasses the trainer.

Agent E: Checkpointing and resume
Owns
checkpoint save/load format
compatibility checks
registry persistence
router persistence
Tasks
Save all CASM state.
Save SMF memory state.
Restore cleanly on resume.
Validate checkpoint compatibility.
Done when
Resume works after every period.
No state is lost between runs.

Agent F: Evaluation and metrics
Owns
per-period evaluation
SMF metrics
CASM metrics
metric naming and logging
Tasks
Add method-specific metric groups.
Ensure per-period reports are consistent.
Preserve comparability with existing baselines.
Done when
Results can be compared across all methods without special processing.

Agent G: Tests and regression coverage
Owns
unit tests
integration tests
checkpoint resume tests
routing and contradiction tests
Tasks
Test config validation.
Test SMF freezing and optimizer filtering.
Test CASM slot branching.
Test registry persistence.
Test period-to-period resume.
Done when
The main failure modes are covered.

Phase 5: Recommended build order
This is the most practical sequencing for coding agents.
Step 1: Config and shared plumbing
Add method-specific config fields and validation.
Confirm evaluation and checkpoint contracts.
Write tests for invalid configs.
Step 2: SMF
Implement frozen-backbone sparse memory.
Filter optimizer parameters.
Wire metrics.
Run a full temporal experiment.
Step 3: CASM memory structure
Add slots, metadata, and registry links.
Persist state in checkpoints.
Step 4: CASM router
Add slot selection and routing metrics.
Add top-k support if needed.
Step 5: Contradiction-aware training
Wire detector into period training.
Branch memory on contradictions.
Update registry after training.
Step 6: Evaluation polish
Ensure metrics are correctly labeled.
Confirm historical facts remain recoverable.
Compare methods using the same script.
Step 7: Regression hardening
Add tests for resume, slot branching, and metric stability.

Phase 6: What “done” should mean in code
SMF is done when:
the backbone is frozen
a sparse trainable memory exists
the optimizer only updates that memory
no routing or versioning is involved
retention improves over full fine-tuning
CASM is done when:
multiple versioned slots exist
contradictions trigger branching
a router selects the correct slot(s)
historical facts are not overwritten
routing and contradiction metrics are reported correctly
checkpoint/resume preserves the full memory history

Phase 7: Practical implementation notes for agents
A few rules should be repeated to every coding agent:
Do not fork the runner unless absolutely necessary.
 The existing period loop should stay the same.
Keep method behavior in the model/trainer/config layers.
 Orchestration should remain thin.
Persist everything needed to resume exactly.
 CASM especially must save registry and router state.
Prefer explicit branching over implicit side effects.
 A method should behave differently because the code says so clearly.
Write tests as you go.
 Especially for optimizer filtering, slot branching, and checkpoint recovery.

































1) config.py or wherever TrainConfig lives
Goal: make smf and casm first-class, validated methods.
Changes to make
Add SMF fields:
smf_memory_size: int
smf_sparsity_ratio: float
smf_update_layers: list[int] | list[str]
smf_regularization_weight: float
smf_freeze_backbone: bool = True
Add CASM fields:
casm_num_slots: int
casm_router_hidden_size: int
casm_top_k: int
casm_router_temperature: float
casm_sparsity_weight: float
casm_overlap_weight: float
casm_branch_on_contradiction: bool = True
Extend any existing method enum / validator to accept smf and casm.
Add method-specific validation:
SMF: memory size > 0, sparsity in (0, 1], freeze backbone must be true.
CASM: slot count >= 1, top-k in valid range, router hidden size > 0, loss weights >= 0.
Ensure config serialization includes all new fields and checkpoint metadata preserves them.
Expected signatures / hooks
TrainConfig.validate(self) -> None
TrainConfig.to_dict(self) -> dict
TrainConfig.from_dict(cls, data: dict) -> TrainConfig
Acceptance check
Invalid SMF/CASM configs fail before training starts.
A saved checkpoint can restore the exact config, including method-specific fields.

2) models.py or the base model factory file
Goal: add wrappers for SMF and CASM without changing the base model implementation unnecessarily.
Changes to make
Create an SMFModelWrapper or equivalent module.
Create a CASMModelWrapper or equivalent module.
Keep the original backbone model untouched as much as possible.
Ensure both wrappers expose a compatible forward interface with the trainer.
SMF wrapper responsibilities
Freeze backbone parameters.
Attach a sparse memory module to selected layers.
Add the memory contribution in the forward pass.
Expose only the memory parameters as trainable.
CASM wrapper responsibilities
Extend the SMF idea into a slot bank.
Expose slot lookup / routing logic.
Support creation of a new slot when the detector indicates contradiction.
Return router-related outputs if the trainer needs them for loss computation.
Expected signatures / hooks
build_model_and_tokenizer(cfg) -> tuple[model, tokenizer]
SMFModelWrapper.forward(...)
CASMModelWrapper.forward(...)
model.trainable_parameters() or equivalent helper if the codebase already uses one
Acceptance check
The trainer can call the wrappers without special-case logic in the training loop.
SMF has one shared sparse memory.
CASM has multiple slots plus routing.

3) memory.py, registry.py, or the existing MemoryRegistry module
Goal: make versioned memory a durable part of CASM.
Changes to make
Define a slot structure if not already present.
Store:
slot id
trainable parameters
subject
relation
valid_from
valid_until
usage count
parent slot id
contradiction link(s)
Add methods for:
creating a new slot
closing a slot
retrieving candidate slots
writing updated facts back to the registry
serializing and deserializing the registry
Expected signatures / hooks
MemoryRegistry.add_slot(...)
MemoryRegistry.close_slot(slot_id, valid_until=...)
MemoryRegistry.lookup(subject, relation, time=None)
MemoryRegistry.to_json()
MemoryRegistry.from_json(...)
MemoryRegistry.update_from_probes(...)
Acceptance check
Older facts remain stored after newer versions are added.
Closed slots are not deleted.
Registry round-trips through checkpoint save/load.

4) detector.py or ContradictionDetector
Goal: convert contradiction detection from a passive utility into an active training signal.
Changes to make
Ensure the detector accepts changed probes and the registry.
Return structured outputs that the trainer can act on.
Distinguish between:
no contradiction
contradiction requiring slot reuse
contradiction requiring new slot creation
Expected signatures / hooks
ContradictionDetector.check(probes, registry) -> list[ContradictionResult]
ContradictionResult should contain enough information to route update logic.
Acceptance check
Trainer code can use detector output to branch memory updates.
The detector result is deterministic and easy to test.

5) trainer.py or CASFTrainer
Goal: make SMF and CASM behave correctly in training without changing the runner shape.
Changes to make
Add method-specific optimizer parameter selection.
Add method-specific training step behavior.
Add CASM branching logic in train_period().
Keep the same outer period loop behavior.
SMF branch
Build optimizer from SMF trainable parameters only.
Freeze all backbone parameters.
Use normal task loss plus optional SMF regularization.
Do not route or branch slots.
CASM branch
Load current period passages.
Collect changed probes.
Call self.detector.check(probes, self.registry).
Route each training example to a slot.
Create a new slot when contradiction is detected.
Train selected slots and router.
Write updated facts back into the registry.
Expected signatures / hooks
CASFTrainer.__init__(...)
CASFTrainer.train_period(period_data, ...)
CASFTrainer._train_step(batch, ...)
CASFTrainer._build_optimizer(...)
CASFTrainer._select_trainable_parameters(...)
Acceptance check
SMF optimizer updates only sparse memory.
CASM optimizer updates router + selected slots.
train_period() still fits the same runner contract.

6) checkpoint.py or checkpoint helpers inside trainer.py
Goal: persist all state needed to resume SMF and CASM faithfully.
Changes to make
Continue saving:
model weights
tokenizer
config
period marker
Add saving for:
memory_registry.json
slot metadata
router state
contradiction links
any SMF sparse-memory state if it is not already inside model weights
Add matching load logic.
Extend compatibility validation to reject mismatched checkpoint/method combinations.
Expected signatures / hooks
trainer.checkpoint(...)
load_checkpoint(...)
validate_checkpoint_compatibility(...)
Acceptance check
Resume works after each period.
CASM can restore version history exactly.
Loading a wrong-method checkpoint fails clearly.

7) train_runner.py
Goal: keep orchestration thin and method-agnostic.
Changes to make
Keep the existing period-by-period loop.
Keep checkpoint-after-each-period behavior.
Add CASM model factory routing if needed.
Make sure cfg.method == "smf" and cfg.method == "casm" are routed to the correct wrappers.
Do not add separate scripts for SMF/CASM unless absolutely necessary.
Expected signatures / hooks
run_training(...)
build_model_and_tokenizer(...)
resume_from_checkpoint(...)
Acceptance check
The runner can execute full_ft, lora, smf, and casm with the same control flow.
The runner remains the only orchestration entrypoint.

8) training_plan.py
Goal: keep the temporal schedule explicit and reusable.
Changes to make
Preserve the current period sequence:
aug_sep
sep_oct
oct_nov
nov_dec
Ensure every method consumes the same plan.
Make the plan object easy for the trainer to inspect period boundaries, changed probes, and passage groups.
Expected signatures / hooks
get_temporal_training_plan(...)
iter_periods(...)
PeriodSpec or equivalent structure
Acceptance check
All methods use the same period schedule.
The plan does not branch by method unless strictly necessary.

9) evaluation.py or metrics modules
Goal: report the right metrics for each method.
Changes to make
Keep the shared metrics:
plasticity
stability
token_f1
routing_acc
Add SMF metrics:
retention on earlier periods
forgetting after each new period
average active sparse parameters
Add CASM metrics:
contradiction accuracy
routing accuracy
stability on unchanged probes
plasticity on changed probes
Ensure metrics are logged per period.
Expected signatures / hooks
evaluate_after_period(...)
evaluate_versioned(...)
compute_retention(...)
compute_forgetting(...)
compute_routing_acc(...)
Acceptance check
Metrics are comparable across methods.
CASM-specific scores are measured against the registry and router behavior, not just generic accuracy.

10) losses.py or loss helpers
Goal: keep CASM loss composition clean and testable.
Changes to make
Add a CASM loss combiner:
task loss
router loss
sparsity penalty
anti-overlap penalty
Add optional SMF regularization if the codebase prefers loss helpers over inline logic.
Expected signatures / hooks
compute_casm_loss(task_loss, router_loss, sparsity_loss, overlap_loss, cfg)
compute_smf_regularization(memory_state, cfg)
Acceptance check
Each loss term is independently visible in logs.
Weighting is controlled by config.

11) tests/
Goal: protect the new behavior with targeted tests before full runs.
Test files to add or expand
test_train_config.py
valid SMF config passes
invalid SMF config fails
valid CASM config passes
invalid CASM config fails
test_smf_model.py
backbone is frozen
only memory parameters require gradients
forward pass includes memory contribution
optimizer excludes frozen params
test_casm_registry.py
slot creation works
slot closure works
old versions remain accessible
registry round-trips through serialization
test_casm_router.py
router returns valid slot ids
top-k behavior works
routing outputs have expected shape
test_trainer_period_flow.py
train_period() still runs once per period
SMF uses sparse memory only
CASM calls contradiction detection and registry writes
test_checkpoint_resume.py
resume after one period works
checkpoint compatibility checks reject mismatches
CASM resume preserves registry and router
Acceptance check
The most likely regressions are covered before long training runs begin.

12) Suggested PR split
If you want clean review boundaries, split implementation into these pull requests:
PR 1: Config + validation
Files:
config.py
tests/test_train_config.py
PR 2: SMF model + trainer hookup
Files:
models.py
trainer.py
tests/test_smf_model.py
PR 3: CASM memory registry + router
Files:
memory.py
registry.py
router.py
tests/test_casm_registry.py
tests/test_casm_router.py
PR 4: CASM trainer branch + contradiction flow
Files:
trainer.py
detector.py
losses.py
tests/test_trainer_period_flow.py
PR 5: Checkpointing + resume
Files:
checkpoint.py
train_runner.py
tests/test_checkpoint_resume.py
PR 6: Evaluation and metrics
Files:
evaluation.py
metric helpers
any reporting scripts

13) Acceptance checklist for the whole system
The implementation is complete when all of these are true:
full_ft, lora, smf, and casm all run through the same runner.
SMF uses one shared sparse memory and no routing.
CASM uses slots, contradiction-aware branching, and routing.
The registry preserves version history.
Checkpoints resume exactly after each period.
The temporal plan stays aug_sep → sep_oct → oct_nov → nov_dec.
Evaluation reports the method-specific metrics clearly.
Older facts are not overwritten in CASM.
