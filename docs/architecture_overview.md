The codebase already has the right skeleton for both methods: a TrainConfig that already reserves full_ft, lora, and smf as valid methods, a CASFTrainer that already owns a MemoryRegistry and ContradictionDetector, a runner that iterates period by period and checkpoints after each unit, and a temporal training plan for temporal_wiki with aug_sep, sep_oct, oct_nov, and nov_dec. The API docs also define the exact continual-learning loop: detect contradictions on changed probes, train on passages plus conflicts, then write the updated probes back into the registry.
1) Shared foundation for both SMF and CASM
First, make the continual-learning machinery explicit instead of hidden inside the trainer. In this repo, run_training() already drives the process unit by unit, trainer.train_period() does the actual learning, trainer.checkpoint() saves model/tokenizer/registry/config, and train_runner.py validates checkpoint compatibility before resuming. That means the clean design is to implement SMF and CASM as training modes inside the same existing loop, not as separate scripts.
Second, define the evaluation contract now. The casf_dataset_api already standardizes plasticity, stability, token_f1, and routing_acc, and the repo already evaluates after each period when eval_after_each_period is enabled. That gives you a strong baseline: SMF should improve retention over full fine-tuning, and CASM should improve retention further while also improving version-aware routing metrics.
2) SMF implementation plan
SMF in your project should mean: frozen base model + one sparse trainable memory module + no version routing. The point is to restrict updates to a small parameter subset while still using a single shared memory space. That fits the existing smf method flag already accepted by TrainConfig, so the first step is to actually give smf its own model wrapper and training path.
2.1 Add SMF-specific config knobs
Extend TrainConfig with SMF hyperparameters such as:
smf_memory_size
smf_sparsity_ratio
smf_update_layers
smf_regularization_weight
smf_freeze_backbone=True
Keep the existing method validator but add SMF-specific validation when method == "smf". For example, require a positive memory size and a valid sparsity ratio. The current config already centralizes method, precision, dataset, batching, and contradiction sampling settings, so this is the right place to put SMF controls.
2.2 Implement the SMF memory module
Create a small module that sits on top of the LLM and exposes a single trainable sparse memory. The easiest version is:
freeze the backbone,
attach one memory block to a small set of layers,
gate updates through a learned sparse mask,
keep one shared memory state across all periods.
You do not need routing or multiple slots for SMF. The memory is just a constrained trainable subspace. That makes SMF a clean baseline against catastrophic forgetting before you add versioning. This baseline also matches the idea of sparse memory finetuning described in your API docs, where the goal is versioned fact tracking only in CASM, not in SMF.
2.3 Modify the trainer for SMF mode
In CASFTrainer.__init__, switch the optimizer from all model parameters to only the SMF parameters when method == "smf". Right now the trainer constructs AdamW(self.model.parameters(), ...), so SMF needs to replace that with a parameter filter that excludes frozen backbone weights. The trainer already stores the model, tokenizer, config, registry, and detector, so you only need to add the right parameter selection logic.
Then update _train_step() so the model forward pass includes the sparse memory contribution, but not any router decision. The trainer already does a standard outputs = self.model(**batch) and backprop on outputs.loss, so the SMF model wrapper should make the sparse memory part of the forward path while keeping the backbone frozen.
2.4 Use the existing period loop unchanged
Keep the existing period sequence from training_plan.py and the existing runner behavior. The runner already iterates over periods, calls trainer.train_period(), checkpoints after each period, and evaluates after each period when configured to do so. For SMF, that loop should remain exactly the same; only the model internals change.
2.5 Add SMF metrics
For SMF, report:
retention on earlier periods,
current-period task performance,
average number of active sparse parameters,
forgetting after each new period.
Because the API already exposes stability and plasticity, you can report SMF gains directly in the same terms as the rest of the repo.
2.6 SMF deliverable
At the end of SMF implementation, you should be able to run:
full_ft baseline,
lora baseline,
smf baseline,
all inside the same runner, with SMF showing less forgetting than full fine-tuning but still using one shared sparse memory and no version routing. The config already recognizes smf, so the main work is implementing the actual sparse-memory model path and optimizer filtering.
3) CASM implementation plan
CASM should be built on top of SMF, not separately. In your repo’s terminology, CASM is the memory system that uses the contradiction detector, the versioned registry, and inference-time selection to preserve multiple coexisting factual states. The API docs explicitly frame the system that way: MemoryRegistry stores versioned facts, ContradictionDetector checks changed probes before training, and evaluate_versioned() measures whether the model routes to the correct version of a fact.
3.1 Extend SMF into a multi-slot memory bank
Replace the single SMF memory with a bank of slots. Each slot should have:
trainable parameters,
metadata (subject, relation, valid_from, valid_until),
usage counts,
parent / contradiction links.
The API docs already define MemorySlot and explain that closed slots are never deleted, so the correct CASM design is a version chain, not an overwrite buffer.
3.2 Add a router
CASM needs a router that chooses which slot(s) should influence a given query. The router input should be a compact query representation plus time signals. The output should be top-1 or top-k slot IDs and routing weights. This is the piece that turns “multiple adapters” into a true memory system, and it is also the part that gives you the routing_acc metric in the API.
3.3 Use the contradiction detector before training each period
The repo’s API docs show the intended sequence very clearly: on each period, load changed probes, run detector.check(changed, registry), train on passages plus the conflicts, then write the changed probes back into the registry. That should become your CASM training protocol. In code terms, the detector is not just a metric; it is the gate that decides whether to reuse a slot or create a new one.
3.4 Modify train_period() for CASM
Inside CASFTrainer.train_period(), the CASM branch should do this:
load the period’s passages,
collect changed probes,
call self.detector.check(probes, self.registry),
route each example to a slot,
create a new slot if the detector flags a contradiction,
train only the selected slots and the router,
write the updated facts to the registry.
The trainer already calls self.detector.check(probes, self.registry) before building the dataloader, and it already holds self.registry. That means the structural hook is already there; CASM just needs to make that call affect the memory update path, not merely run as a side check.
3.5 Add CASM-specific losses
Use a combined loss:
task loss,
router loss,
sparsity penalty,
anti-overlap penalty.
The task loss already exists through the trainer’s normal outputs.loss path. CASM adds the extra terms to force sparse slot usage and reduce slot collision. The API’s versioned evaluation and routing metrics make these losses worth adding because they directly support retention and correct version selection.
3.6 Persist versioning in checkpoints
CASM must checkpoint more than the backbone and tokenizer. The trainer already saves memory_registry.json alongside the model, tokenizer, config, and period marker. That is the right place to persist slot metadata, router state, and contradiction links too. If you do not checkpoint the registry and router cleanly, CASM will lose its version history on resume.
3.7 Keep the runner unchanged except for model loading
train_runner.py already has separate real and synthetic model factories and already uses run_training() to execute the loop. For CASM, add a build_casm_model_and_tokenizer() path parallel to the existing real and synthetic factories, then route mode == "real" or mode == "synthetic" through the CASM wrapper when cfg.method == "casm". That keeps the rest of the runner, manifest validation, and checkpoint logic intact.
3.8 Evaluate CASM with the right metrics
For CASM, the important outputs are:
plasticity on changed probes,
stability on unchanged probes,
contradiction accuracy,
routing accuracy.
Those are already part of the API’s evaluation vocabulary, so CASM should report them directly rather than inventing unrelated metrics.
4) Recommended build order
I would implement it in this order:
finish SMF first as the baseline sparse-memory method,
verify that it beats full fine-tuning on retention,
add multiple slots and the router,
connect the contradiction detector to slot branching,
add versioned checkpointing,
run the same temporal loop across aug_sep, sep_oct, oct_nov, and nov_dec,
compare full_ft, lora, smf, and casm with the same evaluation script. The repo’s existing periodized training plan and checkpointing flow make that progression natural.
5) What “done” looks like for each method
SMF is done when you can say: frozen backbone, one sparse trainable memory, no version routing, better retention than full fine-tuning. CASM is done when you can say: multiple versioned memory slots, contradiction-aware branching, router-based selection, and correct historical answers without overwriting older facts. The repo’s docs already define the evaluation story for both stability and versioned routing, so those are the right success criteria. 

