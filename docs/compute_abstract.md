# Compute application — abstract

A self-contained, conceptual abstract for compute-grant applications.
Under 350 words.

---

**What it is.** A research project on training language-model agents to be *inherently curious* — to probe a problem before committing to an answer. Unlike LLM agents that answer immediately or follow scripted tool-use, the curiosity here is learned through reinforcement learning, with no human-labeled rationale or hand-designed curiosity reward. The agent gets one or more free turns to probe or run scratchpad code, then a committing turn whose final program is scored. Standard RL treats every generated word as equally responsible for the score, which gives the wrong signal when an early turn's job is to gather information: a useful probe and a working program look the same as a useless probe and a lucky guess.

**What is new.** A way to assign credit per turn without training a separate value network. For each problem the model attempts several different early turns and several follow-ups under each; the spread of final scores reveals — without supervision — which early turns were genuinely informative. A memory-saving trick lets the method fit on a single mid-range research GPU.

**What is done.** The trainer and credit-assignment scheme are implemented and tested on smoke runs. A custom evaluation environment is built in which the agent controls a small physical system whose key parameter is hidden, so probing is the only way to know what the answer needs. Reference baselines without RL show the task is far from solved, and that the model spontaneously *writes* a probe-and-adapt protocol but does not *use* what it gathers. The gap to close is not in formatting or instruction-following — it is in transferring value between turns.

**What needs to be done.** A clean comparison of standard one-scalar-for-everything training, branched training without per-turn credit, and our per-turn-credit method — same model, same environment, to isolate whether the per-turn signal closes the value-transfer gap or branching alone suffices. Each arm is a multi-day GPU run; the compute ask covers the full set plus held-out evaluation.
