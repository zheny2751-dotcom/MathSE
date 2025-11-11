# MathSE ✨ Self-Evolution + Reflection SFT

`MathSE` (AAAI 2026) is a three-stage training pipeline that iteratively upgrades multimodal math models through GPT-4o knowledge distillation, self-evolving data collection, and reflection with Outcome Reward Model (ORM) feedback. This repository is the official implementation of *"MathSE: Improving Multimodal Mathematical Reasoning via Self-Evolving Iterative Reflection and Reward-Guided Fine-Tuning."*

---

## Workflow Overview 🚀

### Stage 1 📘 Knowledge Distillation
- **Entry point:** `main.py --stage distill` (see `run_phase1_sft.sh`).
- **Goal:** Bootstrap the base model using GPT-4o-distilled Chain-of-Thought traces (`D_SFT` ~ 100K examples from MathVL + curated open sets).
- **Process:** We fine-tune CogVLM2 / Qwen2-VL-7B / InternVL2.5-8B on the distilled subset to produce `M0`, ensuring the model starts from high-quality reasoning patterns.

### Stage 2 ♻️ Iterative Self-Evolving
- **Entry point:** `main.py --stage evolve`.
- **Goal:** Harvest additional supervision by letting `M_i` solve the remaining MathVL samples (`D_remain`) and filtering them with ORM.
- **Loop:** For each configured round (Algorithm 1):
  1. The **latest** checkpoint runs inference via `llamafactory.chat.ChatModel` (configured under the `inference` block) on a batch sampled from `D_remain` to produce reasoning traces `R_gen`.
  2. **ORM feedback:** accepts `R_correct`, logging `(prompt, input media, reasoning)` into `sft_chosen.jsonl`, and pushes mistakes plus diagnostics into the hard buffer. Incorrect items cycle back into `D_remain` for the next round.
  3. Update datasets: `D_SFT ← D_SFT ∪ R_correct`, refresh `D_remain`, and rewrite the LLaMA-Factory dataset (`mathse_stage_final.json`).
  4. (Optional) Re-train `M_{i+1}` with the expanded dataset via LLaMA-Factory before the next round.
- **Outcome:** After ~240K accepted samples we obtain a much stronger evolving model while caching all incorrect cases for reflection.

### Stage 3 💡 Reflection + Reward-Guided SFT
- **Entry points:** `prepare_reflection_data.py`, `run_phase2_reflection_sft.sh`.
- **Goal:** Turn the stored `R_incorrect` into ORM-guided reflections validated by GPT-4o, then continue SFT to reach the final checkpoint.
- **Process:**
  1. **GPT-4o reflection prompts:** provide each incorrect reasoning path, ORM error step, and analysis to GPT-4o to obtain `R_reflected`.
  2. **ORM validation:** re-score `R_reflected` to keep `R_reflect_correct`; drop noisy reflections.
  3. **Dataset fusion:** combine `R_reflect_correct` with `D_SFT` to create `output/reflection_sft.jsonl` (~280K records).
  4. **Final SFT:** launch LLaMA-Factory (LoRA or QLoRA configs under `configs/llama_factory/`) using the Stage-2 checkpoint plus the reflection dataset to produce `output/reflection_model_final/`.

---

## Outcome Reward Model (ORM) 🧠

- **Architecture:** CogVLM2-based classifier fine-tuned on 60K labeled paths (30K incorrect annotated with error step + explanation via GPT-4o, 30K correct CoT samples).
- **Signals:** (1) Correctness tag for each reasoning path; (2) Faulty step location `s_j`; (3) Natural-language error analysis `E_i`.
- **Usage:** Powers both Stage 2 filtering (accept vs. reject) and the Stage 3 reflection prompts/validation. ORM feedback is persisted inside `orm_reflections/` for traceability.

---

## Reflection Data Format 🧩

```json
{
  "instruction": "Solve the problem ...",
  "input": "",
  "draft": "First attempt reasoning ...",
  "reflection": "ORM critique + GPT-4o plan ...",
  "output": "Corrected reasoning with final answer."
}
```

`prepare_reflection_data.py` assembles the template above (Reflection-in-the-Loop style) before handing the data to LLaMA-Factory. Multiple drafts per question each become an independent row so the model observes diverse failure modes.

---

## Dataset & Benchmarks 📚

- **Training corpus:** MathVL (341K Chinese K12 multimodal problems) plus GeoQA+, Geometry3K, ChartQA, UniGEO-Calculation, MultiMath, MAVIS, Math-PUMA, and MathV360K. MathSE regenerates solutions, stretching average answer length from 325 -> 792 characters.
- **Benchmarks:** MathVL-test, MathVista, MathVerse, MathVision. The released MathSE checkpoints (CogVLM2, Qwen2-VL-7B, InternVL2.5-8B backbones) significantly outperform the corresponding base models on all four suites.

---

## Setup 🛠️

1. **Install core dependencies**
   ```bash
   pip install -r requirements.txt
   ```
2. **Clone & install LLaMA-Factory**
   ```bash
   git clone https://github.com/hiyouga/LLaMA-Factory.git ../LLaMA-Factory
   pip install -e ../LLaMA-Factory
   ```
3. **Configure ORM + GPT-4o credentials (optional reflection stage)**
   - Set `orm.endpoint`, `orm.api_key`, and `gpt4o.api_key` fields inside `config.yaml` (or environment variables consumed by the scripts).
4. **Edit `config.yaml`**
   - Central hub for dataset shards, backbone checkpoints, ORM thresholds, plus the `llama_factory` (training) and `inference` blocks (repo path, dataset directories, templates, batch sizes, inference backend, etc.).

---

## Running the Pipeline ▶️

| Stage | Command | Description |
| --- | --- | --- |
| Stage 1 Distillation | `./run_phase1_sft.sh --stage distill` | Fine-tunes the base model on GPT-4o distilled data and stores `output/sft_model_stage0/`. |
| Stage 2 Self-Evolving | `./run_phase1_sft.sh --stage evolve` | Runs the iterative loop, updates `output/sft_model_final/`, `output/hard_problems.jsonl`, and accumulates `output/sft_{chosen,rejected}.jsonl`. |
| Reflection Prep | `python prepare_reflection_data.py --config config.yaml` | Replays drafts, calls GPT-4o with ORM diagnostics, and writes `output/reflection_sft.jsonl`. |
| Stage 3 Reflection SFT | `./run_phase2_reflection_sft.sh` | Generates the reflection dataset and launches `llamafactory-cli train` for full-parameter SFT (no LoRA) on top of the Stage 2 checkpoint. |
| Full Pipeline | `./run_full_pipeline.sh` | Convenience entry to execute Stage 1 -> Stage 2 -> Reflection prep -> Stage 3 in sequence. |

`run_phase2_reflection_sft.sh` pulls all LLaMA-Factory hyperparameters from `config.yaml` and executes `python -m llamafactory.cli train ...`, so experiments only require editing that config.

---


## Citation 📝

If you use this repository or release derivatives, please cite:

```
@misc{chen2025mathseimprovingmultimodalmathematical,
      title={MathSE: Improving Multimodal Mathematical Reasoning via Self-Evolving Iterative Reflection and Reward-Guided Fine-Tuning},
      author={Jinhao Chen and Zhen Yang and Jianxin Shi and Tianyu Wo and Jie Tang},
      year={2025},
      eprint={2511.06805},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2511.06805},
}
```
