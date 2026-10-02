---
library_name: transformers
license: apache-2.0
datasets:
- ADSKAILab/floora_dataset
base_model:
- Qwen/Qwen3-0.6B
- Qwen/Qwen3-1.7B
---

# FLOORA

FLOORA, short for Floor Layout Optimization with RL Alignment, is a compact domain-specific language model for architectural floor-plan generation. It generates structured multifamily residential layouts from building metadata, structural information, and massing geometry, producing a compact architectural DSL that can be deterministically parsed, validated, normalized, rendered, and converted back into geometry.

FLOORA is released in two sizes, [ADSKAILab/floora-0.6b](https://huggingface.co/ADSKAILab/floora-0.6b) based on Qwen3-0.6B and [ADSKAILab/floora-1.7b](https://huggingface.co/ADSKAILab/floora-1.7b) based on Qwen3-1.7B. Both use domain-specific pretraining, architect-edited supervised fine-tuning, and reinforcement learning with learned architect preferences and automatically verifiable geometric and functional rewards.

## Model Details

FLOORA takes structured architectural prompts containing building metadata, structural material, polygonal massing, and optionally partial space layouts. It autoregressively generates labeled floor-plan polygons for living units, corridors, and vertical circulation cores.

Its custom architectural DSL is compact, token-efficient, human-readable, and deterministically parseable. An equivalent example requires 642 DSL characters versus more than 10,000 characters in IFC.

The training pipeline includes architectural DSL construction and canonicalization, large-scale synthetic data generation, domain-specific pretraining, architect feedback collection, supervised fine-tuning on architect-edited layouts, reward-model training on pairwise architect preferences, and GRPO reinforcement learning with learned and verifiable rewards.

The Qwen3-0.6B and Qwen3-1.7B backbones contain approximately 0.6B and 1.7B parameters. Replacing the original vocabulary with the smaller FLOORA DSL vocabulary reduces the 0.6B variant to approximately 440M effective parameters, with the 1.7B variant reduced by a similar proportion.

- **Developed by:** Autodesk Research
- **Shared by:** Autodesk Research
- **Model type:** Autoregressive causal language model for structured architectural DSL generation
- **Language:** Architectural domain-specific language, not general natural language
- **License:** Apache-2.0
- **Finetuned from:** Qwen/Qwen3-0.6B, Qwen/Qwen3-1.7B
- **Nominal size:** 0.6B or 1.7B parameters
- **Effective size after vocabulary resizing:** approximately 440M parameters for the 0.6B variant
- **Domain:** Architecture, Engineering, and Construction
- **Primary task:** Massing-conditioned multifamily residential floor-plan generation

### Model Sources

- **Repository:** https://github.com/AutodeskAILab/floora
- **Dataset:** https://huggingface.co/datasets/ADSKAILab/floora_dataset
- **Paper:** [FLOORA: A Human-Aligned Domain-Specific Language Model for Architectural Design](https://arxiv.org/abs/2609.36064)

## Uses

### Direct Use

FLOORA is intended for rapid conceptual multifamily residential layout generation and design exploration. Inputs can include building metadata, structural material, building massing geometry, and partial space layouts. Outputs are `spaces` DSL blocks with polygons labeled `core`, `corridor`, and `living_unit`, which can be parsed and converted into geometric floor plans with the FLOORA utilities.

### Downstream Use

FLOORA can support conceptual floor-plan generation, automated layout exploration, architect review workflows, geometry-aware design optimization, architectural design research, structured AEC generation experiments, and research on domain-specific language models for engineering. The DSL and parser support validation, normalization, rendering, and conversion to geometry.

### Out-of-Scope Use

FLOORA is not intended as a general-purpose language model or architectural assistant, a replacement for licensed architectural or engineering judgment, a construction-document or building-code compliance system, a structural, mechanical, electrical, fire-safety, or life-safety engineering system, an unrestricted natural-language model, or an agentic architectural design system. Use outside the demonstrated multifamily residential scope requires additional evaluation.

Training and evaluation are scoped to multifamily residential floor-plan generation conditioned on building massings. Performance may degrade on highly irregular or underrepresented geometries.

## Bias, Risks, and Limitations

**Domain scope.** FLOORA is specialized for conceptual multifamily residential floor-plan generation and should not be treated as a general architectural design model.

**Geographic and architectural conventions.** Real-world evaluation uses multifamily building footprints from more than 50 major North American cities. Human feedback follows architectural conventions commonly observed in North America, so learned preferences may reflect those conventions.

**Synthetic-data dependence.** Most domain pretraining data are procedurally generated, with approximately 4.1 million samples before rotation augmentation and approximately 82 million after augmentation. Architect feedback improves real-world generalization, but procedural biases may remain.

**Distribution shift.** OpenStreetMap footprints differ substantially from the synthetic training distribution. Highly irregular or uncommon massings may be harder, especially when underrepresented during training and post-training. Evaluation on real-world OpenStreetMap massings provides the more informative out-of-distribution measurement.

**Verifier-defined metrics.** Similar geometric and functional specifications are used for synthetic-data filtering and verifiable evaluation rewards, so synthetic verifier performance is not fully independent evidence of architectural quality.

**Human preference data.** Ten practicing architects participated in feedback collection. Architectural quality is subjective, and preferences may differ across practitioners, regions, building programs, and design standards.

**VLM-based evaluation.** Some evaluations use Gemini 3.5 Flash as a pairwise vision-language-model judge. It agrees with architect-implied preferences on 77.5% of approximately 10,000 validation pairs, indicating substantial but imperfect agreement with professional judgment.

**Validity.** Reinforcement learning improves geometric and functional correctness, but outputs are not guaranteed to satisfy every architectural, regulatory, structural, accessibility, safety, or constructability requirement.

### Recommendations

Treat generated layouts as conceptual proposals rather than finished architectural solutions. Parse and validate the DSL, apply the provided geometric and functional verifiers, visually inspect outputs, use qualified architectural professionals for real-world decisions, evaluate before applying the model to new building typologies, regions, or design standards, and do not treat verifier success as equivalent to full architectural correctness or code compliance.

## Citation

```bibtex
@article{rezaei2026floora,
  title={FLOORA: A Human-Aligned Domain-Specific Language Model for Architectural Design},
  author={Rezaei-Shoshtari, Sahand and Wozniczka, Patryk and Ishida, Shu and Streuber, Gregg and Javadi, Farnoosh and Landes, Jeffrey and Ju, Angela and Azam, Muhammad and Lim, Bryan and Luttun, Johan and others},
  journal={arXiv preprint arXiv:2609.36064},
  year={2026}
}
```

## How to Get Started with the Model

The [FLOORA repository](https://github.com/AutodeskAILab/floora) provides inference utilities, the DSL parser, validation tools, and geometry-processing code for the generation workflow. Swap the model identifier for `ADSKAILab/floora-1.7b` to use the larger variant.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "ADSKAILab/floora-0.6b"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
# Disable eos-appending so the prompt ends at "</generate>", not at eos
if hasattr(tokenizer, "add_eos_token"):
    tokenizer.add_eos_token = False

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,
    trust_remote_code=True,
)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

prompt = """<context>
<building>
building { occupancy_type multifamily_residential storeys 6 level 2 elevation 60 }
</building>
<structure>
structure { material wood_frame }
</structure>
<massing>
massing { height 30 polygon 0,0 416,0 416,275 291,275 291,183 124,183 124,275 0,275 }
</massing>
</context>
<generate>
<space>
</generate>
<completion>
"""

inputs = tokenizer(prompt, return_tensors="pt").to(device)

# Stop at eos or, if present, the DSL's end-of-completion marker
stop_token_ids = {tokenizer.eos_token_id}
eoc_token_id = getattr(tokenizer, "eoc_token_id", None)
if eoc_token_id is not None:
    stop_token_ids.add(eoc_token_id)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=2048,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=list(stop_token_ids),
    )

completion = tokenizer.decode(
    outputs[0][inputs["input_ids"].shape[1]:],
    skip_special_tokens=False,
)

print(completion)
```
