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

FLOORA, short for Floor Layout Optimization with RL Alignment, is a compact domain-specific language model for architectural floor-plan generation.

The model generates structured multifamily residential floor layouts conditioned on building metadata, structural information, and building massing geometry. Rather than generating images or unrestricted natural language, FLOORA produces a compact architectural domain-specific language that can be deterministically parsed, validated, normalized, rendered, and converted back into geometry.

FLOORA is released in two sizes, [ADSKAILab/floora-0.6b](https://huggingface.co/ADSKAILab/floora-0.6b) (based on Qwen3-0.6B) and [ADSKAILab/floora-1.7b](https://huggingface.co/ADSKAILab/floora-1.7b) (based on Qwen3-1.7B), both trained using domain-specific pretraining, architect-edited supervised fine-tuning, and reinforcement learning with a combination of learned architect preferences and automatically verifiable geometric and functional rewards.

## Model Details

### Model Description

FLOORA is a family of domain-specific language models developed for structured architectural layout generation. This model card describes the Qwen3-based FLOORA-0.6B and FLOORA-1.7B models.

The model takes a structured architectural prompt containing information such as building metadata, structural material, and a polygonal building massing, and autoregressively generates labeled floor-plan polygons representing living units, corridors, and vertical circulation cores.

A custom architectural DSL is used rather than conventional BIM formats such as IFC. The representation is designed to be compact, token-efficient, human-readable, and deterministically parseable. In the example reported in the paper, the DSL representation requires 642 characters compared with more than 10,000 characters for an equivalent IFC representation.

The complete training pipeline consists of

1. Architectural DSL construction and canonicalization
2. Large-scale synthetic data generation
3. Domain-specific pretraining
4. Architect feedback collection
5. Supervised fine-tuning on architect-edited layouts
6. Reward-model training on pairwise architect preferences
7. GRPO reinforcement learning with learned and verifiable rewards

The nominal Qwen3-0.6B and Qwen3-1.7B backbones contain approximately 0.6B and 1.7B parameters respectively. After replacing the original vocabulary with the smaller FLOORA DSL vocabulary and resizing the embedding layers, the effective parameter count reported in the paper for the 0.6B variant is approximately 440M parameters; the 1.7B variant is reduced by a similar proportion.

- **Developed by:** Autodesk Research
- **Funded by:** Not specified in the paper
- **Shared by:** Autodesk Research
- **Model type:** Autoregressive causal language model specialized for structured architectural DSL generation
- **Language(s) (NLP):** Architectural domain-specific language. The model is not intended as a general natural-language model.
- **License:** Apache-2.0
- **Finetuned from model:** Qwen/Qwen3-0.6B, Qwen/Qwen3-1.7B
- **Nominal model size:** 0.6B or 1.7B parameters, depending on variant
- **Effective parameter count after vocabulary resizing:** approximately 440M parameters for the 0.6B variant
- **Primary domain:** Architecture, Engineering, and Construction
- **Primary task:** Massing-conditioned multifamily residential floor-plan generation

### Model Sources

- **Repository:** https://github.com/AutodeskAILab/floora
- **Dataset:** https://huggingface.co/datasets/ADSKAILab/floora_dataset
- **Paper:** FLOORA: A Human-Aligned Domain-Specific Language Model for Architectural Design
- **Demo:** Not specified in the paper

## Uses

### Direct Use

FLOORA is intended for generating conceptual multifamily residential floor layouts from structured building descriptions.

Typical inputs contain

- Building metadata
- Structural material
- Building massing geometry
- Optionally, partial space layouts

The model generates a `spaces` DSL block containing polygons labeled as

- `core`
- `corridor`
- `living_unit`

The generated DSL can subsequently be parsed and converted into geometric floor-plan representations using the FLOORA parser and inference utilities.

The primary intended use is rapid conceptual architectural layout generation and design exploration rather than final building documentation.

### Downstream Use

FLOORA may be integrated into architectural design and generative-design systems that require structured, geometrically interpretable outputs.

Potential downstream applications include

- Conceptual multifamily floor-plan generation
- Automated layout exploration
- Generation of candidate layouts for architect review
- Geometry-aware design optimization workflows
- Architectural design research
- Structured generation experiments in Architecture, Engineering, and Construction
- Research into domain-specific language models for engineering applications

The DSL and parser allow downstream systems to validate, normalize, render, and convert model outputs into geometric representations.

### Out-of-Scope Use

FLOORA is not intended to be used as

- A general-purpose language model
- A general-purpose architectural assistant
- A replacement for licensed architectural or engineering judgment
- A system for producing construction-ready documents
- A building-code compliance system
- A structural, mechanical, electrical, fire-safety, or life-safety engineering system
- A generator for building types outside the model's demonstrated multifamily residential scope without additional evaluation
- A model for unrestricted natural-language generation
- An agentic architectural design system

The training and evaluation data are specifically scoped to multifamily residential floor-plan generation conditioned on building massings.

Performance may degrade for highly irregular building footprints or geometries that are poorly represented in the training and post-training distributions.

## Bias, Risks, and Limitations

FLOORA has several important limitations.

**Domain scope**

The model is specialized for conceptual multifamily residential floor-plan generation. Results should not be interpreted as demonstrating general architectural design capability.

**Geographic and architectural conventions**

The real-world evaluation dataset contains multifamily building footprints collected from more than 50 major North American cities. The architects providing human feedback were instructed to evaluate layouts according to architectural conventions commonly observed in North America. The learned design preferences may therefore reflect North American multifamily design conventions.

**Synthetic-data dependence**

Most domain pretraining data are procedurally generated. The synthetic corpus contains approximately 4.1 million samples before rotation augmentation and approximately 82 million samples after augmentation. Although post-training with architect feedback substantially improves real-world generalization, the model may retain biases introduced by the procedural generation system.

**Distribution shift**

OpenStreetMap building footprints differ substantially from the synthetic training distribution. Highly irregular or uncommon massings may remain difficult, particularly when their geometry is underrepresented in pretraining and post-training data.

**Verifier-defined metrics**

The same broad geometric and functional specification is used both when filtering portions of the synthetic training data and when constructing verifiable evaluation rewards. Synthetic verifier performance should therefore not be interpreted as fully independent evidence of architectural quality.

The paper identifies evaluation on real-world OpenStreetMap massings as the more informative out-of-distribution measurement.

**Human preference data**

Architect feedback reflects the judgments of a finite group of professional practitioners. Ten practicing architects participated in the feedback collection process.

Architectural quality can be subjective, and alternative practitioners, regions, building programs, or design standards may prefer different layouts.

**VLM-based evaluation**

Some evaluations use Gemini 3.5 Flash as a pairwise vision-language-model judge. The judge agrees with architect-implied preferences on 77.5% of approximately 10,000 validation pairs, indicating substantial but imperfect agreement with professional human judgment.

**No guarantee of validity**

Although reinforcement learning substantially improves geometric and functional correctness, generated layouts are not guaranteed to satisfy every architectural, regulatory, structural, accessibility, safety, or constructability requirement.

### Recommendations

Generated layouts should be treated as conceptual design proposals rather than finished architectural solutions.

Users should

- Parse and validate generated DSL before downstream use
- Apply the provided geometric and functional verifiers
- Visually inspect generated layouts
- Use qualified architectural professionals for real-world design decisions
- Perform additional evaluation before applying the model to other building typologies, regions, or design standards
- Avoid treating verifier success as equivalent to full architectural correctness or code compliance

## How to Get Started with the Model

The FLOORA repository provides the inference utilities, DSL parser, validation tools, and geometry-processing code required for the complete generation workflow.

Repository

https://github.com/AutodeskAILab/floora

A minimal Transformers loading pattern is shown below. Swap the model identifier for `ADSKAILab/floora-1.7b` to use the larger variant.

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