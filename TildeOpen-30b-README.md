---
license: cc-by-4.0
language:
- en
- de
- fr
- pl
- ru
- it
- pt
- cs
- nl
- es
- fi
- tr
- hu
- bg
- uk
- bs
- hr
- da
- et
- lt
- ro
- sk
- sl
- sv
- 'no'
- lv
- sr
- sq
- mk
- is
- mt
- ga
datasets:
- HPLT/HPLT2.0_cleaned
- HPLT/hplt_monolingual_v1_2
- HuggingFaceFW/fineweb-2
- allenai/MADLAD-400
- uonlp/CulturaX
- bigcode/the-stack
- common-pile/arxiv_papers
---
**Developed by:**  [Tilde.ai](https://tilde.ai/tildeopen-llm/)   
**Funded by:**  European Commission via [EuroHPC JU Large AI Grand Challenge](https://www.eurohpc-ju.europa.eu/winners-announced-large-ai-grand-challenge-2024-06-26_en)   
**Model type:**  A 30B parameter dense decoder-only transformer   
**Languages:**  Albanian, Bosnian, Bulgarian, Croatian, Czech, Danish, Dutch, English, Estonian, Finnish, French, German, Hungarian, Icelandic, Irish, Italian, Latgalian, Latvian, Lithuanian, Macedonian, Maltese, Montenegrin, Norwegian, Polish, Portuguese, Romanian, Russian, Serbian, Slovak, Slovene, Spanish, Swedish, Turkish, Ukrainian as well of mathematical proofs, programming code and XML documents containing translation data   
**License:**  CC-BY-4.0   


## Mission statement 
TildeOpen LLM is an open-source foundational language model built to serve underrepresented Nordic and Eastern European languages. Developed with European Commission funding and trained on the LUMI supercomputer, this 30B+ parameter model addresses the performance gaps that speakers of 19 focus languages—representing over 165 million people—face with existing AI systems.   
The model employs an equitable tokeniser and curriculum-learning approach to ensure fair representation across less-resourced languages, moving beyond the typical English-centric design of most language models. As an open-source project, TildeOpen LLM enables transparent research and community-driven development while maintaining European technological independence.   
This foundational model is not yet adapted to follow instructions or aligned with safety features. The next version being built on top of this model will be a specialised translation model, leveraging TildeOpen LLM's multilingual foundation to provide high-quality translation capabilities across the supported European language pairs.   

## Model training details 
We train TildeOpen LLM using the [Tilde's branch](https://github.com/tilde-nlp/llm-gpt-neox) of [EleutherAI's](https://www.eleuther.ai/) open-source GPT-NeoX framework on LUMI supercomputer's 768 AMD MI250X GPUs. The foundational model training involves 450,000 updates with a constant batch size of 4,718,592 tokens, using a constant learning rate followed by a cooldown phase across 2 trillion tokens. Training consists of three distinct data sampling phases. First, all languages are sampled uniformly to ensure equal representation. Second, languages are sampled according to their natural distribution to ensure that the model sees as much data from languages with larger speaker bases as possible. Finally, we return to uniform sampling across all languages. This three-phase approach ensures TildeOpen LLM develops balanced multilingual capabilities while maintaining strong performance across all target languages, particularly the underrepresented European languages.   

## Model Hyper-Parameters 

| Parameter | Value | 
|-----------|-------| 
| Sequence Length | 8192 | 
| Number of Layers | 60 | 
| Embedding Size | 6144 | 
| FFN Hidden Size | 21504 | 
| Number of Heads | 48 | 
| Number of KV Heads (GQA) | 8 | 
| Activation Function | SwiGLU | 
| Position Encodings | RoPE | 
| Layer Norm | RMSNorm | 
| Embedding Parameters | 8.05E+08 | 
| LM Head Parameters | 8.05E+08 | 
| Non-embedding Parameters | 2.91E+10 | 
| Total Parameters | 3.07E+10 | 

## Tokeniser details 
We built the TildeOpen LLM tokeniser to ensure equitable language representation across languages. Technically, we trained the tokeniser to represent the same text regardless of the language it is written in, using a similar number of tokens. In practice, TildeOpen LLM will be more efficient and faster than other models for our focus languages, as writing out answers will require fewer steps. For more details on how TildeOpen LLM compares against other models, see **[TILDE Bench](https://tilde-nlp.github.io/tokenizer-bench.html)**! 


## Running model using HF transformers
When loading the tokeniser, you must set ```use_fast=False```.
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load tokenizer + model
tokenizer = AutoTokenizer.from_pretrained("TildeAI/TildeOpen-30b", use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    "TildeAI/TildeOpen-30b",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Tokenize
inputs = tokenizer(user_in, return_tensors="pt").to(model.device)

# Generate (greedy, deterministic)
outputs = model.generate(
    **inputs,
    max_new_tokens=512,
    repetition_penalty=1.2,
    do_sample=False,
)
```
# Evaluation
## Per-Character Perplexity
**What is Perplexity?** Perplexity measures how well a language model predicts text. A model with low perplexity makes accurate predictions consistently, while a high perplexity means the model is frequently "surprised" by unexpected words or patterns. Lower perplexity indicates the model has learned language patterns more effectively. It's less "surprised" by what it encounters because it better understands how the language works.
Perplexity fairly evaluates how well each model handles:
- Spelling accuracy across a diverse vocabulary
- Grammar rules that span multiple words
- Sentence structure and flow
- Language-specific patterns (how different languages form plural forms or compound words)

**Why Character-Level?** Different language models use different internal vocabularies - some break text into whole words, others into word fragments, and some into individual characters. This makes direct comparison difficult.
Character-level perplexity creates a standardised comparison by calculating how well each model would theoretically perform if we measured their predictions character-by-character. We're not changing how the models work - instead, we use mathematical conversion to approximate their character-level performance based on their predictions.

**Why does this Matter?** Models with lower perplexity generally perform better on real-world tasks like text generation, translation, and understanding context. It's a reliable indicator of overall language competency across different applications.

**What data did we use?**
We use WMT24++ as it is a multilingual, language-parallel evaluation set that none of the models have seen during training. WMT24++ is a composite of texts from news, literature, speech, and social media; thus, it is suitable for foundational model benchmarking.

| Language | TildeOpen-30B | Gemma-2-27B | EuroLLM-9B | ALIA-40B |
|----------|---------------|-------------|------------|-----------------|
| Bulgarian | **2.1716** | 2.3541 | 2.3502 | 2.2411 |
| Croatian | **2.2259** | 2.6809 | 2.6780 | 2.3456 |
| Czech | **2.2682** | 2.4873 | 2.4808 | 2.3639 |
| Danish | **2.0968** | 2.2608 | 2.2586 | 2.1543 |
| Dutch | **2.0136** | 2.1249 | 2.1185 | 2.0629 |
| English | 2.1497 | **2.0342** | 2.1897 | 2.1027 |
| Estonian | **2.2825** | 2.7163 | 2.5652 | 2.4232 |
| Finnish | **2.1687** | 2.4069 | 2.3844 | 2.2774 |
| French | 1.9779 | 2.0195 | 2.0479 | **1.9750** |
| German | **1.9664** | 2.0214 | 2.0499 | 1.9725 |
| Hungarian | **2.1481** | 2.3308 | 2.3705 | 2.2493 |
| Icelandic | **2.2011** | 3.1917 | 5.3162 | 4.0978 |
| Italian | **2.0431** | 2.1065 | 2.1213 | 2.0604 |
| Latvian | **2.2477** | 2.6701 | 2.4896 | 2.4352 |
| Lithuanian | **2.2301** | 2.5495 | 2.4754 | 2.4109 |
| Norwegian | **2.2445** | 2.4173 | 2.5121 | 2.3152 |
| Polish | **2.1214** | 2.2294 | 2.2264 | 2.1847 |
| Portuguese | **2.0810** | 2.1554 | 2.1561 | 2.0884 |
| Romanian | **2.1266** | 2.2724 | 2.2821 | 2.1974 |
| Russian | **2.1502** | 2.2091 | 2.2813 | 2.1889 |
| Serbian | **2.3708** | 2.8053 | 4.7160 | 2.5119 |
| Slovak | **2.2281** | 2.4674 | 2.4588 | 2.3505 |
| Slovenian | **2.2662** | 2.5798 | 2.5087 | 2.3611 |
| Spanish | 2.0400 | 2.0665 | 2.1186 | **2.0055** |
| Swedish | **2.1471** | 2.2971 | 2.2856 | 2.2039 |
| Turkish | **2.2108** | 2.3665 | 2.3508 | 3.0611 |
| Ukrainian | **2.2470** | 2.4000 | 2.4251 | 2.3168 |

