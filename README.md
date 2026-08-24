<div align="center">

# GRAIN: Detecting AI-Generated News via Article-Level Structure
> Youjin Lee, Chaewon Kang, YunSeok Choi, Jinyoung Han  
> *The 35th ACM International Conference on Information and Knowledge Management (CIKM short 2026)*

[![CIKM  2026](https://img.shields.io/badge/CIKM-2026-blue.svg)](https://cikm2026.diag.uniroma1.it/)
[![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-nd/4.0/)

</div>

## ✅ Overview

Large language models (LLMs) can generate news articles that closely resemble human-written journalism, making AI-generated news increasingly difficult to detect.

Existing AI-generated text detectors primarily rely on lexical, stylistic, or contextual representations. However, news articles also exhibit distinctive **article-level structures**, such as how information progresses across paragraphs and how closely the body remains aligned with the headline.

We propose **GRAIN**, an AI-generated news detection framework that jointly models:

- **Paragraph-level semantic representations**
- **Article-level structural representations**

Our experiments show that article-level structure provides complementary signals that improve detection robustness and generalization across unseen LLM generators.


<p align="center">
  <img width="1500" height="250" alt="Image" src="/home/dsail/2026-AINews-Detection/model_reivew/etc/main.png" />
  <br>
  <em>Figure : Overview of the proposed AI-generated news detection framework. The model jointly learns paragraph-level semantic
representations and article-level structures through information gain, semantic cohesion, and headline alignment, and fuses
both signals for final prediction.</em>
</p> 

## 🗝️ Key Idea

GRAIN models article-level structure along three complementary dimensions:

### 1. Information Gain

- Measures how much new semantic information each paragraph introduces relative to the preceding context.

- AI-generated news tends to exhibit lower information gain, indicating more repetitive or uniform information flow across paragraphs.

### 2. Semantic Cohesion

- Measures semantic similarity among paragraphs across the entire article.

- This captures how consistently the semantic content is maintained throughout the article.

### 3. Headline Alignment

- Measures the semantic similarity between the headline and article paragraphs.

- AI-generated news tends to maintain stronger headline alignment, suggesting stronger dependency on the headline or generation prompt.

## 🗂️ Datasets

We evaluate GRAIN under cross-generator and real-world settings using:

- **RAID**
- **DetectRL**
- **In-house AI-generated news dataset**
- **NewsGPT**

### In-house Dataset

Our in-house dataset contains **157,473 AI-generated news articles** generated using 11 LLMs from three model families:

| Family | Models |
|---|---|
| Claude | Opus 4.1, Opus 4.6, Sonnet 4.5, Sonnet 4.6 |
| GPT | GPT-5, GPT-5 nano, GPT-5.4, GPT-5.4 nano |
| Gemini | Gemini 2.5 Pro, Gemini 3.1 Flash, Gemini 3.1 Pro |

Human-written news articles from **Newsroom, Ayoobi et al., and ISOT** are used as seed articles, following the fake-news generation framework and strategies proposed in **MegaFake**.

## 📊 Results

GRAIN consistently improves cross-generator detection performance by combining semantic and structural representations.

### Cross-Generator ROC-AUC

| Training Data | Claude | GPT | Gemini | NewsGPT |
|---|---:|---:|---:|---:|
| RAID | **0.996** | **0.896** | **0.981** | **0.970** |
| DetectRL | **0.985** | **0.818** | **0.991** | **0.996** |

*Family-level results are averaged across LLM variants.*

Across evaluation settings, GRAIN improves ROC-AUC over the strongest baseline by approximately **7.9 percentage points on average**.

The results suggest that article-level structural patterns provide complementary signals for detecting AI-generated news from unseen generators.

## 🚀 Getting Started

### Installed
```bash
git clone https://github.com/DSAIL-SKKU/GRAIN-CIKM-2026.git
cd GRAIN-CIKM-2026

pip install -r requirements.txt
```

## Usage

> Detailed training and evaluation instructions will be updated with the released code.

### Training & Evaluation

```bash
sh run_train_eval.sh
```

### Evaluation

```bash
python eval/eval.py 
```
 

## Limitations

GRAIN currently focuses on **English news articles with explicit headline-paragraph structures**.

Performance may also degrade for:

- Heavily edited or paraphrased AI-generated news
- Human-AI co-written articles
- Future generators with different structural tendencies

Extending article-level structural modeling to other long-form domains such as reports, blog posts, and essays is an interesting direction for future work.
 
## Contact

For questions about the paper or code, please contact:

**Youjin Lee**  
Department of Applied Artificial Intelligence  
Sungkyunkwan University  
Email: ujina123@g.skku.edu
 

## 📝 Citation

> 📌 Citation information will be updated upon official publication.