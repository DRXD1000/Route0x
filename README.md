<p align ="center">
<img width=350 src = "./images/route0x_logo.png">
</p>


<div align="center">

[![Downloads](https://static.pepy.tech/badge/route0x)](https://pepy.tech/project/route0x)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)]()
[![License:](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![package]( https://img.shields.io/badge/Package-PYPI-blue.svg)](https://pypi.org/project/route0x/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.11093524.svg)](https://doi.org/10.5281/zenodo.11093524)

</div>

<hr>

<div align="center">

  <h4 align="center">
    <b>Low latency, High Accuracy, Custom Query routers for Humans and Agents </b>
    <br />
  </h4>
</div>


The query `"How can I improve my credit score?"` is `In-Domain (ID)` but `Out-of-scope (OOS)` for a banking chatbot and `Out-Of-Domain (OOD) OOS` for a e-Commerce chatbot. A good query router should be able to learn to accept **ID In-Scope** queries and **Reject both ID OOS and OOD OOS**. Query routing cannot be jsut mapped to Intent classification alone and it won't work owing to the above requirements. Research literature hints Chatbots (or more formally) Task-Oriented Dialogue Systems (TODS) and Goal Oriented Dialogue Systems (GODS) have been grappling with this hard problem for a long time now. route0x is a humble attempt to tackle this issue and offer a production-grade solution for Humans and Agents.

**KPI for Query Routing:** $ / Query. (this subsumes accuracy, latency)

> Hitherto, route0x is the `only (practically) free, low latency (no external I/O needed), DIY + Improvable, low shot (only 2 user samples per route/intent needed), production-grade solution, that is FT for only 100 steps` that matches or beats contemporay Query Routing Researches and Solutions. We tested it on `8 different TODS datasets (some are deployed systems) from different domains and grains` against 2 different researches and 1 library.*

Check out the highlight reel of empirical evals and/or even dig deep with more numbers or get your hands-on with the starter notebook.

## Table of Contents

- [Route0x: Getting Started](#route0x-getting-started)
  - [Training](#training)
  - [Inference](#inference)
- [Route0x Evals: A Highlight reel](#route0x-evals-a-highlight-reel)
  - [1. Route0x x Amazon Research 2024: HINT3 OOS Dataset](#1-route0x-x-amazon-research-2024-hint3-oos-dataset)
  - [2. Route0x x Salesforce Research 2022: CLINC OOS Dataset](#2-route0x-x-salesforce-research-2022-clinc-oos-dataset)
  - [3. Route0x x Aurelio Labs' Semantic Router](#3-route0x-x-aurelio-labs-semantic-router)
- [I want to use it](#i-want-to-use-it)
- [I want to know how it works](#i-want-to-know-how-it-works)
- [I want to see the detailed empirical evals](#i-want-to-see-the-detailed-empirical-evals)
- [Roadmap](#roadmap)
- [Limitations](#limitations)
- [Citations](#citations)


## Route0x: Getting Started

We have disetangled the resource heavy route building (entails model training) from query routing (entails quick model inference)


### Training
```python 
pip install route0x[build] # installs a package thats few GBs
```
### [Build your router - Starter Notebook]()

### Inference

```python 
pip install route0x[route] # installs a package thats few MBs
```
```python
from route0x import RouteFinder
query_router = RouteFinder(<your_route0x_model_path>)
route_obj = query_router.find_route(<your-query>)
```

## Route0x Evals: A Highlight reel.

### 1. [Route0x x Amazon Research 2024: HINT3 OOS Dataset](https://arxiv.org/pdf/2410.01627)

**Goal of this comparison:** To show that Route0x offers:

- A better perf wrt  SetFit + NA or 
- A competitive perf wrt expensive & slow vanilla LLMs 
- A competitive perf wrt expensive & slow SetFit + NA + LLMs based approaches cited in the paper.

<img src="./images/HINT3-F1.png"/><br/><br/>
<img src="./images/HINT3-OOS-RECALL.png"/><br/><br/>

**P50 Latency:** Numbers taken from the Amazon research paper and compared with Route0x system that uses FP32 ONNX and more.

<img src="./images/p50 Latency.png"/><br/><br/>

Caveat: Route0x uses SetFit as a base. But not as-is, we added a tweak on how we use it and a few more innovations on top of SetFit. (Jump to the sections below for more details). But unlike the SetFit + NA option suggested in the paper, We do not perturb positive samples to create hard negatives, We employ a straight forward Low-shot learning regime that only needs 2-samples from real dataset to simulate real world query routing users (who might find it hard to offer more samples for each route of interest). We augment those 2 sampls into to 12 effective samplesm, more on this later. Also we train only for 100 steps. The LLMs used in the paper as for as we can tell are hosted APIs hence by design suffers high latency due to network I/O and incurs $ which makes it infeasible for query routing which might touch many queries even if with uncertainity routing.

Note: All numbers are based on MPS GPU device. Numbers can slightly vary based on the device and seeds. As the paper shows only the 
best number without any notion of variations denoted usually with ±.

<img src="./images/HINT3-Training Regime.png"/>




### 2. [Route0x x Salesforce Research 2022: CLINC OOS Dataset](https://arxiv.org/pdf/2106.04564)

[Datasets](https://huggingface.co/datasets/Salesforce/dialogstudio) 

**Goal of this comparison:** To show that Route0x offers robust in-scope accuracy in the presence of ID OOS and OOD OOS.

<img src="./images/FS-CLINC-IS-ACC.png"/><br/><br/>
<img src="./images/FS-CLINC-OOS-RECALL.png"/><br/><br/>

Caveat: In this Route0x model is all-mpnet-base-v2 is compared against purpose-built architectures like TOD-BERT which are architected for and trained on TODS style intent detection. 

Note: All numbers are based on MPS GPU device. Numbers can slightly vary based on the device and seeds. As the paper shows the 
all numbers with uncertainity we present numbers from 3 runs and denote the variations with ±.


<img src="./images/FS-CLINC-Training Regime.png"/>


### 3. [Route0x x Aurelio Labs' Semantic Router](https://github.com/aurelio-labs/semantic-router)

**Goal of this comparison:** To show that Route0x beats pure embedding similarity + threshold based approach for query routing.

<img src="./images/FOOD_SR.png"/><br/><br/>
<img src="./images/CC_SR.png"/><br/><br/>
<img src="./images/BANK_SR.png"/><br/><br/>
<img src="./images/BANK77_SR.png"/><br/><br/>
<img src="./images/CUREKART_SR.png"/><br/><br/>
<img src="./images/PP11_SR.png"/><br/><br/>
<img src="./images/SOFM_SR.png"/><br/><br/>

### I want to use it
### I want to know how it works
### I want to see the detailed empirical evals
### Roadmap
### Limitations
### Citations