# PaperTrail

Ask a question about computer science research and get an answer built from real arXiv papers, with every claim cited.

```
$ papertrail ask "How do people defend large language models against prompt injection?"

Defenses fall into several groups: prompt-level filtering and paraphrasing [arXiv:2511.18933],
reusing attack techniques against the injected instruction [arXiv:2411.00459], ...

Sources:
1. A Critical Evaluation of Defenses against Prompt Injection Attacks - https://arxiv.org/abs/2505.18333
...
```

## How it works

1. Each of 1,044,349 arXiv computer science papers was turned into a vector (384 numbers that capture its meaning) on a free Kaggle GPU.
2. Your question is turned into a vector the same way, and the closest papers are found in milliseconds.
3. An LLM reads those papers' abstracts and writes the answer, citing each paper.

## Docs

- [Architecture](docs/architecture.md) - system design, measured timings, and why each component was chosen
- [Setup](docs/setup.md) - what you need, how to build it, and how to run it

## Quick start

```bash
source ~/.venvs/papertrail/bin/activate
papertrail search "graph neural networks for molecules"
papertrail ask "How do diffusion models compare to GANs?"
```

Speed is measured on every run (`logs/timings.log`); changes are tracked in [`logs/CHANGES.md`](logs/CHANGES.md).
