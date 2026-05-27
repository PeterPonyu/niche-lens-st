# Baseline repository checkout instructions

This folder intentionally does **not** vendor third-party code. Run these commands from the NicheLens-ST repository root when baseline execution begins.

```bash
mkdir -p .external_baselines
cd .external_baselines

git clone https://github.com/Super-LzzZ/CellNiche.git CellNiche
cd CellNiche && git checkout af58974ded7cf57299a9f8952d4cc6dffee39c6f && cd ..

git clone https://github.com/ZijieJin/scComm.git scComm
cd scComm && git checkout ed0f372fdc122333afa834150c566948bef68a29 && cd ..
```

Before code reuse, re-check licenses, repository HEADs, and paper/code terms.
