# GGSOM

MATLAB code for novelty detection from the paper:

**Granular growing self-organizing map for novelty detection and incremental feature selection**

Yuanhao Sun, Ping Zhu, Kanglin Qu, and Jiucheng Xu.
*Information Sciences*, 758, 123969 (2027). Available online 1 August 2026.

[Read the paper](https://doi.org/10.1016/j.ins.2026.123969)

GGSOM combines granular computing and fuzzy rough sets with a growing self-organizing map for novelty detection in partially labeled data streams.

The main model is `GGSOM_ND.m`. The original Windows MEX file and its C++ source are included; on other platforms, compile it with `mex my_FN_Relation_U.cpp`.

If you use this code in your research, please cite the paper above.
