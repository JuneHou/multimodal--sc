# References

- Synthetic Historical Control for Policy Evaluation.
  - Source used for this skill: the user-provided preprint `Synthetic Historical Control for Policy Evaluation` by Yi-Ting Chen, Jui-Chung Yang, and Tzu-Ting Yang.
  - Core implementation hooks carried into this skill:
    - residualize the observed series before matching;
    - smooth the latent component with local linear regression;
    - match a treated pre-block against earlier historical blocks from the same series;
    - keep donor weights on the simplex;
    - use a stepwise donor selection procedure to stay parsimonious;
    - switch to differenced data when raw-level matching is poor, following the paper's detrending guidance.
