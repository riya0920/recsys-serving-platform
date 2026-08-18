# Deliberately omitted

Things this project could have contained and does not, with the reason. Scope is
a design decision; an unbounded project is a project that never gets finished.

| Omitted | Why |
|---|---|
| Kubernetes | The failure modes this project is about (fallback on model death, shared limiter state, index memory) all reproduce with two containers behind Compose. K8s would add YAML and remove nothing from the risk list. Horizontal-scale plan is in `SCALING.md`. |
| GPU serving | The retrieval tower is 2 matmuls on a 64-d vector. It is memory-bandwidth trivial; a GPU would sit idle and multiply the cost per QPS. GPUs matter for the *ranker* once it grows past a small MLP, and that is stated as the trigger condition. |
| A feature store | Real-time features are the single biggest source of train/serve skew, and building a credible one is a project of its own. Instead this repo measures the *cost of staleness* directly, which is the input you would need to justify buying one. |
| Distributed FAISS / sharded index | 50K items x 64 dims x 4 bytes = ~13 MB. Sharding an index that fits in L3-adjacent memory is theatre. The threshold at which it stops fitting, and what to do then, is arithmetic in `SCALING.md`. |
| Online A/B testing | I do not have users. Every number here is labelled OFFLINE. Claiming A/B results without traffic is the fastest way to fail a portfolio review, and offline metrics are not a proxy for it. |
| Rich multi-objective ranking (dwell, diversity, freshness blending) | The ranker here optimises one objective. Multi-objective blending is where recommender work actually gets hard, and doing it badly would be worse than not doing it. Named as the next milestone, not claimed. |

## The rule behind the table

Every row is a thing I can describe the design of in an interview but chose not
to build, because building it would not have changed what this project
demonstrates. The things that *are* built are the ones where the interesting
behaviour only appears when you actually run it.
