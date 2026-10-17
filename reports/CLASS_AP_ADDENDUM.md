# Exploratory class AP addendum

This secondary analysis was added while the 512-image RAM inference was running, before computing the primary comparison. It is not part of the original registered primary endpoint. The primary F1@5 protocol and parameters remain unchanged.

For RAM, CLIP and fused raw scores, calculate non-interpolated AP across the 384 evaluation images for each of the 69 shared classes. Group equal score thresholds; weight precision by the increase in recall, following the [scikit-learn metric definition](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html). Average over classes that have at least one positive reference; report every class's support and identify absent classes explicitly.

Do not present this restricted macro AP as full COCO-80 mAP or as verification of the thesis benchmark. Do not invent a cross-image class confidence score for the MMR top-five output; no class AP is assigned to that method. Treat rare-class AP cautiously. Report this as exploratory regardless of whether it favors the proposed fusion.
