# Conservative ontology alignment

Seven terminology mappings expand the available vocabulary from 69 to 76 COCO categories: airplane â†’ plane; fire hydrant â†’ hydrant; dining table â†’ dinning table (RAM vocabulary spelling); refrigerator â†’ fridge; skis â†’ ski; teddy bear â†’ teddy; tv â†’ television.

Mappings were recorded before new inference, without selecting aliases from prediction scores. The complete mapping is in `ontology-registration.json`. Canonical COCO reference labels and CLIP prompts stay unchanged; only RAM query indices change.

On the historic 384-image evaluation partition, this vocabulary can represent 1131/1177 annotation positives (96.09%). This is annotation coverage, **not measured recognition accuracy**. The earlier 69-class results remain unchanged and cannot be compared directly with a different candidate space.

Four categories remain unresolved: cell phone (smartphone is narrower; phone is broader), potted plant (houseplant does not imply a pot), suitcase (luggage is broader), toilet (toilet bowl is a component). No proxy mappings are silently scored as equivalent.

The versioned `experiment_ontology` runner preserves the frozen historical runner. It uses the same verified model weights, native-query equivalence check, canonical prompts and ranking parameters. The new eight-image real-model run is an integration check on previously observed images; it is not a fresh independent benchmark. A larger newly registered cohort is required before drawing performance conclusions. Pretraining overlap remains unknown.

## Recorded integration result

RAM and CLIP completed on all eight real images. All 76 query scores were finite. Maximum discrepancy between full native RAM queries and selected queries: 0.0. The saved raw outputs include scores, canonical predictions, image checksums, timing and model checksums. No accuracy improvement is inferred from this small observed cohort.
