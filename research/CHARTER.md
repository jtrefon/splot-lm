# SPLOT foundational research charter
Maintainer-approved scientific direction, 2026-11-28. English canonical · [Polski](CHARTER.pl.md). This charter defines the research programme, not an accepted architectural change or an empirical result.

## North star
**Achieve more through better architecture, not greater scale.**

## Vision
A future in which advances in artificial intelligence are driven increasingly by architectural innovation rather than proportional increases in computational resources.

## Mission
To research, develop and openly validate language-model architectures that improve the relationship between model quality and the resources required for learning and inference.

## Fundamental research question
How can the architectural organization of language models be improved to achieve better capabilities at equivalent or lower computational cost?

## Scientific identity
Independent, open, architecture-first research into efficient machine intelligence. Scale is not the enemy: we investigate how efficiently additional resources translate into capabilities, at multiple scales. Commodity hardware is an accessible experimental environment and a demanding validation constraint, not the scientific objective or ceiling. SPLOT is a research programme, not a commitment to one model, vendor, benchmark or mechanism.

## Foundational principles
1. **Architecture over brute force:** investigate better organization of computation rather than relying exclusively on scale.
2. **Quality without compromise:** efficiency is meaningful relative to the capabilities preserved or gained; disclose trade-offs, do not conceal regressions.
3. **Efficiency by design:** computation, memory, numerical representation and data movement are architectural considerations.
4. **Evidence over assumptions:** every candidate must earn its place through falsifiable, controlled experiments; negative results matter.
5. **Open science:** publish methods, evidence, limitations and prior art to enable independent verification, within data rights and privacy constraints.

## First scientific milestone
Demonstrate whether one or more SPLOT mechanisms can achieve equivalent or superior held-out language-model quality at measurably lower training or inference cost than a comparable, well-optimized conventional architecture, without material regression in relevant capabilities. This is an objective, not a claim or guaranteed outcome. A quality improvement at equivalent cost is also a valid advance; report actual trade-offs.

Nano is a scientific instrument for mechanism discovery, Mini a subsequent validation scale, and larger configurations a means to test generalization and scaling. A tiny synthetic or bounded real pilot establishes feasibility, not general intelligence. No universal IQ score, composite footprint index or fixed hardware target is required.

The [evaluation and decision framework](evaluation/README.md) operationalizes this charter. [Hypotheses](hypotheses/README.md), [roadmap](../ROADMAP.md) and architecture contracts are subordinate and revisable. The founding notes remain historical. Changes to the charter require explicit maintainer review; evidence, not adherence to current technology, determines architectural adoption.
