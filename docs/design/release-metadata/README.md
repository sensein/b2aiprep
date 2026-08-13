# Design: pipeline-generated release metadata

Design documents for moving RO-Crate and C2M2 publication metadata out of a hand-written notebook and
standalone scripts, and into the release pipeline — so that metadata is derived from the run that
produced the data rather than typed afterwards.

These were produced with a [spec-kit](https://github.com/github/spec-kit) workflow in a separate
planning workspace and copied here so they sit alongside the code they describe. Read in this order:

| Document | What it is |
|---|---|
| [spec.md](./spec.md) | the problem, with each asserted defect verified against the published 3.0.0 record; 24 requirements, 12 success criteria |
| [research.md](./research.md) | R1–R9, the decisions and the evidence for each, checked against the release scripts and the built trees |
| [plan.md](./plan.md) | technical context, structure, delivery order, risks |
| [data-model.md](./data-model.md) | the run-record and per-unit-record shapes, and the artifacts generated from them |
| [contracts/cli.md](./contracts/cli.md) | the two new commands, their preconditions and exit codes |
| [quickstart.md](./quickstart.md) | end-to-end on the synthetic fixtures in `data/` |
| [tasks.md](./tasks.md) | 88 tasks in 8 phases; the MVP is Setup + Foundational + User Story 1 |

Two things gate a real release independently of the work itself: the release environment must be
installed from a tagged commit rather than a dirty tree, and the software cited in the provenance graph
needs durable identifiers registered before the release that cites them is published.
