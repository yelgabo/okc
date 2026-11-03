# Short-token incumbent diagrams

[Open all diagrams](index.html).

These diagrams expand the public607 decoder-layer source. They retain its selected native T1/T8 paths and distinguish disabled alternatives.

- [Tiny / Tiny2 hardware](tiny-hardware.html)
- [Tiny / Tiny2 phase handoff](tiny-phases.html)
- [Attention internals](attention-detail.html)
- [BF16 projection internals](projection-detail.html)
- [Post-attention launch chain](post-attention-detail.html)

Each node links to pinned source. T is batch times query length. T1 is decode1; T8 includes verify1 with eight query tokens. Other shapes with T8 can use different attention arithmetic.

The diagrams describe operations and dependencies. Box area carries no timing meaning. Internal wait costs remain unqualified. The checked-in CSS keeps the plain gray appearance.
