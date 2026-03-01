# RAW - What TriX Is Pointing At

Date: 2026-03-01
Method: Lincoln Manifold (Phase 1)

TriX is pretending to be an FFN, but it keeps mutating into a routing runtime.

The "real" thing here isn't ternary weights, or MoE, or even sparsity. It's the idea that *an address space of computation* is a product surface.

In normal ML, weights are diffuse. There is no address. If you ask "where does this concept live?" the honest answer is "everywhere and nowhere". TriX is trying to reverse that: carve the model into a bunch of little machines (tiles), then make the selection of those machines the core operation. The selection is observable, editable, compilable, and exportable. That smells like an operating system.

Once you see it as an OS, everything clicks:
- signatures are not just vectors; they are *addresses* (or at least keys)
- compiled dispatch is not just a speed trick; it is a *contract* (a program)
- surgery is not a hack; it is *patching the address space*
- bundles are not just serialization; they are *deploying a compute-plane*
- routing telemetry is not "logging"; it is *profiling a machine*

So what is TriX pointing at? A new kind of compute substrate where "what runs" is selected by content/address rather than by an explicit program counter.

But: "content addressing" is not enough. The hard part is meaning over time.

If tile 17 is the "string parser tile" today and the "random junk tile" tomorrow, then addressability was a mirage. So the real thesis becomes:

  Stable addresses are possible in learned systems, if you:
  - shape the geometry (regularizers, quantization, symmetry breaking)
  - monitor drift (churn curves, margins, near-ties)
  - freeze/compile where it matters (contracts)
  - keep semantics separate from acceleration (native is boring)

This is very close to "programming" but not in the normal sense. It's closer to:

  (1) create a library of subroutines
  (2) induce a key space that selects them
  (3) compile the selection for stable regions
  (4) ship the library + key space as an artifact

This repo has also been flirting with "cartridges": FFT, WHT, adders, opcodes. Those are not random. They are attempts to make the tile library feel like hardware: fixed semantics, composable atoms, deterministic.

And then it gets weirder: if routing is the primitive, then you can do compute by partitioning an input manifold and assigning subroutines to regions. That's a piecewise function machine.

The wild idea is: intelligence is not just a big dense function; it is an *addressable ecology* of small functions, where the address system is the thing you debug.

I keep thinking about "addressable intelligence" as: you don't ask a model "what is the output"; you ask "which parts did you invoke, and why." This becomes a governance / safety / interpretability primitive.

If TriX wins, the killer feature isn't speed. It's controllability.

You get:
- patching: change behavior by editing a few addresses
- provenance: track which addresses fired
- certification: compile known-safe address paths
- drift alarms: detect when the address semantics are changing

The scary part: if you get the address plane wrong, you get a chaotic MoE that collapses or thrashes.

So the question becomes: what is the simplest, most honest "address plane" that can remain stable, while still letting the system learn new subroutines?

Also: is the ternary thing essential, or just convenient?

Ternary makes packing and XOR+POPCNT distance cheap. It also makes signatures crisp. But the deeper idea could survive in float land as long as the address plane is discrete and monitored.

But there is a special vibe in ternary: it feels like building a digital computer inside a neural net. Not metaphorically. Literally: bits, popcount, muxes, LUTs.

TriX might be pointing at a synthesis of:
- MoE routing (conditional compute)
- content-addressable memory (address plane)
- circuit compilation (verified atoms)
- deployment contracts (compiled dispatch)
- OS-like observability (telemetry + drift)

The best future is a toolchain:

  train -> discover addresses -> edit -> compile -> validate -> bundle -> deploy -> monitor

And the most fun future is "cartridges": load a new compute plane into the router.

If I go fully wild: the next horizon is not "bigger models"; it's "models with stable internal address spaces" that can be updated like software.

Questions I can't stop thinking about:
- what is the minimal set of primitives (shapes) to make a tile library universal enough?
- does stability require an explicit memory model (state) and sequencing, or can it emerge?
- how do you prevent address hijacking (a tile drifting into a malicious behavior while keeping the same signature)?
- can you sign/certify a bundle and enforce it at runtime?
- can routing be made into a formal interface like an ABI?
