# Leapable Law - 15-minute operator pitch

This script is for the sealed Cuyahoga County proof of concept. It demonstrates
what the production artifacts prove; it is not legal advice, a prediction
product, or a claim about courts outside the frozen corpus.

## Before the clock

Open a fresh GPU-workstation terminal. The production ten-lens resident must already
be ready on `127.0.0.1:18401`. Run:

```bash
cd <clean-calyx-checkout>
tools/lawdemo/run_demo.sh --out "$HOME/calyx/fsv/demo-harness-$(date -u +%Y%m%dT%H%M%SZ)"
```

For a determinism replay, preserve the first run and add:

```bash
tools/lawdemo/run_demo.sh \
  --out "$HOME/calyx/fsv/demo-harness-replay-$(date -u +%Y%m%dT%H%M%SZ)" \
  --expect-headline <first-run>/headline.json
```

The harness prints every underlying command. Timings vary; `headline.json`
contains only deterministic display numbers and identities.

## 0:00-1:00 - establish the boundary

Setup line: “This is 12,596 physical Cuyahoga County appellate-opinion rows,
served by ten resident GPU content lenses with zero CPU model lanes.”

Command: let the harness execute `panel resident ready` and a fresh unbounded
`readback cx-list` against the production vault.

What appears: ready resident identity, ten GPU lenses, zero CPU lenses, and a
12,596-row live Base readback whose county field is Cuyahoga on every row.

Takeaway: the show begins from durable live records and an attested measurement
panel, not presentation text.

Anticipated objection — “Does this cover Ohio?” Answer: “No. The claim is
Cuyahoga-only. A result outside that boundary is refused or labeled frontier.”

## 1:00-3:00 - Demo 1, doctrine reading list

Setup line: “A machine-discovered 3,282-opinion criminal/juvenile-procedure
scope was reduced to a 331-member recall kernel without losing measured held-
out coverage.”

Command: point to the printed fresh `readback kernel-health` and scoped live
Base readback. The ranking build is explicitly checksum-verified, then its top
ten identities are rejoined to live rows.

What appears: 331 kernel members, 422 kernel-graph members, recall 1.0 over
n=17 against a 0.95 gate, groundedness 1.0 over n=331, and ten named cases.

Takeaway: the reading list is a traceable presentation slice of a measured
recall kernel, not an editor's unexplained list.

Anticipated objection — “Are these the only controlling cases?” Answer: “No.
They are ranked, provenance-backed research hypotheses inside the sealed
scope. Counsel still reads the cases and current law.”

## 3:00-5:00 - Demo 2, judge authority diet

Setup line: “For Mary J. Boyle, the accepted corpus resolves 862 authored
opinions and builds an independent 88-member recall kernel.”

Command: point to the printed fresh Boyle-vault `kernel-health` and Base
readback. The citation-diet projection is checksum-verified and its top five
authorities are joined back to live production rows.

What appears: complete authored denominator n=862, dissent count 1/862, 2,221
in-corpus cited-authority edges, 10,399 frontier edges, and named top
authorities with edge and depth counts.

Takeaway: this is a denominator-bearing historical research profile with an
explicit corpus boundary.

Anticipated objection — “Is this predicting how a judge votes?” Answer: “No.
Vote and outcome prediction are not grounded here. The artifact describes
authorship and cited-authority history only, with low-n and unresolved rows
kept visible.”

## 5:00-7:00 - Demo 3, sanctions shield

Setup line: “This three-page fixture contains nine correctly characterized
real cases, two fabricated identities, and one real case attached to the wrong
proposition.”

Command: point to the printed immutable-generation verification and the live
Base readback of the flagged *State v. Bonnell* target. Say aloud that the
original 32-search build took 13:26.58 and is therefore verify-then-display in
the 15-minute harness.

What appears: 12 citations, 10 found, two `NOT_IN_CORPUS`, one `LOW_SUPPORT`,
12/12 reviewed agreement, and the negative paired ten-lens score for C12.

Takeaway: fabricated identities cannot acquire physical provenance, while a
real identity can still be separately flagged for weak proposition support.

Anticipated objection — “Does NOT_IN_CORPUS mean the case does not exist?”
Answer: “No. It means no matching physical identity exists in this frozen
Cuyahoga generation. Another court, county, or snapshot is outside the claim.”

## 7:00-8:30 - Demo 4, dissent intelligence

Setup line: “The live opinion-type census contains 55 separate dissent rows
and 42 concurrence-family rows among 12,596 opinions.”

Command: point to the fresh Base-derived type counts and the checksum-verified
dissent report.

What appears: fully denominated counts; community 11 with 32 dissents among
n=585 members; and a structural null — zero of 28,834 accepted citation edges
targets a separately stored dissent opinion.

Takeaway: the system refuses to manufacture dissent traction that its current
edge representation cannot attribute.

Anticipated objection — “So no one cited those dissents?” Answer: “That is not
the claim. The current overlay cannot distinguish a case citation from a
citation to a separately stored dissent part. The display is a representation
deficit, not a legal conclusion.”

## 8:30-13:30 - Demo 5, paths and refusal

Setup line: “Now the engine derives two paths live, refuses Delaware, and
replays one recorded derivation bit-for-bit.”

Command: let the harness print and execute two `kernel-answer` commands, the
Delaware `kernel-answer`, `ledger-tail` before and after refusal, and live
`reproduce --record`.

What appears: named traffic-stop and foreclosure association paths; similarities
above the sealed admission threshold; structured
`CALYX_KERNEL_QUERY_OUT_OF_SCOPE`; byte-identical ledger head across refusal;
and equal original/reproduced derivation hashes.

Takeaway: a recorded path can be re-derived; a foreign jurisdiction gets a
named refusal rather than a plausible paragraph.

Anticipated objection — “How is this not an LLM wrapper?” Answer: “There is no
LLM answer generator on this path. The command measures embeddings, selects a
physical node, traverses a persisted weighted graph, and records ledger rows.
It returns named paths, not generated legal prose.”

Anticipated objection — “What happens when citations leave Cuyahoga?” Answer:
“The citation overlay records frontier exits. The current `kernel-answer`
command is association-only and cannot yet emit a typed named frontier stop;
that limit is tracked and is never disguised as a citation path.”

## 13:30-15:00 - close on verification

Setup line: “We finish by rereading the entire append-only chain and the same
resident process.”

Command: point to the printed full `verify-chain` and final `panel resident
ready` commands.

What appears: chain status `ok`, no break, the same resident PID, ten warmed
GPU lenses, zero CPU lenses, per-section timing, canonical headline JSON, and
a checksum manifest for the run.

Takeaway: the claims survive full physical readback after the live mutations.

Anticipated objection — “What is not in this PoC?” Answer: “Trial-court docket
materials and RECAP documents are absent; coverage is opinions-only and
Cuyahoga-only; judge prediction is not grounded; Q&A paths are association-
only and use the accepted kernel's single embedding lane pending the recorded
fused-panel repair.”
