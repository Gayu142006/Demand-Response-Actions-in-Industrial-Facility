# Stakeholder Validation

**Status: NOT PERFORMED.** This prototype was built and tested in a single
automated session with no access to real facility managers, occupants, or
energy-market stakeholders. Anything claiming real stakeholder sign-off
would be fabricated, so none is claimed.

What *is* included below is the interview/validation protocol the spec
calls for, ready to run with real people, plus the questions a facility
manager or occupant advocate should be asked before this moves past
prototype status.

## Facility manager review checklist (to be run with a real person)

1. Walk through one full day's Dashboard → Planner → Approvals flow for a
   real peak event. Does the recommended action list match what they would
   actually do?
2. Show a NO FEASIBLE PLAN case. Is the explanation and the suggested
   next-steps list (relax target / extend window / override / schedule
   later) actually useful, or does it need to be more specific to their
   equipment?
3. Review the override audit log. Is "authorized_user + role + reason +
   affected_constraint + duration" enough detail for their compliance
   process, or do they need more?
4. Ask directly: would they trust this system's recommendation enough to
   approve it in under 30 seconds during a real peak event, or does it need
   more explanation per action?

## Occupant/zone-owner review checklist (to be run with a real person)

1. Show the opt-out flow. Is TEMPORARY_OPT_OUT easy to set before a known
   sensitive period (e.g. a client visit, a sensitive production run)?
2. Ask: does the system ever feel like it's tracking *them* individually,
   or does the zone-level/band-level framing feel appropriately anonymous?
3. Review `docs/privacy.md` with them directly and ask what's missing.

## Why this matters

The spec's own evaluation framework (peak reduction %, comfort risk,
occupant trust) can only be partially measured by code -- comfort *risk* is
computable from the constraint model, but comfort *trust* and
*acceptability* are not, and shouldn't be simulated as if they were. This
file exists so that gap is visible rather than papered over with invented
survey results.
