# Privacy Notes

This prototype follows the occupant-privacy principles implied by the spec
(zone-level, aggregated signals only):

1. **No individual identity is ever collected or stored.** Field
   observations (`app/database.py: field_observations`) capture
   `occupancy_band` (LOW/MEDIUM/HIGH) per zone, never a headcount, badge ID,
   name, or device identifier.
2. **Occupancy is a band, not a count.** The synthetic dataset and the field
   capture form both express occupancy as a 3-level band, deliberately
   coarse enough that it cannot be reverse-engineered into an individual's
   presence/absence.
3. **Opt-out preferences are never used for individual scoring.** The
   `opt_outs` table stores a zone-level preference (`PARTICIPATE`,
   `PARTICIPATE_ONLY_IF_NECESSARY`, `TEMPORARY_OPT_OUT`) with a reason and
   validity window -- there is no link from an opt-out record to any
   individual's identity or performance record, and the schema has no
   foreign key that could create one.
4. **Overrides are attributed to a role and username for audit purposes
   only** (accountability for safety-relevant decisions), not to build a
   behavioral profile -- the `overrides` table only supports "who approved
   this and why", not "how often does person X override".

## What this prototype does *not* implement (and would need to, in production)

- No authentication/authorization system -- `authorized_user` in the
  overrides table is a free-text field, not tied to a real identity
  provider. A production deployment needs real RBAC.
- No data-retention policy or deletion workflow.
- No encryption-at-rest configuration for the SQLite file (fine for a local
  prototype; not fine for a facility deployment holding real occupancy
  data).
- No consent-management UI beyond the opt-out preference itself.

These are flagged explicitly in `docs/scope_and_limitations.md` rather than
silently assumed away.
