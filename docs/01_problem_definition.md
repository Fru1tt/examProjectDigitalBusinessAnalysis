# 01 Problem Definition

## Business Context

- Organization/industry: Retail sector (multi-channel consumer retail).
- Current situation:
  Retailers need to decide how to distribute resources between online stores, physical stores, and omni-channel solutions, but customer channel preference varies significantly.
- Why this matters now:
  If firms misjudge channel preference, they risk inefficient resource allocation, weaker customer experience, and lower return on marketing/channel investment.

## Problem Statement

- Primary business problem:
  The retailer cannot reliably predict whether individual customers prefer online, in-store, or hybrid shopping.
- Scope (in/out):
  In scope: customer-level preference prediction and driver analysis based on available dataset variables.
  Out of scope: causal proof from randomized experiments, channel cost optimization under real retailer-specific financial constraints.
- Decision(s) this analysis should support:
  Budget/resource allocation across channels, segment-level campaign targeting, and omni-channel development priorities.

## Stakeholders

- Decision owner: Retail management (commercial/channel leadership).
- Users of insights: Marketing teams, channel managers, business analysts.
- Other impacted parties: Store operations, e-commerce operations, CRM/customer experience teams.

## Success Criteria

- Business KPI(s):
  Better channel targeting decisions, improved fit between customer segment and channel strategy, more efficient resource allocation.
- Analytical KPI(s):
  Robust multi-class classification performance (e.g., macro F1, accuracy, per-class recall), interpretable top drivers.
- Minimum acceptable outcome:
  A working and documented model that predicts three channel preferences and produces actionable feature insights for decision-makers.

## Risks and Constraints

- Data constraints:
  Dataset is simulated and not from one specific retailer; external validity is limited.
- Time/resource constraints:
  Semester timeline and group size (2 students) constrain depth of model experimentation and dashboard scope.
- Ethical/legal constraints:
  No direct personal identifiers are expected; still apply good practice in handling demographic variables and avoid discriminatory interpretation.
