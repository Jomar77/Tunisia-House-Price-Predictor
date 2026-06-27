# PLAN.md

<!-- You write this before each /yolo run. -->
<!-- Claude works through unchecked [ ] items top to bottom, applies TDD, -->
<!-- checks them off, and updates STATUS.md as it goes. -->
<!-- Delete these examples and write your own. Keep tasks small and scoped. -->

## Tasks for tonight

- [ ] Example: Fix pagination bug when filters are applied
  - Tests first: write a failing test showing pagination breaks with an active filter
  - Then implement the fix
  - Acceptance: full test suite passes, pagination + filter work together

- [ ] Example: Add OAuth2 login endpoint
  - Tests first: test the auth flow (login → token issued → token verified)
  - Then implement OAuth logic with secure httpOnly cookie storage
  - Acceptance: full test suite passes, integration tests green

- [ ] Example: Add email verification to signup
  - Tests first: test that a verification email is sent and the link verifies the account
  - Then implement
  - Acceptance: full test suite passes

<!-- Tips for good tasks:
- One self-contained deliverable per item (a function, an endpoint, a fix)
- Always include an explicit "Acceptance:" line — this is Claude's done signal
- 3-5 items is plenty for one overnight session
- If a task is fuzzy, split it until each piece fits in one sentence
-->
