# Planning conventions (`plan/`)

I prefer a Product Manager style plan: concise, easy to scan, and kept up to date as work lands.

1. **Header table:** status, revised date, competition, metric, deadline, working notebook.
2. **WHY (objective):** the problem, goals, principles, and what success looks like.
3. **WHAT (requirements):** a checkbox TODO list grouped by phase.
   - Tick `- [x]` only when the item is done **and** its check has actually run.
   - When ticking, append the result in one line (e.g. "→ 0.94825, keep").
   - Record changes of approach on the item itself (what changed and why).
   - Include an "Out of scope (for now)" list.
4. **HOW (execution):** how each requirement is solved: data splits, workflow, config, design, risks, and a **decisions log** (D1, D2, …) that records every decision I make.

Rules:
- Update the plan *before* executing a new request, then tick items as they land.
- Keep diagrams few (one or two that carry real information); prefer tables and short bullets.
- Open questions go at the end with a recommendation for each.
