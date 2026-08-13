<claude-mem-context>
# Memory Context

# [snow-grass] recent context, 2026-08-12 2:34pm GMT+8

No previous sessions found.
</claude-mem-context>

## Time handling contract

- Persist timestamps in UTC. API datetime values must always include an explicit UTC offset
  (`Z` or `+00:00`); never emit timezone-naive ISO datetime strings.
- SQLite drops `tzinfo` on read, so restore UTC at the persistence boundary before values reach
  services or response schemas.
- Frontends parse the offset-aware API value and only convert to the user's local timezone for
  display. Never fix timezone bugs by adding a hard-coded number of hours.
