# SQL conventions

- State the intended output grain before writing a complex query.
- Document join keys and expected cardinality.
- Check for duplicate amplification after joins.
- Make null handling and timezone assumptions explicit.
- Specify the target SQL dialect.
- Avoid SELECT * in production transformations.