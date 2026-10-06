# `src/generated/` — output of `numopt export` (never hand-edit)

```
npm run gen        # = ../.venv/bin/numopt export src/generated
```

writes

| file                     | content                                                                                 |
| ------------------------ | --------------------------------------------------------------------------------------- |
| `registry.json`          | every Python `MethodSpec` (id, family, name, params, needs, order, summary, references) |
| `problems.json`          | metadata of every Python problem (`kind`, id, latex, dim, domain, x0, minima, ...)      |
| `fixtures/<family>.json` | parity cases `[{method, problem, params, result}]` with full traces                     |

The app and the tests must work when these files are missing (the Python side may be
incomplete). Load them only through `src/generated/index.ts`, which uses
`import.meta.glob` (lazy) and returns empty data when a file does not exist.
