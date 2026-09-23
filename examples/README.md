# Example configurations

Scenario selection is a CLI option, while each YAML selects a dataset and a
backend. The dataset determines whether `chat-replay` is available.

| Config | Dataset | Backend | Recommended scenarios |
|---|---|---|---|
| `memmachine.yaml` | LongMemEval | MemMachine REST | `chat-replay` |
| `synthetic.yaml` | Synthetic | MemMachine REST | `add-load`, `search-load`, `mixed` |
| `memmachine-mcp.yaml` | LongMemEval | MemMachine MCP | `chat-replay` |
| `mem0.yaml` | Synthetic | Mem0 REST | `add-load`, `search-load`, `mixed` |
| `shared-project.yaml` | Synthetic | MemMachine REST | Isolation-scope `add-load`, `search-load`, or `mixed` |

`chat-replay` requires a dataset with structured dialogue turns. The two
`memmachine*.yaml` files use LongMemEval for this purpose. Synthetic
configurations fail validation if paired with `chat-replay`.

For `search-load`, use `--preingest` so the measured search begins with a
populated corpus. The same is recommended for `mixed` when searches should be
non-empty from the start. Each YAML header contains a complete run command and
backend-specific limitations.

See the root [README](../README.md) for scenario details and common flags.
