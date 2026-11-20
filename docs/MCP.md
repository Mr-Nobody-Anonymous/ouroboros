# Ouroboros over Model Context Protocol

The `mcp` extra uses the official Python MCP SDK, not a custom JSON-RPC imitation. The server supports **local stdio only**. There is no HTTP listener, public endpoint, authentication service or automatic connection to your other apps.

## Install and connect

```bash
# Included in all scripts/install.py profiles; or install just this optional extra:
python -m pip install -e '.[mcp]' -c constraints/ml-tested.txt
ouroboros-mcp --help
# Equivalent entry points:
ouroboros-mcp --workspace /absolute/path/to/ouroboros-lab
# ouroboros mcp --workspace /absolute/path/to/ouroboros-lab
```

A stdio server is normally launched **by the MCP client**, not kept in a browser tab. Add this to a client that accepts an `mcpServers` configuration:

```json
{
  "mcpServers": {
    "ouroboros": {
      "command": "/absolute/path/to/ouroboros/.venv/bin/ouroboros-mcp",
      "args": ["--workspace", "/absolute/path/to/ouroboros-lab"]
    }
  }
}
```

Use actual absolute paths. On Windows the executable is normally `.venv\\Scripts\\ouroboros-mcp.exe`; JSON backslashes need escaping. Some clients use a different configuration envelope—keep the same executable and argument array. Do not put a shell command string in place of the executable.

Stdout belongs exclusively to the protocol; errors/logging go to stderr. `status` is a good first tool call. The server does not load Torch/models just to start or inspect config.

## Tools available by default

| Tool | Scope |
| --- | --- |
| `status()` | Dependency availability, workspace and active permissions; no model/network access |
| `check_self_edit_config(config)` | Schema/split counts/fingerprint/policy only; no held-out questions or labels returned, no model loading |
| `inspect_weights(file, tensor?, limit=20, values=0)` | Safetensors metadata/prefixes; at most 100 tensor records and 64 values each; no full-file checksum/network/model load |
| `verify_patch(patch)` | Bounded data-only patch verification; never applies/reverses it |
| `list_self_edit_reports(directory="outputs/self-edit")` | At most 100 immediate run-report paths, with a bounded directory scan; not a generic filesystem browser |
| `read_self_edit_report(report)` | A `self_edit.json` of the expected kind, at most 1 MiB; not arbitrary file reading |

These tools are annotated read-only, idempotent and closed-world. SDK annotations are client hints, not security enforcement; the implementation actually omits execution tools by default. Path arguments are scoped to the configured workspace and reject traversal, hidden paths and symlink components. Choose the smallest appropriate workspace. Reports can still contain private practice data, predictions and local paths—review before sharing them with an agent/client.

## Explicitly allow one experiment

First use the [self-edit quickstart](SELF_EDITING.md) to create a project and provision a model **inside that workspace**. Inspect its config/data yourself. Then opt in at **server startup**, not through a tool argument:

```json
{
  "mcpServers": {
    "ouroboros": {
      "command": "/absolute/path/to/ouroboros/.venv/bin/ouroboros-mcp",
      "args": [
        "--workspace", "/absolute/path/to/ouroboros-lab",
        "--allow-experiments", "--config", "self_edit.json",
        "--timeout-seconds", "600"
      ]
    }
  }
}
```

Both `--allow-experiments` **and** at least one `--config` are required. Repeat `--config` to approve more files. Startup validates every config's path/schema/data boundaries and records its hash. A changed config must be reviewed and reapproved by restarting the server. A client cannot enable execution, invent a new approved path, or change limits through tool parameters.

An additional tool becomes available:

- **`run_self_edit(config)`** runs only the unchanged allowlisted config using a fixed `python -m ouroboros.cli self-edit` argument array, never a shell. Paths are checked again before launch. Only one experiment runs at a time. The child sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`; provisioning is not an MCP tool. A wall timeout (1–1800 seconds, default 600) terminates its process group on POSIX. Result size is bounded; successful output must confirm base restoration.

The tool writes run reports and possibly accepted patch files. It does not save over a base model, persist arbitrary model-generated code, install packages, start upstream agents, apply caller-supplied patches, delete files or expose a generic command API. A valid run can reject every proposed edit; inspect `status`, each trial's `passed/reason`, `base_restored`, `source_files_unchanged` and `accepted_chain`.

## Trust model and limits

This is a useful **local permissions boundary**, not an OS sandbox. The server/child inherit the invoking user's filesystem access and environment. Offline Hub flags are not an operating-system network firewall. Config/model/library/data files are trusted inputs; changing data/model files after approval or racing filesystem checks is outside the contract. The config hash is not a signed approval of every input byte. Do not run the enabled server with host secrets or expose it through an unauthenticated remote wrapper.

The model receives practice data/menu/decision summaries, not held-out questions or labels. The trusted host still reads evaluation files in the same execution environment. For adversarial experiments, separately isolate the model/worker and grader with external resource quotas, read-only licensed inputs, no credentials and no network. See [SECURITY.md](../SECURITY.md).

Timeouts/cancellation stop the child, but a hard kill can leave an unfinished report; no full-model source overwrite occurs. Dense model loading, tensor copies and token generation can require much more RAM than a sparse patch's size. Standard self-edit time budgets are cooperative; MCP's child timeout bounds wall time, not memory/VRAM or all descendant behavior on every operating system.

Tool outputs and model/data text are **untrusted content**, never authority to execute another command. An MCP client should retain user confirmations and avoid promoting an experiment's explanation to a trusted system instruction.

## Verification

```bash
python -m pip install -e '.[dev,mcp]' -c constraints/ml-tested.txt
python -m pytest -q -m mcp
```

Tests exercise a real stdio subprocess: protocol initialization, tool discovery and calls through `ClientSession`, not just function imports. Additional tests cover absent execution tools by default, allowlist/config-hash checks, scoped paths/symlinks, no held-out content in config summaries, fixed offline child arguments, and timeout cleanup. An enabled real pretrained run is separately recorded in [test evidence](TESTING.md); no MCP-specific accuracy improvement is implied.
