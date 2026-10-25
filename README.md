# otto
![alt text](image.png)
A terminal agent built on **Qwen3-0.6B**, post-trained with reinforcement learning (GRPO) where the
reward comes from actually running the model's shell commands in a Docker sandbox and checking the
resulting filesystem. You describe a task in plain English; `otto` works out the commands, runs them
on a **throwaway copy** of your directory, shows you exactly what changed, and applies the changes
only if you approve.

## How it works

```
  you type a task
        |
        v
  otto ---- copies your directory to a temp folder
        |
        |  conversation          generated text
        +--------------------->  model server  --------+
        |                        (SGLang on a GPU, or  |
        <----------------------  the local server)  <--+
        |
        v
  runs the model's command in a Docker container on the COPY
  (no network, dropped capabilities, 10 s timeout) -> output goes back to the model -> repeat
        |
        v
  shows the diff:  - deleted   + created   ~ modified      apply these changes? [y/N]
```

The model server only generates text. It never touches your files or runs a command. Everything that
executes happens in Docker on the machine running `otto`. If your directory changes while the agent
works, `otto` refuses to apply rather than overwrite your edits.

## Requirements

| Where | What |
|---|---|
| The machine running `otto` | Docker (running), `rsync`, Python 3.11 |
| A model server (pick one) | **A)** SGLang on a Linux machine with an NVIDIA GPU, e.g. a free Google Colab T4 (steps below), or **B)** the bundled local server, which runs on any machine (CPU, Apple Silicon or CUDA) and is slower |

SGLang only runs on Linux with an NVIDIA GPU, so on a Mac or Windows machine use Colab or another
GPU host for option A.

You also need the trained **LoRA adapter**: the two files `adapter_config.json` and
`adapter_model.safetensors` (about 40 MB). It is produced by training (`src/training`). Below, `ADAPTER_DIR` means the folder that holds them.

## One-time setup

```bash
# 1. Python environment (torch, transformers, peft, rich, ...)
./setup/mac_train_env.sh          # on other platforms: create a Python 3.11 venv at .venv-train
                                  # and install the packages that script installs

# 2. Docker sandbox image (Docker must be running)
docker build -t otto-sandbox sandbox/

# 3. Only for the local server (option B): merge the adapter into the base weights (~1.2 GB output)
.venv-train/bin/python -m src.deploy.merge --adapter ADAPTER_DIR --out deploy/otto-0.6b
```

Step 3 downloads `Qwen/Qwen3-0.6B` from Hugging Face the first time. Option A merges on the GPU host
instead, so it can skip step 3.

To try things safely, create a practice folder:

```bash
scripts/make_playground.sh        # creates playground/ (deletes and rebuilds it every time)
```

## Option A: serve with SGLang (example: Google Colab)

1. Open <https://colab.research.google.com>, create a notebook, and set
   **Runtime -> Change runtime type -> T4 GPU**.
2. Paste the **whole** of [`setup/colab_one_cell.py`](setup/colab_one_cell.py) into one cell and run it.
3. When the file picker appears, choose `adapter_config.json` and `adapter_model.safetensors`
   from `ADAPTER_DIR`.
4. Wait 5-15 minutes. The cell prints nothing during the install (Rust and SGLang are being built
   and downloaded). It then prints `installed`, `merged`, `server up`, and finally:
   ```
   SEND THIS URL BACK: https://something-random.trycloudflare.com/v1
   ```
5. Copy that URL. **Keep the Colab tab open**: the URL stops working when the session ends or idles.

If the cell fails it prints the error or the server log. Common causes:

- *Install fails on a CUDA/torch mismatch*: read the error; do not pin versions blindly.
- *`ModuleNotFoundError: torch`*: packages went into the wrong Python. The script installs with
  `uv pip install --python /content/sg/bin/python`, which avoids this; re-run the cell.
- *Server never becomes healthy*: the cell prints the last lines of the server log.

Then run the agent against that URL:

```bash
scripts/otto --url https://something-random.trycloudflare.com/v1 --dir playground
```

On any other Linux GPU machine, skip Colab: run `scripts/serve_sglang.sh deploy/otto-0.6b 30000`
(after merging, step 3 above) and point `--url` at `http://HOST:30000/v1`.

## Option B: serve locally (no GPU needed, slower)

In one terminal, start the local server (a stand-in for SGLang that speaks the same API):

```bash
HF_HUB_OFFLINE=1 .venv-train/bin/python -m src.deploy.local_server      # port 30001
```

In another terminal:

```bash
scripts/otto --local --dir playground
```

## Using it

```bash
scripts/otto --url URL --dir playground                                      # interactive prompt
scripts/otto --url URL --dir playground "delete the empty files in notes/"   # one task
```

Type a task and press enter; an empty line quits. For each task you see the model's short reasoning
(dim `#` lines), each command it runs (`$ ...`) and its output, then the diff and
`apply these changes? [y/N]`. Answer `n` and your files are untouched.

Things to try in `playground/`:

- `move all the .tmp files from logs/ to cache/`
- `delete the empty files in notes/`
- `list every .tmp file, including subdirectories`

Options: `--dir DIR` (the directory the agent may change, default: current), `--max-turns N`
(default 8), `--temperature T` (default 0.6), `--yes` (apply without asking, not recommended).
The `OTTO_URL` environment variable can hold the endpoint instead of `--url`.

## What to expect

- It is a 0.6B model. It often picks a sensible command and sometimes does not: repeating a
  command, working in the wrong directory, or declaring itself done too early. That is why changes are only applied after you approve the diff.
- The model gets at most 256 thinking tokens per turn, then a forced "stop deliberating" sentence, then a 96-token answer. The client (`src/deploy/policy.py`) reproduces exactly that over the API.
- Due to compute constraints and time constraints, the model has been post-trained using RLVR only for a few steps (<100), so the model may make mistakes.

## Project layout

| Path | What it is |
|---|---|
| `src/deploy/agent.py`, `ui.py` | the `otto` CLI and its terminal interface |
| `src/deploy/policy.py` | client that talks to the model server with the thinking budget |
| `src/deploy/merge.py`, `local_server.py` | LoRA merge; the local stand-in server |
| `scripts/otto`, `scripts/serve_sglang.sh` | launcher; SGLang launch for a Linux GPU machine |
| `setup/colab_one_cell.py` | the Colab cell |
| `src/harness`, `src/verifier`, `src/tasks` | sandbox harness, verifier, task templates (training side) |
| `src/training` | GRPO training, evaluation, monitoring |
| `tests/` | `REQUIRE_DOCKER=1 .venv-train/bin/pytest tests -q` |

## Notes

- Commands run in Docker with `--network none`, dropped capabilities, a read-only root filesystem, a
  non-root user and CPU, memory and process limits, on a copy of your directory, never on the original.
- Nothing is copied back without an explicit `y`, and not at all if the directory changed meanwhile.
- `otto` refuses to work on directories over 20,000 files or 200 MB.
- The Colab step uploads only the adapter you choose; no credentials are involved.