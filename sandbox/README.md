# Isolated execution of candidate source (optional, not exercised here)

Core Ouroboros never runs source proposals. If you need to test untrusted generated Python, use a disposable container/microVM and keep the trusted grader outside it. The following policy is a starting point, not a claim of perfect isolation. Requires a separately provisioned Docker runtime and image; no container was built or run in the Arena environment.

Build the provided minimal Python image, then mount **only a candidate directory containing no secrets**. Replace `YOUR_CANDIDATE_DIR` with its absolute path.

```bash
docker build -t ouroboros-candidate sandbox
timeout 40s docker run --rm \
  --network none --read-only --user 65534:65534 \
  --cap-drop ALL --security-opt no-new-privileges \
  --pids-limit 64 --memory 512m --cpus 1 \
  --ulimit cpu=30:30 --ulimit fsize=8192:8192 \
  --tmpfs /tmp:rw,noexec,nosuid,nodev,size=32m \
  --mount type=bind,src=YOUR_CANDIDATE_DIR,dst=/candidate,readonly \
  ouroboros-candidate /candidate/test_candidate.py
```

Do not mount the repository root, evaluation answer files, home directory, credentials, SSH agent or Docker socket. Do not expose a model API with privileged tools to the candidate. Enforce a hard outer timeout, review the patch and its measured effects independently, and dispose of the environment after a trial. For a hostile threat model, prefer a dedicated VM with kernel isolation and appropriate runtime hardening.
