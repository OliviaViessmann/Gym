#!/usr/bin/env python3
"""Reveal the difference between OpenSandbox's one-shot exec and a persistent kernel.

This is a self-contained test intended to be shared with the OpenSandbox team. It
uses ONLY the ``opensandbox`` SDK (no nemo_gym, no litmus) and makes the point
concrete with a two-cell example:

    cell 1:  x = 42
    cell 2:  print(x)

Part A runs those two cells on the exec path we currently integrate
(``sandbox.commands.run`` -> POST /command). Each call is a fresh process, so
cell 2 fails with NameError. THIS is the behaviour we call "one-shot": it is not
about sandbox lifetime -- the sandbox stays alive the whole time -- it is about
the Python interpreter *process* not surviving between calls.

Part B runs the SAME two cells on the code-interpreting path already present in
the SDK's generated client (POST /code/context to create a context, then POST
/code to run against that context id). Per OpenSandbox's own docstrings this is
"Jupyter kernel" execution that "maintains execution state within the session".
If the deployed execd exposes it, cell 2 prints 42 and the point is proven; if
not, Part B reports that the endpoint is not available on this deployment.

The distinction we need the team to see:
    sandbox alive for 8h   != persistent Python kernel
    (container/pod uptime)    (one interpreter process holding in-memory state)

Run (needs the ``opensandbox`` SDK; httpx ships with it):

    source opensandbox.env
    python3 opensandbox_persistent_kernel_smoke.py

Env:
    OPENSANDBOX_DOMAIN    ELB hostname, no scheme (required)
    OPENSANDBOX_API_KEY   service API key (required)
    OPENSANDBOX_PROTOCOL  http | https (default http)
    SANDBOX_IMAGE         inner image (default docker.io/library/python:3.12-slim)
"""

import asyncio
import json
import os
import shlex
import sys
from datetime import timedelta

try:
    import httpx
    from opensandbox import Sandbox
    from opensandbox.config import ConnectionConfig
    from opensandbox.constants import DEFAULT_EXECD_PORT
    from opensandbox.models.sandboxes import PlatformSpec
except Exception as exc:  # pragma: no cover
    sys.exit(f"cannot import opensandbox SDK (pip install opensandbox): {exc}")

DOMAIN = os.environ.get("OPENSANDBOX_DOMAIN")
API_KEY = os.environ.get("OPENSANDBOX_API_KEY")
PROTOCOL = os.environ.get("OPENSANDBOX_PROTOCOL", "http")
# The code-interpreter endpoints (/code) only work when the sandbox runs the
# opensandbox/code-interpreter image (ships the language kernels) with its
# entrypoint. A bare python image serves /command but hangs on /code/context.
IMAGE = os.environ.get("SANDBOX_IMAGE", "opensandbox/code-interpreter")
ENTRYPOINT = os.environ.get("SANDBOX_ENTRYPOINT", "/opt/code-interpreter/code-interpreter.sh")

if not DOMAIN or not API_KEY:
    sys.exit("set OPENSANDBOX_DOMAIN and OPENSANDBOX_API_KEY (source opensandbox.env)")

# The two "notebook cells" the whole test hinges on.
CELL_1 = "x = 42"
CELL_2 = "print(x)"


def _hdr(label):
    print(f"\n{'=' * 70}\n{label}\n{'=' * 70}")


def _one_shot_stdout(result):
    """Extract stdout text from a commands.run() result."""
    try:
        return "\n".join(m.text for m in result.logs.stdout)
    except Exception:
        return ""


def _one_shot_stderr(result):
    try:
        return "\n".join(m.text for m in result.logs.stderr)
    except Exception:
        return ""


async def part_a_one_shot(sandbox):
    """Two cells on the /command path -- proves in-memory state does NOT survive."""
    _hdr("PART A  --  one-shot /command path (what we integrate today)")
    print(f"  cell 1: {CELL_1!r}   cell 2: {CELL_2!r}\n")

    # Each cell is its own `python3 -c` invocation == its own process, which is
    # exactly what one tool call maps to on this path.
    r1 = await sandbox.commands.run(f"python3 -c {json.dumps(CELL_1)}")
    print(f"  cell 1 -> exit={r1.exit_code} stdout={_one_shot_stdout(r1)!r} stderr={_one_shot_stderr(r1)!r}")

    r2 = await sandbox.commands.run(f"python3 -c {json.dumps(CELL_2)}")
    out2, err2 = _one_shot_stdout(r2), _one_shot_stderr(r2)
    print(f"  cell 2 -> exit={r2.exit_code} stdout={out2!r} stderr={err2!r}")

    state_lost = out2.strip() != "42"
    if state_lost:
        print("\n  [EXPECTED] cell 2 could NOT see x from cell 1 -> in-memory state did not survive.")
        print("             The sandbox never died; the interpreter process did. This is 'one-shot'.")
    else:
        print("\n  [SURPRISE] cell 2 saw x -- the /command path preserved interpreter state (unexpected).")
    return state_lost


DEBUG_SSE = os.environ.get("DEBUG_SSE") == "1"


def _sse_event(line):
    """Decode one SSE line to a dict, mirroring the SDK's _decode_sse_event_line.

    execd emits either raw JSON lines or `data:`-prefixed lines; framing lines
    (event:/id:/retry:/comments) are ignored.
    """
    if not line.strip():
        return None
    if line.startswith((":", "event:", "id:", "retry:")):
        return None
    data = line[5:].strip() if line.startswith("data:") else line.strip()
    if not data:
        return None
    try:
        return json.loads(data)
    except Exception:
        return None


async def _stream_code(sse_client, url, code, context_id):
    """POST /code and collect (stdout, stderr, results, error) from the SSE stream."""
    stdout, stderr, results, error, status = [], [], [], None, None
    body = {"code": code, "context": {"id": context_id, "language": "python"}}
    async with sse_client.stream("POST", url, json=body) as resp:
        status = resp.status_code
        if status != 200:
            await resp.aread()
            return "", "", [], f"HTTP {status}: {resp.text[:200]}", status
        async for line in resp.aiter_lines():
            if DEBUG_SSE and line.strip():
                print(f"    [sse] {line}")
            evt = _sse_event(line)
            if not evt:
                continue
            etype = evt.get("type")
            text = evt.get("text")
            if etype == "stdout" and text:
                stdout.append(text)
            elif etype == "stderr" and text:
                stderr.append(text)
            elif etype == "result" and evt.get("results"):
                results.append(evt["results"])
            elif etype == "error" and evt.get("error"):
                error = evt["error"]
            # Some builds put print output only in `text` regardless of type.
            elif text and etype not in ("init", "status", "ping", "execution_count", "execution_complete"):
                stdout.append(text)
    return "".join(stdout), "".join(stderr), results, error, status


async def part_b_persistent_context(sandbox):
    """Two cells on the /code + /code/context path -- persistent Jupyter kernel.

    Reuses the SDK command service's already-wired httpx clients (same base_url,
    headers, and connection_config.transport that Part A used successfully), so a
    failure here reflects the deployed execd, not a hand-rolled client mistake.
    """
    _hdr("PART B  --  persistent code-interpreter context (POST /code/context + /code)")
    print(f"  cell 1: {CELL_1!r}   cell 2: {CELL_2!r}\n")

    cs = getattr(sandbox, "_command_service", None)
    if cs is None or not hasattr(cs, "_httpx_client"):
        print("  [SKIP] could not access the SDK command-service transport on this SDK version.")
        return None
    http_client = cs._httpx_client          # base_url set, proxy transport wired
    sse_client = cs._sse_client             # read-timeout disabled, for /code stream
    ctx_url = cs._get_execd_url("/code/context")
    code_url = cs._get_execd_url("/code")

    try:
        # 1) Create a persistent context (returns a session id). Bounded so a
        #    non-existent endpoint reports cleanly instead of hanging.
        resp = await asyncio.wait_for(
            http_client.post(ctx_url, json={"language": "python"}, headers={"Content-Type": "application/json"}),
            timeout=30,
        )
    except asyncio.TimeoutError:
        print("  [NO RESPONSE] POST /code/context did not respond within 30s.")
        print("                execd likely does not serve the code-interpreter API on this deployment.")
        return None

    if resp.status_code in (404, 405, 501):
        print(f"  [NOT DEPLOYED] POST /code/context -> HTTP {resp.status_code}. This execd build does")
        print("                 not expose the code-interpreter API. That is the deployment gap.")
        return None
    resp.raise_for_status()
    ctx = resp.json()
    context_id = ctx.get("id")
    print(f"  created context: id={context_id!r} language={ctx.get('language')!r}")

    # 2) Run cell 1 (defines x) then cell 2 (prints x) against the SAME context.
    out1, serr1, res1, err1, st1 = await _stream_code(sse_client, code_url, CELL_1, context_id)
    print(f"  cell 1 -> http={st1} stdout={out1!r} stderr={serr1!r} results={res1} error={err1}")

    out2, serr2, res2, err2, st2 = await _stream_code(sse_client, code_url, CELL_2, context_id)
    print(f"  cell 2 -> http={st2} stdout={out2!r} stderr={serr2!r} results={res2} error={err2}")

    # Cleanup best-effort.
    try:
        await http_client.request("DELETE", cs._get_execd_url(f"/code/context/{context_id}"))
    except Exception:
        pass

    printed = "42" in out2 or "42" in serr2 or any("42" in str(r) for r in res2)
    if printed:
        print("\n  [PROVEN] cell 2 saw x=42 from cell 1 -> the context is a persistent kernel.")
        print("           This is exactly the capability litmus needs; it lives in the SDK today.")
    else:
        print("\n  [UNEXPECTED] context exists but cell 2 did not see x -- share this output with us.")
    return printed


async def main():
    print(f"endpoint:   {PROTOCOL}://{DOMAIN}")
    print(f"image:      {IMAGE}")
    print(f"entrypoint: {ENTRYPOINT or '(image default)'}")
    # Large request timeout so a slow image *pull* doesn't look like a failure.
    # If create still times out at ~10min, the image is likely not pullable in
    # this deployment's registry (vs merely slow).
    config = ConnectionConfig(
        domain=DOMAIN,
        api_key=API_KEY,
        protocol=PROTOCOL,
        use_server_proxy=True,
        request_timeout=timedelta(seconds=600),
    )

    create_kwargs = dict(
        connection_config=config,
        platform=PlatformSpec(os="linux", arch="amd64"),
        timeout=timedelta(minutes=30),
        skip_health_check=True,
    )
    if ENTRYPOINT:
        create_kwargs["entrypoint"] = shlex.split(ENTRYPOINT)

    print("creating sandbox (allowing up to 10min for image pull)...")
    try:
        sandbox = await Sandbox.create(IMAGE, **create_kwargs)
    except Exception as exc:
        _hdr("RESULT  --  sandbox never came up")
        print(f"  create failed for image {IMAGE!r}: {type(exc).__name__}: {str(exc)[:300]}")
        print("\n  With image 'python:3.12-slim' the sandbox creates fine, so this is specific to")
        print("  the code-interpreter image -- most likely it is not pullable in this deployment's")
        print("  registry. Ask infra/OpenSandbox: is 'opensandbox/code-interpreter' available to execd?")
        sys.exit(2)
    print(f"sandbox created: id={sandbox.id}")

    try:
        state_lost = await part_a_one_shot(sandbox)
        persistent_ok = await part_b_persistent_context(sandbox)
    finally:
        try:
            await sandbox.kill()
        finally:
            await sandbox.close()

    _hdr("SUMMARY")
    print(f"  A. one-shot /command drops in-memory state : {'yes (as expected)' if state_lost else 'no'}")
    if persistent_ok is None:
        print("  B. persistent /code context               : NOT AVAILABLE on this deployment")
        print("\n  => The SDK ships the persistent-kernel API, but this execd does not expose it.")
        print("     Ask: can /code + /code/context be enabled on the deployed service?")
    elif persistent_ok:
        print("  B. persistent /code context               : WORKS (cell 2 saw x=42)")
        print("\n  => The capability is live. Remaining work is ours: wire nemo_gym's provider to")
        print("     create a context + run_code instead of one-shot commands.run.")
    else:
        print("  B. persistent /code context               : context created but state not preserved")


if __name__ == "__main__":
    asyncio.run(main())
