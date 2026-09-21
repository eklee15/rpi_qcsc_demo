"""Utilities."""

import asyncio


async def run_command(command: str, *args) -> str:
    """
    Asynchronously run shell command and return stdout.

    Args:
        command: Shell command to execute.
        args: Arguments of the command.

    Raises:
        RuntimeError: When the return code is nonzero.

    Returns:
        Stdout from the command.
    """
    proc = await asyncio.create_subprocess_exec(
        command,
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        text=False,
    )
    stdout, stderr = await proc.communicate()

    if proc.returncode != 0:
        raise RuntimeError(f"Command {command} failed: {stderr.decode('utf-8').strip()}")

    return stdout.decode("utf-8", errors="replace")
