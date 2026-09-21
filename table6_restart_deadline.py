"""Conservative, single-node Slurm allocation budgets; no submission operations.

Compute one budget in the batch shell, before starting ranks.  The budget uses
TimeLimit - RunTime from ONE scontrol response, less the complete query duration
and at least 300 seconds of uncertainty margin.  EndTime is diagnostic only.
Both exported deadlines already include that margin; a worker may retain its
additional shutdown/new-fit guards.  Never recompute a fresh budget per rank.

EnvironmentBudget prefers the shared node's monotonic deadline.  Its wall-clock
deadline is exported for compatibility with the immutable old short worker.
Monotonic timestamps may only be transferred between processes on the same node.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import datetime as dt
import json
import math
import os
import re
import shlex
import socket
import subprocess
import sys
import time
from typing import Callable, Mapping


class DeadlineError(ValueError):
    """Unverifiable or insufficient allocation budget; do not start work."""


def finite_number(value, label: str) -> float:
    try:
        number = float(value)
    except (ValueError, TypeError) as exc:
        raise DeadlineError(f"{label} is not numeric: {value!r}") from exc
    if not math.isfinite(number):
        raise DeadlineError(f"{label} is not finite: {value!r}")
    return number


def parse_duration(value: str) -> int:
    """Parse Slurm [D-]HH:MM:SS (also MM:SS); never guess raw-number units."""
    if not isinstance(value, str):
        raise DeadlineError(f"invalid Slurm duration: {value!r}")
    match = re.fullmatch(r"(?:(\d+)-)?(\d+):(\d{2})(?::(\d{2}))?", value)
    if match is None:
        raise DeadlineError(f"invalid or unbounded Slurm duration: {value!r}")
    days, first, second, third = match.groups()
    if third is None:
        if days is not None:
            raise DeadlineError("day-form duration requires HH:MM:SS")
        minutes, seconds = int(first), int(second)
        hours = 0
    else:
        hours, minutes, seconds = int(first), int(second), int(third)
        if minutes >= 60 or (days is not None and hours >= 24):
            raise DeadlineError(f"out-of-range Slurm duration: {value!r}")
    if seconds >= 60:
        raise DeadlineError(f"out-of-range Slurm duration: {value!r}")
    return int(days or 0) * 86400 + hours * 3600 + minutes * 60 + seconds


def parse_fields(raw: str) -> dict[str, str]:
    """Reject repeated identities instead of accidentally combining job rows."""
    fields: dict[str, str] = {}
    for key, value in re.findall(r"(?:^|\s)([^\s=]+)=([^\s]+)", raw):
        if key in fields:
            raise DeadlineError(f"duplicate scontrol field: {key}")
        fields[key] = value
    return fields


def parse_end_time(value: str) -> float:
    """scontrol is queried with TZ=UTC; naive diagnostics therefore mean UTC."""
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError, AttributeError) as exc:
        raise DeadlineError(f"invalid Slurm EndTime: {value!r}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return finite_number(parsed.timestamp(), "Slurm EndTime epoch")


@dataclass(frozen=True)
class AllocationDeadline:
    job_id: str
    job_state: str
    time_limit_seconds: int
    runtime_seconds: int
    query_elapsed_seconds: float
    safety_margin_seconds: float
    minimum_remaining_seconds: float
    safe_remaining_seconds: float
    observed_local_epoch: float
    observed_monotonic: float
    local_epoch_deadline: float
    monotonic_deadline: float
    monotonic_host: str
    slurm_end_time: str
    slurm_end_epoch: float
    slurm_end_minus_local_epoch_seconds: float
    slurm_end_vs_runtime_projection_seconds: float
    schema: str = "table6_restart_deadline_v1"
    budget_source: str = "single_response_TimeLimit_minus_RunTime"

    def environment(self) -> dict[str, str]:
        return {
            "JOB_BUDGET_END_EPOCH": format(self.local_epoch_deadline, ".9f"),
            "JOB_BUDGET_END_MONOTONIC": format(self.monotonic_deadline, ".9f"),
            "JOB_BUDGET_MONOTONIC_HOST": self.monotonic_host,
            "JOB_BUDGET_JOB_ID": self.job_id,
        }

    def record(self) -> dict:
        return {**asdict(self), "environment": self.environment()}


def derive_deadline(
    raw: str, *, job_id: str, query_started_monotonic: float,
    query_finished_monotonic: float, observed_local_epoch: float,
    expected_limit_seconds: int = 7200, safety_margin_seconds: float = 300,
    minimum_remaining_seconds: float = 120, hostname: str | None = None,
) -> AllocationDeadline:
    """Pure derivation; resources like NumTasks/CPUs/Task are intentionally ignored."""
    job_id = str(job_id)
    if not re.fullmatch(r"[0-9]+", job_id):
        raise DeadlineError(f"invalid allocation job id: {job_id!r}")
    expected = finite_number(expected_limit_seconds, "expected limit")
    if expected <= 0 or expected != int(expected):
        raise DeadlineError("expected limit must be positive whole seconds")
    margin = finite_number(safety_margin_seconds, "safety margin")
    minimum = finite_number(minimum_remaining_seconds, "minimum remaining")
    if margin < 300 or minimum < 0:
        raise DeadlineError("safety margin must be >=300 seconds and minimum >=0")
    started = finite_number(query_started_monotonic, "query start")
    finished = finite_number(query_finished_monotonic, "query finish")
    now = finite_number(observed_local_epoch, "local epoch")
    elapsed = finished - started
    if started < 0 or elapsed < 0:
        raise DeadlineError("invalid monotonic query interval")
    fields = parse_fields(raw)
    for key in ("JobId", "JobState", "TimeLimit", "RunTime", "EndTime"):
        if key not in fields:
            raise DeadlineError(f"missing scontrol field: {key}")
    if fields["JobId"] != job_id:
        raise DeadlineError("scontrol allocation job id mismatch")
    if fields["JobState"] != "RUNNING":
        raise DeadlineError(f"allocation is not RUNNING: {fields['JobState']}")
    limit = parse_duration(fields["TimeLimit"])
    runtime = parse_duration(fields["RunTime"])
    if limit != expected:
        raise DeadlineError(f"TimeLimit contract mismatch: expected {int(expected)}, got {limit}")
    if runtime >= limit:
        raise DeadlineError("allocation expired: RunTime >= TimeLimit")
    end = parse_end_time(fields["EndTime"])
    remaining = limit - runtime - elapsed - margin
    if remaining <= minimum:
        raise DeadlineError(f"insufficient safe remaining budget: {remaining:.3f} seconds")
    host = socket.gethostname() if hostname is None else hostname
    if not host or any(character.isspace() for character in host):
        raise DeadlineError("invalid monotonic clock host")
    return AllocationDeadline(
        job_id=job_id, job_state=fields["JobState"], time_limit_seconds=limit,
        runtime_seconds=runtime, query_elapsed_seconds=elapsed,
        safety_margin_seconds=margin, minimum_remaining_seconds=minimum,
        safe_remaining_seconds=remaining, observed_local_epoch=now,
        observed_monotonic=finished, local_epoch_deadline=now + remaining,
        monotonic_deadline=finished + remaining, monotonic_host=host,
        slurm_end_time=fields["EndTime"], slurm_end_epoch=end,
        slurm_end_minus_local_epoch_seconds=end - now,
        slurm_end_vs_runtime_projection_seconds=end - now - (limit - runtime),
    )


def query_deadline(
    job_id: str | None = None, *, expected_limit_seconds: int = 7200,
    safety_margin_seconds: float = 300, minimum_remaining_seconds: float = 120,
    query_timeout_seconds: float = 30, run: Callable = subprocess.run,
    wall_clock: Callable = time.time, monotonic_clock: Callable = time.monotonic,
) -> AllocationDeadline:
    job_id = str(job_id if job_id is not None else os.environ.get("SLURM_JOB_ID", ""))
    if not re.fullmatch(r"[0-9]+", job_id):
        raise DeadlineError("a numeric SLURM_JOB_ID/--job-id is required")
    timeout = finite_number(query_timeout_seconds, "query timeout")
    if timeout <= 0:
        raise DeadlineError("query timeout must be positive")
    # Do not inherit a custom display format or local timezone for diagnostics.
    env = {**os.environ, "TZ": "UTC"}
    env.pop("SLURM_TIME_FORMAT", None)
    started = monotonic_clock()
    try:
        completed = run(["scontrol", "show", "job", "-o", job_id],
                        text=True, capture_output=True, check=False,
                        timeout=timeout, env=env)
    except (OSError, subprocess.SubprocessError) as exc:
        raise DeadlineError(f"scontrol query failed: {exc}") from exc
    now = wall_clock()
    finished = monotonic_clock()
    if completed.returncode != 0:
        raise DeadlineError(f"scontrol failed ({completed.returncode}): {completed.stderr.strip()}")
    return derive_deadline(
        completed.stdout, job_id=job_id, query_started_monotonic=started,
        query_finished_monotonic=finished, observed_local_epoch=now,
        expected_limit_seconds=expected_limit_seconds,
        safety_margin_seconds=safety_margin_seconds,
        minimum_remaining_seconds=minimum_remaining_seconds,
    )


@dataclass(frozen=True)
class EnvironmentBudget:
    """Use once per worker; late ranks inherit a shrinking, never renewed budget."""
    hard_end_epoch: float
    monotonic_end: float | None
    wall_clock: Callable = time.time
    monotonic_clock: Callable = time.monotonic

    @classmethod
    def from_environment(cls, environ: Mapping[str, str] | None = None, *,
                         hostname: str | None = None, wall_clock: Callable = time.time,
                         monotonic_clock: Callable = time.monotonic):
        env = os.environ if environ is None else environ
        if "JOB_BUDGET_END_EPOCH" not in env:
            raise DeadlineError("missing JOB_BUDGET_END_EPOCH; no fresh-duration fallback")
        epoch = finite_number(env["JOB_BUDGET_END_EPOCH"], "local epoch deadline")
        monotonic_end = None
        if "JOB_BUDGET_END_MONOTONIC" in env:
            host = socket.gethostname() if hostname is None else hostname
            if env.get("JOB_BUDGET_MONOTONIC_HOST") != host:
                raise DeadlineError("monotonic deadline belongs to a different or missing host")
            monotonic_end = finite_number(env["JOB_BUDGET_END_MONOTONIC"], "monotonic deadline")
        if env.get("JOB_BUDGET_JOB_ID") and env.get("SLURM_JOB_ID"):
            if env["JOB_BUDGET_JOB_ID"] != env["SLURM_JOB_ID"]:
                raise DeadlineError("budget belongs to another Slurm allocation")
        return cls(epoch, monotonic_end, wall_clock, monotonic_clock)

    def remaining(self) -> float:
        if self.monotonic_end is not None:
            return self.monotonic_end - self.monotonic_clock()
        return self.hard_end_epoch - self.wall_clock()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id")
    parser.add_argument("--expected-limit-seconds", type=int, default=7200)
    parser.add_argument("--safety-margin-seconds", type=float, default=300)
    parser.add_argument("--minimum-remaining-seconds", type=float, default=120)
    parser.add_argument("--format", choices=("json", "exports"), default="json")
    args = parser.parse_args(argv)
    try:
        result = query_deadline(args.job_id, expected_limit_seconds=args.expected_limit_seconds,
                                safety_margin_seconds=args.safety_margin_seconds,
                                minimum_remaining_seconds=args.minimum_remaining_seconds)
    except DeadlineError as exc:
        print(json.dumps({"event": "allocation_deadline_rejected", "error": str(exc)}), file=sys.stderr)
        return 2
    record = json.dumps(result.record(), sort_keys=True, allow_nan=False)
    if args.format == "json":
        print(record)
    else:
        print(record, file=sys.stderr)
        for key, value in result.environment().items():
            print(f"export {key}={shlex.quote(value)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
