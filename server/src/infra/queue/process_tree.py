from __future__ import annotations

from contextlib import suppress

import psutil


def terminate_pid_tree(
    pid: int,
    *,
    expected_create_time: float | None = None,
    timeout_sec: float,
) -> None:
    processes = get_process_tree(
        pid,
        expected_create_time=expected_create_time,
    )

    for process in processes:
        with suppress(psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            process.terminate()

    _, alive = psutil.wait_procs(processes, timeout=timeout_sec)

    for process in alive:
        with suppress(psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            process.kill()


def kill_pid_tree(
    pid: int,
    *,
    expected_create_time: float | None = None,
) -> None:
    processes = get_process_tree(
        pid,
        expected_create_time=expected_create_time,
    )

    for process in processes:
        with suppress(psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            process.kill()


def get_process_tree_rss_bytes(
    pid: int,
    *,
    expected_create_time: float,
) -> int:
    total = 0

    for process in get_process_tree(
        pid,
        expected_create_time=expected_create_time,
    ):
        with suppress(psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            total += process.memory_info().rss

    return total


def prime_process_tree_cpu_percent(
    pid: int,
    *,
    expected_create_time: float,
) -> None:
    for process in get_process_tree(
        pid,
        expected_create_time=expected_create_time,
    ):
        with suppress(psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            process.cpu_percent(interval=None)


def get_process_tree_cpu_percent_of_system(
    pid: int,
    *,
    expected_create_time: float,
) -> float:
    cpu_count = psutil.cpu_count(logical=True) or 1
    raw_percent = 0.0

    for process in get_process_tree(
        pid,
        expected_create_time=expected_create_time,
    ):
        with suppress(psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            raw_percent += process.cpu_percent(interval=None)

    return max(0.0, raw_percent / cpu_count)


def get_process_tree(
    pid: int,
    *,
    expected_create_time: float | None,
) -> list[psutil.Process]:
    root = get_root_process(
        pid,
        expected_create_time=expected_create_time,
    )

    if root is None:
        return []

    with suppress(psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return root.children(recursive=True) + [root]

    return [root]


def get_root_process(
    pid: int,
    *,
    expected_create_time: float | None,
) -> psutil.Process | None:
    try:
        process = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return None

    if expected_create_time is None:
        return process

    with suppress(psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        actual_create_time = process.create_time()

        if abs(actual_create_time - expected_create_time) <= 1.0:
            return process

    return None
