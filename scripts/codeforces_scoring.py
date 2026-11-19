"""Run Evalchemy's native Codeforces tests without forking scoring threads."""

import multiprocessing as mp
import sys
import time


class TestReporter:
    def __init__(self, connection):
        self.connection = connection

    def append(self, value):
        self.connection.send(("test", value))


def run_tests(tests, completion, is_extracted, connection):
    from eval.chat_benchmarks.CodeForces.codeforces_utils import run_tests_for_one_example

    with connection:
        try:
            run_tests_for_one_example(tests, completion, TestReporter(connection), is_extracted)
        except Exception as error:
            connection.send(("error", str(error)))
        finally:
            connection.send(("done", None))


def run_submission(problem, completion, timeout, is_extracted):
    tests = [{**test, "testtype": "stdin"} for test in problem["official_tests"]]
    context = mp.get_context("forkserver")
    reader, writer = context.Pipe(duplex=False)
    child = context.Process(target=run_tests, args=(tests, completion, is_extracted, writer))
    results = []
    try:
        child.start()
        writer.close()
        deadline = time.monotonic() + (timeout + 1) * len(tests) + 60
        while reader.poll(max(0, deadline - time.monotonic())):
            try:
                kind, value = reader.recv()
            except EOFError:
                raise RuntimeError(
                    "Codeforces worker closed its pipe before reporting completion"
                ) from None
            if kind == "done":
                break
            if kind == "error":
                raise RuntimeError(f"Codeforces worker failed: {value}")
            results.append(value)
    finally:
        writer.close()
        reader.close()
        if child.pid is not None:
            child.join(timeout=1)
            if child.is_alive():
                child.kill()
                child.join()
        child.close()

    # Match Evalchemy's unsuccessful verdict for tests the worker did not finish.
    incomplete = (False, "Time out!.", "Error: Time out!", float("inf"))
    results.extend([incomplete] * (len(tests) - len(results)))
    return results


def score_codeforces(task, responses):
    context = mp.get_context("forkserver")
    context.set_forkserver_preload(["eval.chat_benchmarks.CodeForces.codeforces_utils"])
    # Initialize the forkserver in the main thread before Evalchemy starts its pool.
    server_start = context.Process()
    server_start.start()
    server_start.join()
    exitcode = server_start.exitcode
    server_start.close()
    if exitcode != 0:
        raise RuntimeError(f"Codeforces forkserver initialization failed (exit {exitcode})")

    module = sys.modules[task.__class__.__module__]
    original = module.codeforces_run
    module.codeforces_run = run_submission
    try:
        scored = task.evaluate_responses(responses)
    finally:
        module.codeforces_run = original
    errors = [
        row["reason"]
        for row in scored["examples"]
        if (row.get("reason") or "").startswith(
            ("Evaluation error:", "Critical error:", "Future error:")
        )
    ]
    if errors:
        raise RuntimeError(f"Codeforces infrastructure errors in {len(errors)} examples: {errors[0]}")
    return scored
