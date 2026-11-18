import test from "node:test";
import assert from "node:assert/strict";
import { WorkerClient } from "../worker-client.mjs";

test("cancel and worker errors release the worker and allow a fresh retry", async () => {
  const workers = [];
  const client = new WorkerClient(() => {
    const worker = { terminate() { this.stopped = true; }, postMessage() {} };
    workers.push(worker);
    return worker;
  });
  const first = client.run({}, () => {});
  client.cancel();
  await assert.rejects(first, /anceled/);
  assert.equal(workers[0].stopped, true);
  const second = client.run({}, () => {});
  // A queued event from the terminated worker must not reset the replacement.
  workers[0].onmessageerror();
  assert.equal(workers[1].stopped, undefined);
  workers[1].onerror({ message: "Runtime unavailable", preventDefault() {} });
  await assert.rejects(second, /Runtime unavailable/);
  assert.equal(workers[1].stopped, true);
  const third = client.run({}, () => {});
  workers[2].onmessage({ data: { type: "result", result: "fresh" } });
  assert.equal(await third, "fresh");
});
