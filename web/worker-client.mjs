export class WorkerClient {
  constructor(createWorker) {
    this.createWorker = createWorker;
    this.worker = null;
    this.pending = null;
  }

  run(payload, progress) {
    if (this.pending) return Promise.reject(Error("Another model request is running."));
    return new Promise((resolve, reject) => {
      this.pending = { resolve, reject };
      try {
        const worker = this.worker ??= this.createWorker();
        worker.onmessage = ({ data }) => {
          if (worker !== this.worker || !this.pending) return;
          if (data.type === "progress") progress(data.message);
          if (data.type === "error") this.reset(Error(data.message));
          if (data.type === "result") {
            this.pending = null;
            resolve(data.result);
          }
        };
        worker.onerror = event => {
          event.preventDefault();
          if (worker === this.worker) this.reset(Error(event.message || "Model worker failed to start."));
        };
        worker.onmessageerror = () => {
          if (worker === this.worker) this.reset(Error("Unable to read the model response."));
        };
        worker.postMessage(payload);
      } catch (error) {
        this.reset(error);
      }
    });
  }

  reset(error) {
    this.worker?.terminate();
    this.worker = null;
    this.pending?.reject(error);
    this.pending = null;
  }

  cancel() {
    this.reset(new DOMException("Canceled. You can start another search.", "AbortError"));
  }
}
