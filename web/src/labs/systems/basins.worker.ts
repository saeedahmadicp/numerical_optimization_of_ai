/// <reference lib="webworker" />
/**
 * Basins-of-attraction worker: the methods and problems register themselves in this worker's own
 * module graph; each request is answered with its labels and iteration counts (transferred).
 */
import '../../methods/roots/systems';
import '../../problems/systems';
import { computeBasins, type BasinRequest } from './basins';

self.onmessage = (e: MessageEvent<BasinRequest>) => {
  const res = computeBasins(e.data);
  (self as unknown as DedicatedWorkerGlobalScope).postMessage(res, [
    res.labels.buffer,
    res.iters.buffer,
  ]);
};
