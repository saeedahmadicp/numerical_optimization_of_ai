/// <reference lib="webworker" />
import { computeField, type FieldRequest } from './contourField';

self.onmessage = (e: MessageEvent<FieldRequest>) => {
  const result = computeField(e.data);
  (self as unknown as DedicatedWorkerGlobalScope).postMessage(result, [
    result.pixels.buffer,
    result.segments.buffer,
    result.segLevel.buffer,
  ]);
};
