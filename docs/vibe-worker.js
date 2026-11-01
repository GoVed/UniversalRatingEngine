/**
 * vibe-worker.js - In-Browser WebWorker for Neural Query Embeddings
 * Uses Transformers.js (Xenova/all-MiniLM-L6-v2) via WebAssembly/WebGPU.
 */

import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2';

// Configure transformers environment
env.allowLocalModels = false;
env.useBrowserCache = true;

let extractor = null;
let isInitializing = false;

async function getExtractor(progressCallback) {
  if (extractor) return extractor;
  if (isInitializing) {
    while (isInitializing) {
      await new Promise(r => setTimeout(r, 100));
    }
    return extractor;
  }

  isInitializing = true;
  try {
    extractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2', {
      quantized: true,
      progress_callback: (p) => {
        if (progressCallback) progressCallback(p);
      }
    });
  } catch (err) {
    console.error('Failed to load Transformers.js pipeline in worker:', err);
    throw err;
  } finally {
    isInitializing = false;
  }
  return extractor;
}

self.onmessage = async (e) => {
  const { type, text, queryId } = e.data;

  if (type === 'init') {
    try {
      self.postMessage({ type: 'status', status: 'loading', message: 'Downloading lightweight AI model (~23MB once)...' });
      await getExtractor((progress) => {
        self.postMessage({ type: 'progress', progress });
      });
      self.postMessage({ type: 'status', status: 'ready', message: 'AI model loaded and ready in local cache.' });
    } catch (err) {
      self.postMessage({ type: 'error', error: err.message });
    }
  } else if (type === 'embed') {
    try {
      const pipe = await getExtractor((progress) => {
        self.postMessage({ type: 'progress', progress });
      });
      
      const t0 = performance.now();
      const output = await pipe(text, { pooling: 'mean', normalize: true });
      const vector = Array.from(output.data);
      const elapsed = Math.round(performance.now() - t0);

      self.postMessage({
        type: 'result',
        queryId,
        text,
        vector,
        elapsed
      });
    } catch (err) {
      self.postMessage({ type: 'error', queryId, error: err.message });
    }
  }
};
