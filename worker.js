// Runs the Kokoro neural voice off the main thread so the page stays responsive.
import { KokoroTTS } from 'https://cdn.jsdelivr.net/npm/kokoro-js@1.2.1/dist/kokoro.web.js';

const MODEL = 'onnx-community/Kokoro-82M-v1.0-ONNX';
let tts = null;
let loading = null;
let epoch = 0;
let chain = Promise.resolve();

async function hasWebGPU() {
  try {
    if (!self.navigator || !self.navigator.gpu) return false;
    const adapter = await self.navigator.gpu.requestAdapter();
    return !!adapter;
  } catch (e) {
    return false;
  }
}

function load() {
  if (loading) return loading;
  loading = (async () => {
    const progress_callback = (p) => {
      if (p && p.status === 'progress' && p.total) {
        postMessage({ type: 'progress', file: p.file, loaded: p.loaded, total: p.total });
      }
    };
    let device = null;
    if (await hasWebGPU()) {
      try {
        tts = await KokoroTTS.from_pretrained(MODEL, { dtype: 'fp32', device: 'webgpu', progress_callback });
        device = 'webgpu';
      } catch (e) {
        tts = null;
      }
    }
    if (!tts) {
      tts = await KokoroTTS.from_pretrained(MODEL, { dtype: 'q8', device: 'wasm', progress_callback });
      device = 'wasm';
    }
    postMessage({ type: 'ready', device });
  })().catch((e) => {
    loading = null;
    postMessage({ type: 'loadError', message: String((e && e.message) || e) });
    throw e;
  });
  return loading;
}

self.onmessage = (ev) => {
  const m = ev.data || {};
  if (m.type === 'load') {
    load().catch(() => {});
  } else if (m.type === 'epoch') {
    epoch = m.epoch;
  } else if (m.type === 'generate') {
    chain = chain.then(async () => {
      if (m.epoch < epoch) return;
      await load();
      if (m.epoch < epoch) return;
      try {
        const out = await tts.generate(m.text, { voice: m.voice, speed: m.speed });
        const pcm = new Float32Array(out.audio);
        postMessage({ type: 'audio', key: m.key, epoch: m.epoch, sr: out.sampling_rate, pcm }, [pcm.buffer]);
      } catch (e) {
        postMessage({ type: 'genError', key: m.key, epoch: m.epoch, message: String((e && e.message) || e) });
      }
    }).catch(() => {});
  }
};
