from __future__ import annotations

import json
from pathlib import Path
from threading import Lock

import numpy as np
import onnxruntime as ort
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from transformers import AutoConfig, AutoTokenizer

MODEL_DIR = Path(__file__).resolve().parent.parent / "static" / "classifier" / "models" / "finetuned-int8"
ONNX_PATH = MODEL_DIR / "onnx" / "model_quantized.onnx"

_tokenizer = None
_config = None
_session = None
_runtime_lock = Lock()


def index(request):
    return render(request, "classifier/index.html")


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits)
    exps = np.exp(logits)
    return exps / np.sum(exps)


def get_runtime():
    global _tokenizer, _config, _session

    if _tokenizer is None or _config is None or _session is None:
        with _runtime_lock:
            if _tokenizer is None:
                _tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)

            if _config is None:
                _config = AutoConfig.from_pretrained(MODEL_DIR)

            if _session is None:
                sess_options = ort.SessionOptions()

                # tiny machine friendly settings
                sess_options.intra_op_num_threads = 1
                sess_options.inter_op_num_threads = 1
                sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

                _session = ort.InferenceSession(
                    str(ONNX_PATH),
                    sess_options=sess_options,
                    providers=["CPUExecutionProvider"],
                )

    return _tokenizer, _config, _session


@csrf_exempt
@require_POST
def predict_api(request):
    try:
        payload = json.loads(request.body.decode("utf-8"))
    except Exception:
        return JsonResponse({"error": "Invalid JSON body"}, status=400)

    title = str(payload.get("title", "")).strip()
    abstract = str(payload.get("abstract", "")).strip()

    if not title or not abstract:
        return JsonResponse({"error": "Both title and abstract are required"}, status=400)

    try:
        tokenizer, config, session = get_runtime()

        encoded = tokenizer(
            title,
            text_pair=abstract,
            truncation="longest_first",
            max_length=192,
            return_tensors="np",
        )

        # ONNX expects numpy arrays, usually int64
        ort_inputs = {}
        for key, value in encoded.items():
            if isinstance(value, np.ndarray):
                ort_inputs[key] = value.astype(np.int64, copy=False)

        outputs = session.run(None, ort_inputs)
        logits = np.asarray(outputs[0])[0]
        probs = _softmax(logits)

        id2label = getattr(config, "id2label", {}) or {}

        predictions = []
        for idx, prob in enumerate(probs):
            label = id2label.get(str(idx), id2label.get(idx, f"class_{idx}"))
            predictions.append(
                {
                    "label": label,
                    "prob": float(prob),
                }
            )

        predictions.sort(key=lambda x: x["prob"], reverse=True)

        return JsonResponse(
            {
                "mode": "api",
                "predictions": predictions,
            }
        )

    except Exception as e:
        return JsonResponse({"error": f"Prediction failed: {e}"}, status=500)