"""
neurosync/app.py — Main Gradio entry point
Run: python app.py  →  http://localhost:7860
"""

import numpy as np
import gradio as gr
import plotly.graph_objects as go
import time, random, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.eeg.generator import generate_eeg_epoch, generate_dataset, EEGConfig
from src.eeg.features import extract_features_batch, compute_band_power, get_band_power_summary
from src.cognitive.model import CognitiveStateDetector
from src.adaptation.ai_adapter import AIAdapter

config = EEGConfig()
detector = CognitiveStateDetector(input_dim=40)
adapter = AIAdapter()
session_log = []
MODEL_TRAINED = False

DEMO_ANSWERS = {
    "focus": [
        "Transformer attention: O(n²) complexity. For long sequences use Flash Attention (tiled SRAM ops, O(n) memory) or linear attention variants. KV-cache at inference collapses repeated prefixes.",
        "RAG pipeline: embed query → ANN search in vector DB → top-k chunk retrieval → prepend to context window → generate. Latency bottleneck is usually embedding, not retrieval. Hybrid BM25+dense usually beats dense-only.",
        "LoRA injects low-rank decomposition ΔW = AB (rank r) into attention projections. Typical r=8–64. Merge at inference for zero overhead. QLoRA adds 4-bit quantization of frozen weights.",
    ],
    "high_effort": [
        "Let me break this down step by step.\n\n**What is attention?** Think of it like a spotlight — the model decides which words to 'focus on' when generating each word.\n\n**Why O(n²)?** Every token attends to every other token. 100 tokens = 10,000 pairs. 1,000 tokens = 1,000,000 pairs. That's the scaling problem.\n\n**How do we fix it?** Flash Attention processes chunks in fast SRAM instead of slow HBM — same result, much less memory.\n\nDoes this make sense so far?",
        "Great question! Let's go step by step.\n\n**Step 1:** You ask a question.\n**Step 2:** Your question gets converted to a vector (a list of numbers).\n**Step 3:** We search a database for documents with similar vectors.\n**Step 4:** The AI reads those documents AND your question to answer.\n\nAnalogy: It's like an open-book exam. The AI doesn't need to memorize everything — it can look things up.\n\nWant me to dig deeper into any step?",
        "LoRA is a way to fine-tune huge AI models cheaply. Here's the analogy:\n\nImagine you have a massive textbook (the AI). Instead of reprinting the whole thing, you attach sticky notes (the LoRA updates).\n\nThe 'rank' controls how detailed your sticky notes are. Rank 8 = short note (fewer parameters, trains faster). Rank 64 = more detailed (captures more nuance).\n\nTake-away: LoRA makes fine-tuning 10–100× cheaper with minimal quality loss.",
    ],
}


def train_model_fn():
    global MODEL_TRAINED, detector
    X_raw, y = generate_dataset(n_epochs_per_class=150, config=config, seed=42)
    X_features = extract_features_batch(X_raw)
    detector = CognitiveStateDetector(input_dim=X_features.shape[1])
    losses = detector.fit(X_features, y, epochs=60, verbose=False)
    MODEL_TRAINED = True
    session_log.append(f"[{_ts()}] Model trained — loss: {losses[-1]:.4f} | {len(y)} epochs | {X_features.shape[1]}d features")
    return f"✓ Trained (loss: {losses[-1]:.4f}) | Dataset: {len(y)} EEG epochs | Features: {X_features.shape[1]}d"


def run_loop_fn(user_query, simulated_state, noise_level):
    global session_log
    if not MODEL_TRAINED:
        return "⚠️ Train the model first.", None, None, "Not trained.", "", _fmt_log()
    if not user_query.strip():
        user_query = random.choice(["How does attention work in transformers?", "Explain RAG.", "What is LoRA?"])

    eeg = generate_eeg_epoch(state=simulated_state, config=EEGConfig(noise_level=noise_level))
    features = compute_band_power(eeg)
    state, confidence, prob_dict = detector.predict(features)
    band_summary = get_band_power_summary(eeg)

    adapter.update_state(state, confidence, band_summary)
    base = random.choice(DEMO_ANSWERS[state])
    response = adapter.adapt_response(base)
    sys_prompt = adapter.get_system_prompt()
    summary = adapter.get_adaptation_summary()

    session_log.append(f"[{_ts()}] {state.upper()} (conf:{confidence:.2f}) | {user_query[:45]}...")

    state_md = (
        f"**Detected:** {state.replace('_',' ').title()} | **Confidence:** {confidence:.1%}\n\n"
        f"**Adaptations:** {summary['n_adaptations']} | **Trend:** {summary['trend'] or 'building…'}\n\n"
        f"**Mode:** verbosity={summary['personality']['verbosity']}, step-by-step={summary['personality']['step_by_step']}, tone={summary['personality']['tone']}"
    )
    return response, _eeg_fig(eeg), _band_fig(band_summary, state), state_md, f"*System prompt:* _{sys_prompt}_", _fmt_log()


def _eeg_fig(eeg):
    t = np.linspace(0, 2, eeg.shape[1])
    colors = ["#1D9E75","#378ADD","#7F77DD","#D4537E"]
    fig = go.Figure()
    for ch in range(4):
        fig.add_trace(go.Scatter(x=t, y=eeg[ch]+ch*3, mode="lines",
            line=dict(color=colors[ch], width=1), name=f"CH{ch+1}"))
    fig.update_layout(title="EEG Epoch", xaxis_title="Time (s)", height=260,
        margin=dict(l=40,r=20,t=40,b=40), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    return fig


def _band_fig(band_summary, state):
    bands, vals = list(band_summary.keys()), list(band_summary.values())
    colors = ["#BA7517","#E24B4A","#7F77DD","#1D9E75","#378ADD"] if state=="high_effort" else ["#1D9E75","#378ADD","#7F77DD","#D4537E","#BA7517"]
    fig = go.Figure(go.Bar(x=bands, y=vals, marker_color=colors,
        text=[f"{v:.2f}" for v in vals], textposition="outside"))
    fig.update_layout(title=f"Band Power — {state.replace('_',' ').title()}", yaxis=dict(range=[0,1.2]),
        height=240, margin=dict(l=40,r=20,t=40,b=40), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    return fig


def _ts(): return time.strftime("%H:%M:%S")
def _fmt_log(): return "\n\n".join(f"`{e}`" for e in reversed(session_log[-8:])) if session_log else "_No events yet._"


with gr.Blocks(title="NeuroSync", theme=gr.themes.Soft()) as demo:
    gr.Markdown("## NeuroSync — Closed-Loop BCI Co-Pilot\n*Brain signals → cognitive state → adaptive AI behavior → closed loop*")
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Setup")
            train_btn = gr.Button("1. Train Cognitive Classifier", variant="primary")
            train_out = gr.Textbox(label="Status", lines=2, interactive=False)
            train_btn.click(fn=train_model_fn, outputs=train_out)

            gr.Markdown("### Simulate EEG Input")
            state_sel = gr.Radio(["focus","high_effort"], value="focus", label="Simulated cognitive state")
            noise_sl = gr.Slider(0.05, 0.5, value=0.1, step=0.05, label="Noise level")
            query_box = gr.Textbox(label="Your query", placeholder="e.g. How does transformer attention work?", lines=2)
            run_btn = gr.Button("2. Run Closed-Loop Iteration ↻", variant="primary")

            gr.Markdown("### System State")
            state_disp = gr.Markdown("_Waiting for iteration…_")
            prompt_disp = gr.Markdown("")

        with gr.Column(scale=2):
            gr.Markdown("### Adaptive AI Response")
            ai_resp = gr.Markdown("_Response will appear here._")
            eeg_plot = gr.Plot(label="EEG Signal")
            band_plot = gr.Plot(label="Band Power")
            gr.Markdown("### Session Log")
            log_disp = gr.Markdown("_No events yet._")

    run_btn.click(
        fn=run_loop_fn,
        inputs=[query_box, state_sel, noise_sl],
        outputs=[ai_resp, eeg_plot, band_plot, state_disp, prompt_disp, log_disp]
    )

if __name__ == "__main__":
    demo.launch(share=True, server_name="127.0.0.1", server_port=7860)
