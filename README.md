> Live demo: run `python app.py` after `pip install -r requirements.txt`
# NeuroSync — Closed-Loop BCI Co-Pilot

> A human-AI co-adaptation system for knowledge work. Brain signals → cognitive state → adaptive AI behavior → closed loop.

## What is this?

NeuroSync demonstrates a **closed-loop brain-computer interface** where:

1. **EEG signals** are read (simulated here, real hardware in production)
2. A **cognitive state classifier** detects whether you're focused or experiencing high mental effort
3. An **AI co-pilot** adapts its communication style in real-time based on your brain state
4. **User responses** to AI output become implicit feedback, enabling the system to improve over time

This creates a **bidirectional symbiotic loop** — the AI shapes your cognitive experience, and your cognitive state shapes the AI.

## Why it matters

Current AI assistants are static. They give the same response regardless of whether you're in flow or overwhelmed. NeuroSync makes the interface *neurally adaptive* — a fundamentally more natural form of human-computer interaction.

**Applications:** enterprise knowledge work, developer tools, education, medical decision support.

## Architecture

```
EEG Signal (256 Hz, 8 channels)
    ↓
Band-Power Feature Extraction (alpha, theta, beta, gamma, delta)
    ↓
Cognitive State Classifier (PyTorch MLP → Focus vs High Effort)
    ↓
AI Adapter (dynamic system prompt + response style)
    ↓
Adapted LLM Response
    ↓
User Reaction → Online Adaptation Signal → Model Update (co-adaptation)
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app
python app.py

# 3. Open browser at http://localhost:7860

# 4. Click "Train Cognitive Classifier", then "Run Closed-Loop Iteration"
```

## Project Structure

```
neurosync/
├── src/
│   ├── eeg/
│   │   ├── generator.py     # Synthetic EEG with realistic band profiles
│   │   └── features.py      # Band-power extraction (Butterworth filter)
│   ├── cognitive/
│   │   └── model.py         # PyTorch MLP classifier + online adaptation
│   └── adaptation/
│       └── ai_adapter.py    # State → AI personality mapping + session memory
├── app.py                   # Gradio UI + closed-loop orchestration
└── requirements.txt
```

## Technical Details

- **EEG simulation:** Physiologically realistic — summed sinusoids at band frequencies + 1/f colored noise
- **Feature extraction:** Log band power per channel (Butterworth bandpass, order 4)
- **Classifier:** 3-layer MLP with BatchNorm + Dropout; trains in ~5s on CPU
- **Co-adaptation:** Online gradient update on single samples with reduced LR to prevent catastrophic forgetting
- **State profiles based on published BCI literature:** alpha suppression under load, theta elevation with working memory demand, gamma increase with active processing

## Roadmap

- [ ] Phase 3: Replace MLP with Spiking Neural Network (snnTorch) for biological realism
- [ ] Phase 3: Real MOABB dataset integration for validation
- [ ] Phase 4: Live LLM integration (OpenAI / Anthropic API)
- [ ] Phase 4: Longitudinal user model that persists across sessions

---

*Built to demonstrate the vision for neurally-adaptive enterprise AI.*
