"""
neurosync/src/adaptation/ai_adapter.py
Adaptive AI response logic — the "closed loop" payoff.

WHY THIS EXISTS:
This module translates cognitive state predictions into AI behavior changes.
It's the bridge between neuroscience (what is the brain doing?) and AI product
design (how should the assistant respond differently?). For investors, this is
the core value proposition: implicit, zero-friction feedback that makes AI
assistants genuinely adaptive.

The adapter maintains a session memory of past states, enabling the system
to detect patterns over time (e.g., repeated effort states → simplify curriculum).
"""

from dataclasses import dataclass, field
from typing import Optional
import time


@dataclass
class AdaptationConfig:
    # How many past states to remember
    history_window: int = 10
    # Threshold for switching modes
    effort_threshold: float = 0.60  # if confidence > this, trigger effort mode


@dataclass
class AIPersonality:
    """
    The AI's communication style, modulated by cognitive state.
    Think of this as a set of "dials" the system turns.
    """
    verbosity: str = "medium"         # "low" | "medium" | "high"
    use_analogies: bool = False
    step_by_step: bool = False
    scaffolding: bool = False
    check_understanding: bool = False
    density: str = "rich"             # "rich" | "balanced" | "sparse"
    tone: str = "direct"              # "direct" | "supportive"


# Personality presets for each detected state
PERSONALITIES = {
    "focus": AIPersonality(
        verbosity="low",
        use_analogies=False,
        step_by_step=False,
        scaffolding=False,
        check_understanding=False,
        density="rich",
        tone="direct",
    ),
    "high_effort": AIPersonality(
        verbosity="high",
        use_analogies=True,
        step_by_step=True,
        scaffolding=True,
        check_understanding=True,
        density="sparse",
        tone="supportive",
    ),
}

# Example AI response templates for each state
RESPONSE_TEMPLATES = {
    "focus": [
        "**[Dense mode]** {content}\n\n_Neural signature: focused. Output optimized for information density._",
        "**[Flow state detected]** {content}\n\n_Scaffolding suppressed. Delivering raw insights._",
        "{content}\n\n_Alpha-dominant state. Minimal overhead enabled._",
    ],
    "high_effort": [
        "**[Guided mode]** Let me break this down step by step.\n\n{content}\n\n_Does this make sense so far? I've detected elevated cognitive load._",
        "**[Scaffolded response]** Here's an analogy first, then the detail.\n\n{content}\n\n_Take your time — I've noticed this topic is demanding more effort._",
        "**[Support mode]** {content}\n\n_I'm simplifying my output — your neural patterns suggest high mental effort. Let me know if you need a different angle._",
    ],
}

import random


class AIAdapter:
    """
    Manages the closed-loop AI adaptation loop.

    Responsibilities:
    1. Track cognitive state history across the session
    2. Select appropriate personality/response style
    3. Generate adapted responses (wrapping a base answer)
    4. Detect co-adaptation opportunities (patterns that signal model drift)
    """

    def __init__(self, config: AdaptationConfig = None):
        self.config = config or AdaptationConfig()
        self.state_history: list[dict] = []
        self.personality = PERSONALITIES["focus"]
        self.current_state = "focus"
        self.n_adaptations = 0
        self.session_start = time.time()

    def update_state(self, state: str, confidence: float, band_powers: dict):
        """
        Called each time a new EEG epoch is classified.
        Updates the personality and history.
        """
        self.current_state = state
        self.personality = PERSONALITIES[state]
        self.n_adaptations += 1
        self.state_history.append({
            "state": state,
            "confidence": confidence,
            "timestamp": time.time(),
            "band_powers": band_powers,
        })
        # Keep only recent history
        if len(self.state_history) > self.config.history_window:
            self.state_history = self.state_history[-self.config.history_window:]

    def adapt_response(self, base_content: str) -> str:
        """
        Wrap a base AI response with state-appropriate framing.
        In a real system, you'd pass the personality to the LLM as a system prompt.
        Here we demonstrate the concept with template wrapping.
        """
        templates = RESPONSE_TEMPLATES[self.current_state]
        template = random.choice(templates)
        return template.format(content=base_content)

    def get_system_prompt(self) -> str:
        """
        Generate a dynamic system prompt based on current cognitive state.
        This is what you'd send to an LLM API (GPT-4, Claude, etc.).
        """
        p = self.personality
        if self.current_state == "focus":
            return (
                "The user is in a focused, high-attention cognitive state. "
                "Be concise and information-dense. Skip preamble. "
                "No analogies unless asked. Assume high expertise. "
                "Respond as if talking to a peer."
            )
        else:
            return (
                "The user is experiencing high mental effort or cognitive load. "
                "Break down concepts step by step. Use simple language and analogies. "
                "Add checkpoints like 'Does this make sense?'. "
                "Be encouraging and supportive. Use shorter sentences."
            )

    def get_dominant_trend(self) -> Optional[str]:
        """
        Analyze state history to find patterns.
        Returns a trend description useful for co-adaptation.
        """
        if len(self.state_history) < 3:
            return None
        recent = [h["state"] for h in self.state_history[-5:]]
        effort_ratio = recent.count("high_effort") / len(recent)
        if effort_ratio >= 0.7:
            return "persistent_effort"  # user consistently struggling
        elif effort_ratio <= 0.2:
            return "sustained_focus"    # user in deep work
        else:
            return "fluctuating"        # mixed state

    def get_adaptation_summary(self) -> dict:
        """Return a summary dict for UI display."""
        elapsed = time.time() - self.session_start
        focus_count = sum(1 for h in self.state_history if h["state"] == "focus")
        effort_count = len(self.state_history) - focus_count
        return {
            "n_adaptations": self.n_adaptations,
            "dominant_state": "focus" if focus_count >= effort_count else "high_effort",
            "focus_ratio": focus_count / max(len(self.state_history), 1),
            "trend": self.get_dominant_trend(),
            "session_minutes": round(elapsed / 60, 1),
            "personality": {
                "verbosity": self.personality.verbosity,
                "step_by_step": self.personality.step_by_step,
                "tone": self.personality.tone,
            }
        }
