# streamlit_app.py
# -----------------
# MOBA 5v5 Drafting Tool (Picks, Bans, and God selection)
# Usage:
#   1) pip install streamlit
#   2) streamlit run streamlit_app.py
#
# Customize HEROES and GODS with your actual names below or paste lists via the sidebar.
# The sequence matches the one you provided exactly. It enforces no duplicate heroes,
# prevents picking banned heroes, and (by default) makes gods unique per team.
# Includes Undo, Reset, and Export JSON.

import json
from dataclasses import dataclass
from typing import List, Dict, Optional

import streamlit as st

# -----------------------
# Defaults (edit freely)
# -----------------------
DEFAULT_HEROES = [f"Hero {i+1}" for i in range(53)]
DEFAULT_GODS = [f"God {i+1}" for i in range(6)]

# Allow same god for both teams? Set to False to enforce uniqueness.
ALLOW_DUPLICATE_GODS = False

@dataclass
class Step:
    kind: str   # 'ban' | 'pick' | 'god'
    player: str # 'P1' | 'P2'
    label: str  # user-facing label

# Draft order as specified
DRAFT_SEQUENCE: List[Step] = [
    Step('ban','P1','Player 1 Ban'),
    Step('ban','P2','Player 2 Ban'),
    Step('pick','P1','Player 1 Pick 1st Hero'),
    Step('pick','P2','Player 2 Pick 1st Hero'),
    Step('ban','P1','Player 1 Ban'),
    Step('ban','P2','Player 2 Ban'),
    Step('ban','P1','Player 1 Ban'),
    Step('ban','P2','Player 2 Ban'),
    Step('pick','P2','Player 2 Pick 2nd Hero'),
    Step('pick','P1','Player 1 Pick 2nd Hero'),
    Step('ban','P2','Player 2 Ban'),
    Step('ban','P1','Player 1 Ban'),
    Step('pick','P1','Player 1 Pick 3rd Hero'),
    Step('pick','P2','Player 2 Pick 3rd Hero'),
    Step('ban','P2','Player 2 Ban'),
    Step('ban','P1','Player 1 Ban'),
    Step('ban','P1','Player 1 Ban'),
    Step('ban','P2','Player 2 Ban'),
    Step('pick','P2','Player 2 Pick 4th Hero'),
    Step('pick','P1','Player 1 Pick 4th Hero'),
    Step('ban','P1','Player 1 Ban'),
    Step('ban','P2','Player 2 Ban'),
    Step('pick','P1','Player 1 Pick 5th Hero'),
    Step('pick','P2','Player 2 Pick 5th Hero'),
    Step('god','P1','Player 1 Pick God'),
    Step('god','P2','Player 2 Pick God'),
]

# -----------------------
# Session State Helpers
# -----------------------

def init_state():
    if 'heroes' not in st.session_state:
        st.session_state.heroes = DEFAULT_HEROES.copy()
    if 'gods' not in st.session_state:
        st.session_state.gods = DEFAULT_GODS.copy()
    if 'available_heroes' not in st.session_state:
        st.session_state.available_heroes = st.session_state.heroes.copy()
    if 'available_gods' not in st.session_state:
        st.session_state.available_gods = st.session_state.gods.copy()
    if 'bans' not in st.session_state:
        st.session_state.bans: List[str] = []
    if 'picks' not in st.session_state:
        st.session_state.picks: Dict[str, List[str]] = {'P1': [], 'P2': []}
    if 'gods_picked' not in st.session_state:
        st.session_state.gods_picked: Dict[str, Optional[str]] = {'P1': None, 'P2': None}
    if 'step_idx' not in st.session_state:
        st.session_state.step_idx = 0
    if 'history' not in st.session_state:
        st.session_state.history: List[dict] = []  # action log for Undo


def reset_draft():
    st.session_state.available_heroes = st.session_state.heroes.copy()
    st.session_state.available_gods = st.session_state.gods.copy()
    st.session_state.bans = []
    st.session_state.picks = {'P1': [], 'P2': []}
    st.session_state.gods_picked = {'P1': None, 'P2': None}
    st.session_state.step_idx = 0
    st.session_state.history = []


def apply_action(step: Step, choice: str):
    # Validate
    if step.kind in ('ban','pick'):
        if choice not in st.session_state.available_heroes:
            st.toast("That hero is not available.", icon="⚠️")
            return False
        if step.kind == 'pick' and choice in st.session_state.bans:
            st.toast("That hero is banned.", icon="🚫")
            return False
    elif step.kind == 'god':
        if choice not in st.session_state.available_gods:
            st.toast("That god is not available.", icon="⚠️")
            return False
        if not ALLOW_DUPLICATE_GODS:
            # If other team already picked this god, it won't be in available_gods anyway
            pass

    # Apply
    record = {'step_idx': st.session_state.step_idx, 'step': step.__dict__, 'choice': choice}

    if step.kind == 'ban':
        st.session_state.bans.append(choice)
        st.session_state.available_heroes.remove(choice)
    elif step.kind == 'pick':
        st.session_state.picks[step.player].append(choice)
        st.session_state.available_heroes.remove(choice)
    elif step.kind == 'god':
        st.session_state.gods_picked[step.player] = choice
        if not ALLOW_DUPLICATE_GODS:
            st.session_state.available_gods.remove(choice)
    else:
        st.toast("Unknown action.", icon="❓")
        return False

    st.session_state.history.append(record)
    st.session_state.step_idx += 1
    return True


def undo_last():
    if not st.session_state.history:
        st.toast("Nothing to undo.", icon="ℹ️")
        return

    record = st.session_state.history.pop()
    step = Step(**record['step'])
    choice = record['choice']

    # Revert state
    if step.kind == 'ban':
        if choice in st.session_state.bans:
            st.session_state.bans.remove(choice)
        if choice not in st.session_state.available_heroes:
            st.session_state.available_heroes.append(choice)
            st.session_state.available_heroes.sort()
    elif step.kind == 'pick':
        if choice in st.session_state.picks[step.player]:
            st.session_state.picks[step.player].remove(choice)
        if choice not in st.session_state.available_heroes:
            st.session_state.available_heroes.append(choice)
            st.session_state.available_heroes.sort()
    elif step.kind == 'god':
        st.session_state.gods_picked[step.player] = None
        if not ALLOW_DUPLICATE_GODS and (choice not in st.session_state.available_gods):
            st.session_state.available_gods.append(choice)
            st.session_state.available_gods.sort()

    st.session_state.step_idx = record['step_idx']


def export_state() -> str:
    payload = {
        'sequence': [s.__dict__ for s in DRAFT_SEQUENCE],
        'bans': st.session_state.bans,
        'picks': st.session_state.picks,
        'gods': st.session_state.gods_picked,
        'remaining_heroes': st.session_state.available_heroes,
        'remaining_gods': st.session_state.available_gods,
    }
    return json.dumps(payload, indent=2)


# -----------------------
# UI
# -----------------------

st.set_page_config(page_title="MOBA Draft Tool", page_icon="🎮", layout="wide")
init_state()

with st.sidebar:
    st.header("⚙️ Setup")
    st.caption("Paste your own lists (one per line) and click Apply.")
    heroes_text = st.text_area("Heroes (one per line)", value='\n'.join(st.session_state.heroes), height=180)
    gods_text = st.text_area("Gods (one per line)", value='\n'.join(st.session_state.gods), height=100)
    enforce_unique = st.checkbox("Enforce unique gods (no duplicates)", value=(not ALLOW_DUPLICATE_GODS))
    if enforce_unique == ALLOW_DUPLICATE_GODS:
        # If user toggles, reflect in runtime toggle only (not module constant)
        st.session_state._enforce_unique_gods = enforce_unique
    else:
        st.session_state._enforce_unique_gods = enforce_unique

    if st.button("Apply Lists / Reset Draft", type="primary"):
        new_heroes = [h.strip() for h in heroes_text.splitlines() if h.strip()]
        new_gods = [g.strip() for g in gods_text.splitlines() if g.strip()]
        st.session_state.heroes = new_heroes or DEFAULT_HEROES.copy()
        st.session_state.gods = new_gods or DEFAULT_GODS.copy()
        reset_draft()

    st.markdown("---")
    st.subheader("Actions")
    colA, colB = st.columns(2)
    with colA:
        if st.button("↩️ Undo"):
            undo_last()
    with colB:
        if st.button("♻️ Full Reset"):
            reset_draft()

    st.markdown("---")
    st.subheader("Export")
    export_str = export_state()
    st.download_button("Download Draft JSON", data=export_str, file_name="draft_state.json", mime="application/json")


# Header / Overview
st.title("🎮 MOBA 5v5 Drafting Tool")

left, mid, right = st.columns([1.2, 1, 1])

with left:
    st.subheader("Sequence")
    for i, step in enumerate(DRAFT_SEQUENCE):
        prefix = "➡️" if i == st.session_state.step_idx else "  "
        st.write(f"{prefix} {i+1:02d}. {step.label}")

with mid:
    st.subheader("Player 1")
    st.markdown("**Picks**")
    st.write(", ".join(st.session_state.picks['P1']) or "—")
    st.markdown("**God**")
    st.write(st.session_state.gods_picked['P1'] or "—")
    st.markdown("**Bans**")
    st.write(", ".join([b for b in st.session_state.bans if b not in st.session_state.picks['P1'] and b not in st.session_state.picks['P2']]) or "—")

with right:
    st.subheader("Player 2")
    st.markdown("**Picks**")
    st.write(", ".join(st.session_state.picks['P2']) or "—")
    st.markdown("**God**")
    st.write(st.session_state.gods_picked['P2'] or "—")

st.markdown("---")

# Current Step Interaction
if st.session_state.step_idx < len(DRAFT_SEQUENCE):
    step = DRAFT_SEQUENCE[st.session_state.step_idx]
    st.header(step.label)

    if step.kind in ('ban','pick'):
        # Filter choices: available heroes; for clarity, show banned heroes struck? Keep simple.
        options = st.session_state.available_heroes.copy()
        choice = st.selectbox("Choose a hero:", options, index=0 if options else None, key=f"select_{st.session_state.step_idx}")
        confirm = st.button("Confirm", type="primary")
        if confirm and choice:
            ok = apply_action(step, choice)
            if ok:
                st.success(f"{step.player} {step.kind}ed {choice}")

    elif step.kind == 'god':
        # Available gods (enforce uniqueness based on sidebar toggle)
        available_gods = st.session_state.gods.copy()
        # Recompute availability considering picked gods and uniqueness toggle
        picked = [g for g in st.session_state.gods_picked.values() if g]
        if st.session_state.get('_enforce_unique_gods', True):
            available_gods = [g for g in available_gods if g not in picked]
        choice = st.selectbox("Choose a god:", available_gods, index=0 if available_gods else None, key=f"god_{st.session_state.step_idx}")
        confirm = st.button("Confirm God", type="primary")
        if confirm and choice:
            # Temporarily adjust available_gods if uniqueness is enforced
            if st.session_state.get('_enforce_unique_gods', True):
                if choice not in st.session_state.available_gods:
                    st.session_state.available_gods = [g for g in st.session_state.gods if g not in picked]
            ok = apply_action(step, choice)
            if ok:
                st.success(f"{step.player} picked {choice}")
else:
    st.header("Draft Complete ✅")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Player 1 Roster")
        st.write(", ".join(st.session_state.picks['P1']) or "—")
        st.write(f"God: {st.session_state.gods_picked['P1'] or '—'}")
    with col2:
        st.subheader("Player 2 Roster")
        st.write(", ".join(st.session_state.picks['P2']) or "—")
        st.write(f"God: {st.session_state.gods_picked['P2'] or '—'}")

    st.markdown("---")
    st.subheader("Summary JSON")
    st.code(export_state(), language='json')


