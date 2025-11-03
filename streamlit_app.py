# streamlit_app.py
# -----------------
# Judgement: Eternal Champions Draft Tool (Picks, Bans, and God selection)
# Usage:
#   1) pip install streamlit
#   2) streamlit run streamlit_app.py

import json
from dataclasses import dataclass
from typing import List, Dict, Optional

import streamlit as st
import numpy as np

# -----------------------
# Defaults (edit freely)
# -----------------------
DEFAULT_HEROES = [
    "Abothas","Allandir","Aria","Asbrand","Aschell","Bale & Sarna","Barnascus","Bastian","Brok","Carva",
    "Cradol","Doenregar","Drelgoth","Fazeal","Gendris","Grael","Haksa","Isabel","Istariel","Jaegar",
    "Kain","Kogan","Kruul","Kvarto","Loribela","Lugdrug","Maltique","Marcus","Masuzi","Naias",
    "Nephenee","Onkura","Piper","Rakkir","Ramona","Ravenos","Saiyin","Sharn","Skoll","Skye",
    "Styx","Svetlana","Thorgar","Thrommel","Urvexis","Viktor","Xyvera","Yasmin","Yorgawth","Zaffen",
    "Zaron","Zhim'gigrak","Zhonyja"
]
DEFAULT_GODS = ["Bruelin","Grul","Ista","Krognar","Tomas","Torin"]

# Allow same god for both teams? Set to False to enforce uniqueness.
ALLOW_DUPLICATE_GODS = False

# -----------------------
# Lightweight Data Model
# -----------------------
@dataclass
class HeroMeta:
    name: str
    classes: List[str]
    affiliations: List[str]  # two gods OR ["avatar"]

# Build a basic DB with blank data (classes=[]; affiliations=["avatar"]).
HERO_DB: Dict[str, HeroMeta] = {
    h: HeroMeta(name=h, classes=[], affiliations=["avatar"]) for h in DEFAULT_HEROES
}

ALL_CLASSES = sorted({c for m in HERO_DB.values() for c in m.classes})
if not ALL_CLASSES:
    ALL_CLASSES = []  # none yet
ALL_AFFILIATIONS = sorted({a for m in HERO_DB.values() for a in m.affiliations})

# Placeholder portrait (128x128 black square)
def get_placeholder_portrait():
    img = np.zeros((128, 128, 3), dtype=np.uint8)
    return img

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
    if '_filter_classes' not in st.session_state:
        st.session_state._filter_classes = []
    if '_filter_affils' not in st.session_state:
        st.session_state._filter_affils = []

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

# Helper to render a hero chip with portrait + tags
def render_hero_chip(hname: str):
    meta = HERO_DB.get(hname, HeroMeta(hname, [], ["avatar"]))
    col_img, col_text = st.columns([1, 3])
    with col_img:
        st.image(get_placeholder_portrait(), width=48)
    with col_text:
        classes = ", ".join(meta.classes) if meta.classes else "—"
        affils = ", ".join(meta.affiliations) if meta.affiliations else "—"
        st.markdown(f"**{hname}**\\n\\nClasses: {classes}  |  Affil: {affils}")

# -----------------------
# UI
# -----------------------
st.set_page_config(page_title="Judgement: Eternal Champions Draft Tool", page_icon=None, layout="wide")
init_state()

with st.sidebar:
    st.header("⚙️ Setup")
    st.caption("Using built-in heroes & gods. You can still override below if needed.")

    enforce_unique = st.checkbox("Enforce unique gods (no duplicates)", value=True)
    st.session_state._enforce_unique_gods = enforce_unique

    # Filters
    st.markdown("---")
    st.subheader("Filters")
    sel_classes = st.multiselect("Classes", options=ALL_CLASSES, default=st.session_state._filter_classes)
    sel_affils = st.multiselect("Affiliations", options=sorted(set(ALL_AFFILIATIONS + ["avatar"])), default=st.session_state._filter_affils)
    st.session_state._filter_classes = sel_classes
    st.session_state._filter_affils = sel_affils

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

# Optional: Advanced roster overrides (outside the sidebar, as an expander)
with st.expander("Advanced: custom rosters"):
    heroes_text = st.text_area("Heroes (one per line)", value="\n".join(st.session_state.heroes), height=180)
    gods_text = st.text_area("Gods (one per line)", value="\n".join(st.session_state.gods), height=100)

    if st.button("Apply Lists / Reset Draft", type="primary"):
        # NOTE: No 'global' needed at module scope
        new_heroes = [h.strip() for h in heroes_text.splitlines() if h.strip()]
        new_gods = [g.strip() for g in gods_text.splitlines() if g.strip()]
        st.session_state.heroes = new_heroes or DEFAULT_HEROES.copy()
        st.session_state.gods = new_gods or DEFAULT_GODS.copy()

        # Rebuild DB with defaults for new heroes
        HERO_DB = {h: HeroMeta(name=h, classes=[], affiliations=["avatar"]) for h in st.session_state.heroes}
        ALL_CLASSES = sorted({c for m in HERO_DB.values() for c in m.classes})
        ALL_AFFILIATIONS = sorted({a for m in HERO_DB.values() for a in m.affiliations})
        reset_draft()

# Header / Overview
st.title("Judgement: Eternal Champions Draft Tool")

left, mid, right = st.columns([1.2, 1, 1])

with left:
    st.subheader("Sequence")
    for i, step in enumerate(DRAFT_SEQUENCE):
        prefix = "➡️" if i == st.session_state.step_idx else "  "
        st.write(f"{prefix} {i+1:02d}. {step.label}")

with mid:
    st.subheader("Player 1")
    st.markdown("**Picks**")
    if st.session_state.picks['P1']:
        for h in st.session_state.picks['P1']:
            render_hero_chip(h)
    else:
        st.write("—")
    st.markdown("**God**")
    st.write(st.session_state.gods_picked['P1'] or "—")
    st.markdown("**Bans**")
    st.write(", ".join([b for b in st.session_state.bans if b not in st.session_state.picks['P1'] and b not in st.session_state.picks['P2']]) or "—")

with right:
    st.subheader("Player 2")
    st.markdown("**Picks**")
    if st.session_state.picks['P2']:
        for h in st.session_state.picks['P2']:
            render_hero_chip(h)
    else:
        st.write("—")
    st.markdown("**God**")
    st.write(st.session_state.gods_picked['P2'] or "—")

st.markdown("---")

# Current Step Interaction
if st.session_state.step_idx < len(DRAFT_SEQUENCE):
    step = DRAFT_SEQUENCE[st.session_state.step_idx]
    st.header(step.label)

    # Compute filtered choices for heroes
    def hero_passes_filters(h: str) -> bool:
        meta = HERO_DB.get(h, HeroMeta(h, [], ["avatar"]))
        cls_ok = True if not st.session_state._filter_classes else any(c in st.session_state._filter_classes for c in meta.classes)
        aff_ok = True if not st.session_state._filter_affils else any(a in st.session_state._filter_affils for a in meta.affiliations)
        return cls_ok and aff_ok

    if step.kind in ('ban','pick'):
        base_opts = st.session_state.available_heroes.copy()
        options = [h for h in base_opts if hero_passes_filters(h)]
        choice = st.selectbox("Choose a hero:", options, index=0 if options else None, key=f"select_{st.session_state.step_idx}")
        if choice:
            render_hero_chip(choice)
        confirm = st.button("Confirm", type="primary")
        if confirm and choice:
            ok = apply_action(step, choice)
            if ok:
                st.success(f"{step.player} {step.kind}ed {choice}")

    elif step.kind == 'god':
        # Available gods (enforce uniqueness based on sidebar toggle)
        available_gods = st.session_state.gods.copy()
        picked = [g for g in st.session_state.gods_picked.values() if g]
        if st.session_state.get('_enforce_unique_gods', True):
            available_gods = [g for g in available_gods if g not in picked]
        choice = st.selectbox("Choose a god:", available_gods, index=0 if available_gods else None, key=f"god_{st.session_state.step_idx}")
        confirm = st.button("Confirm God", type="primary")
        if confirm and choice:
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
