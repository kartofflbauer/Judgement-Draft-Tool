# streamlit_app.py
# -----------------
# Judgement: Eternal Champions Draft Tool (Picks, Bans, God selection)
# Usage:
#   1) pip install streamlit pillow numpy
#   2) streamlit run streamlit_app.py

import os
import re
import json
import csv
from io import StringIO, BytesIO
from dataclasses import dataclass
from typing import List, Dict, Optional

import streamlit as st
import numpy as np
from PIL import Image

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
# Preloaded CSV (classes & affiliations) — canonicalized
# -----------------------
PRELOAD_AFFILIATIONS_CSV = """name,classes,affiliations
Abothas,"Controller;Shooter","Krognar;Torin"
Allandir,"Shooter","Ista;Torin"
Aria,"Controller;Shooter","Grul;Torin"
Asbrand,"Enhancer;Tank","Bruelin;Avatar"
Aschell,"Shooter","Tomas;Torin"
Bale & Sarna,"Bruiser;Soulist","Bruelin;Grul"
Barnascus,"Bruiser;Shooter","Bruelin;Ista"
Bastian,"Enhancer","Torin;Avatar"
Brok,"Bruiser","Ista;Bruelin"
Carva,"Controller;Tank","Grul;Avatar"
Cradol,"Bruiser;Enhancer","Bruelin;Krognar"
Doenregar,"Tank","Grul;Ista"
Drelgoth,"Bruiser","Bruelin;Grul"
Fazeal,"Assassin;Controller","Tomas;Avatar"
Gendris,"Controller;Shooter","Grul;Avatar"
Grael,"Enhancer;Soulist","Krognar;Torin"
Haksa,"Controller;Enhancer","Ista;Torin"
Isabel,"Bruiser;Enhancer","Ista;Avatar"
Istariel,"Sniper","Ista;Torin"
Jaegar,"Controller;Shooter","Torin;Krognar"
Kain,"Bruiser;Shooter","Krognar;Tomas"
Kogan,"Bruiser;Shooter","Krognar;Bruelin"
Kruul,"Bruiser;Soulist","Bruelin;Krognar"
Kvarto,"Bruiser;Controller","Bruelin;Krognar"
Loribela,"Controller;Enhancer","Ista;Grul"
Lugdrug,"Enhancer;Tank","Bruelin;Krognar"
Maltique,"Controller;Sniper","ALL"
Marcus,"Controller;Tank","Ista;Tomas"
Masuzi,"Controller;Soulist","Tomas;Torin"
Naias,"Controller;Soulist","Krognar;Torin"
Nephenee,"Assassin","Ista;Tomas"
Onkura,"Controller;Tank","Tomas;Torin"
Piper,"Bruiser;Shooter","Krognar;Grul"
Rakkir,"Assassin","Tomas;Krognar"
Ramona,"Controller;Enhancer","Grul;Krognar"
Ravenos,"Controller;Tank","Torin;Tomas"
Saiyin,"Enhancer;Soulist","Ista;Tomas"
Sharn,"Bruiser;Tank","Tomas;Grul"
Skoll,"Bruiser;Tank","Krognar;Bruelin"
Skye,"Bruiser;Enhancer","Tomas;Torin"
Styx,"Controller;Soulist","Torin;Krognar"
Svetlana,"Controller;Soulist","Grul;Ista"
Thorgar,"Bruiser","Grul;Bruelin"
Thrommel,"Bruiser;Tank","Ista;Torin"
Urvexis,"Assassin","Torin;Avatar"
Viktor,"Controller;Shooter","Krognar;Tomas"
Xyvera,"Controller;Soulist","Torin;Grul"
Yasmin,"Enhancer;Shooter","Ista;Bruelin"
Yorgawth,"Bruiser","Grul;Krognar"
Zaffen,"Sniper","Grul;Bruelin"
Zaron,"Controller;Soulist","Krognar;Avatar"
Zhim'gigrak,"Controller;Shooter","Tomas;Torin"
Zhonyja,"Assassin","Bruelin;Avatar"
"""

# -----------------------
# Lightweight Data Model
# -----------------------
@dataclass
class HeroMeta:
    name: str
    classes: List[str]
    affiliations: List[str]  # two gods OR ["Avatar"]

# Build DB defaults (classes=[]; affiliations=["Avatar"])
HERO_DB: Dict[str, HeroMeta] = {
    h: HeroMeta(name=h, classes=[], affiliations=["Avatar"]) for h in DEFAULT_HEROES
}
ALL_CLASSES: List[str] = []
ALL_AFFILIATIONS: List[str] = []

# -----------------------
# Image config & helpers
# -----------------------
HERO_IMG_DIR = "assets/heroes"
GOD_IMG_DIR  = "assets/gods"
ACCEPT_EXTS  = (".png", ".jpg", ".jpeg", ".webp")

# Irregular filename aliases (full stem, including the "avatar-" prefix when you want it)
NAME_ALIASES = {
    "Bale & Sarna": "avatar-bale",
}

def _slug_nopunct(name: str) -> str:
    s = name.lower().replace("'", "")
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s

def _find_image(base_dir: str, stem: str) -> Optional[str]:
    for ext in ACCEPT_EXTS:
        p = os.path.join(base_dir, stem + ext)
        if os.path.exists(p):
            return p
    return None

def hero_image_path(name: str) -> Optional[str]:
    alias = NAME_ALIASES.get(name)
    if alias:
        p = _find_image(HERO_IMG_DIR, alias)
        if p: return p
    stem = "avatar-" + _slug_nopunct(name)
    p = _find_image(HERO_IMG_DIR, stem)
    if p: return p
    alt1 = _find_image(HERO_IMG_DIR, _slug_nopunct(name))
    if alt1: return alt1
    return _find_image(HERO_IMG_DIR, "avatar_" + _slug_nopunct(name))

def god_image_path(name: str) -> Optional[str]:
    stem = _slug_nopunct(name) + "-logo"
    p = _find_image(GOD_IMG_DIR, stem)
    if p: return p
    return _find_image(GOD_IMG_DIR, _slug_nopunct(name))

@st.cache_data(show_spinner=False)
def load_image_bytes(path: Optional[str]) -> Optional[bytes]:
    if not path:
        return None
    try:
        with open(path, "rb") as f:
            return f.read()
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def grayscale_bytes(img_bytes: bytes) -> Optional[bytes]:
    try:
        im = Image.open(BytesIO(img_bytes)).convert("RGB")
        gray = Image.fromarray(
            np.stack([np.array(im.convert("L"))]*3, axis=-1).astype(np.uint8)
        )
        out = BytesIO()
        gray.save(out, format="PNG")
        return out.getvalue()
    except Exception:
        return None

# Placeholder portrait (128x128 black square)
def get_placeholder_portrait():
    img = np.zeros((128, 128, 3), dtype=np.uint8)
    return img

# -----------------------
# Draft sequence
# -----------------------
@dataclass
class Step:
    kind: str   # 'ban' | 'pick' | 'god'
    player: str # 'P1' | 'P2'
    label: str  # user-facing label

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
# CSV → HERO_DB
# -----------------------
def _split_semis(s: str) -> List[str]:
    return [x.strip() for x in s.split(';') if x.strip()]

def _normalize_affils(affils: List[str]) -> List[str]:
    norm: List[str] = []
    for a in affils:
        a_clean = a.strip().strip('"').strip("'")
        if not a_clean: continue
        if a_clean.upper() == "ALL":
            norm.extend(DEFAULT_GODS); continue
        if a_clean.lower() == "avatar":
            norm.append("Avatar"); continue
        for g in DEFAULT_GODS:
            if a_clean.lower() == g.lower():
                a_clean = g; break
        norm.append(a_clean)
    seen, out = set(), []
    for x in norm:
        if x not in seen:
            seen.add(x); out.append(x)
    return out

def load_preloaded_meta():
    reader = csv.DictReader(StringIO(PRELOAD_AFFILIATIONS_CSV))
    updates: Dict[str, HeroMeta] = {}
    for row in reader:
        name = (row.get("name") or "").strip()
        if not name or name not in HERO_DB: continue
        classes = _split_semis((row.get("classes") or "").strip())
        affils  = _normalize_affils(_split_semis((row.get("affiliations") or "").strip()))
        updates[name] = HeroMeta(name=name, classes=classes, affiliations=affils or ["Avatar"])
    for k, v in updates.items(): HERO_DB[k] = v
    global ALL_CLASSES, ALL_AFFILIATIONS
    ALL_CLASSES = sorted({c for m in HERO_DB.values() for c in m.classes})
    ALL_AFFILIATIONS = sorted({a for m in HERO_DB.values() for a in m.affiliations})

# -----------------------
# Session State
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
        st.session_state.history: List[dict] = []
    if '_filter_classes' not in st.session_state:
        st.session_state._filter_classes = []
    if '_filter_affils' not in st.session_state:
        st.session_state._filter_affils = []

def reset_draft():
    st.session_state.available_heroes = st.session_state.heroes.copy()
    st.session_state.available_gods   = st.session_state.gods.copy()
    st.session_state.bans             = []
    st.session_state.picks            = {'P1': [], 'P2': []}
    st.session_state.gods_picked      = {'P1': None, 'P2': None}
    st.session_state.step_idx         = 0
    st.session_state.history          = []

# -----------------------
# Draft logic
# -----------------------
@dataclass
class Step:
    kind: str
    player: str
    label: str

def apply_action(step: Step, choice: str):
    if step.kind in ('ban','pick'):
        if choice not in st.session_state.available_heroes:
            st.toast("That hero is not available.", icon="⚠️"); return False
        if step.kind == 'pick' and choice in st.session_state.bans:
            st.toast("That hero is banned.", icon="🚫"); return False
    elif step.kind == 'god':
        if choice not in st.session_state.available_gods:
            st.toast("That god is not available.", icon="⚠️"); return False

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
    st.session_state.history.append(record)
    st.session_state.step_idx += 1
    return True

def undo_last():
    if not st.session_state.history:
        st.toast("Nothing to undo.", icon="ℹ️"); return
    record = st.session_state.history.pop()
    step = Step(**record['step'])
    choice = record['choice']
    if step.kind == 'ban':
        if choice in st.session_state.bans:
            st.session_state.bans.remove(choice)
        if choice not in st.session_state.available_heroes:
            st.session_state.available_heroes.append(choice); st.session_state.available_heroes.sort()
    elif step.kind == 'pick':
        if choice in st.session_state.picks[step.player]:
            st.session_state.picks[step.player].remove(choice)
        if choice not in st.session_state.available_heroes:
            st.session_state.available_heroes.append(choice); st.session_state.available_heroes.sort()
    elif step.kind == 'god':
        st.session_state.gods_picked[step.player] = None
        if not ALLOW_DUPLICATE_GODS and (choice not in st.session_state.available_gods):
            st.session_state.available_gods.append(choice); st.session_state.available_gods.sort()
    st.session_state.step_idx = record['step_idx']

def export_state() -> str:
    all_picked = st.session_state.picks['P1'] + st.session_state.picks['P2']
    all_banned = st.session_state.bans
    remaining  = [h for h in st.session_state.heroes if h not in all_picked and h not in all_banned]
    summary = {
        "bans": all_banned,
        "warband_P1": st.session_state.picks['P1'],
        "god_P1": st.session_state.gods_picked.get('P1'),
        "warband_P2": st.session_state.picks['P2'],
        "god_P2": st.session_state.gods_picked.get('P2'),
        "remaining_heroes": remaining
    }
    return json.dumps(summary, indent=2)

# -----------------------
# UI helpers
# -----------------------
def render_hero_chip(hname: str):
    meta = HERO_DB.get(hname, HeroMeta(hname, [], ["Avatar"]))
    col_img, col_text = st.columns([1, 3])
    with col_img:
        img_bytes = load_image_bytes(hero_image_path(hname))
        if img_bytes:
            st.image(img_bytes, width=48)
        else:
            st.image(get_placeholder_portrait(), width=48)
    with col_text:
        classes = ", ".join(meta.classes) if meta.classes else "—"
        affils  = ", ".join(meta.affiliations) if meta.affiliations else "—"
        st.markdown(f"""**{hname}**

Classes: {classes}  |  Affil: {affils}""")

# -----------------------
# Page
# -----------------------
st.set_page_config(page_title="Judgement: Eternal Champions Draft Tool", page_icon=None, layout="wide")
init_state()
load_preloaded_meta()

# Title
st.title("Judgement: Eternal Champions Draft Tool")

# ====== TOP: Current Step Interaction (moved here) ======
if st.session_state.step_idx < len(DRAFT_SEQUENCE):
    step = DRAFT_SEQUENCE[st.session_state.step_idx]
    st.header(step.label)

    # Filters affect choices
    def hero_passes_filters(h: str) -> bool:
        meta = HERO_DB.get(h, HeroMeta(h, [], ["Avatar"]))
        cls_ok = True if not st.session_state._filter_classes else any(c in st.session_state._filter_classes for c in meta.classes)
        aff_ok = True if not st.session_state._filter_affils  else any(a in st.session_state._filter_affils  for a in meta.affiliations)
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
                st.toast(f"{step.player} {step.kind}ed {choice}", icon="✅")
                st.rerun()
    elif step.kind == 'god':
        available_gods = st.session_state.gods.copy()
        picked = [g for g in st.session_state.gods_picked.values() if g]
        if st.session_state.get('_enforce_unique_gods', True):
            available_gods = [g for g in available_gods if g not in picked]
        choice = st.selectbox("Choose a god:", available_gods, index=0 if available_gods else None, key=f"god_{st.session_state.step_idx}")
        if choice:
            gbytes = load_image_bytes(god_image_path(choice))
            if gbytes:
                st.image(gbytes, width=64, caption=choice)
        confirm = st.button("Confirm God", type="primary")
        if confirm and choice:
            if st.session_state.get('_enforce_unique_gods', True):
                if choice not in st.session_state.available_gods:
                    st.session_state.available_gods = [g for g in st.session_state.gods if g not in picked]
            ok = apply_action(step, choice)
            if ok:
                st.toast(f"{step.player} picked {choice}", icon="✅")
                st.rerun()
else:
    st.header("Draft Complete ✅")

# ====== SIDEBAR ======
with st.sidebar:
    st.header("⚙️ Setup")
    st.caption("Built-in heroes, classes, affiliations, and portraits are loaded.")
    enforce_unique = st.checkbox("Enforce unique gods (no duplicates)", value=True)
    st.session_state._enforce_unique_gods = enforce_unique

    st.markdown("---")
    st.subheader("Filters")
    sel_classes = st.multiselect("Classes", options=ALL_CLASSES, default=st.session_state._filter_classes)
    sel_affils  = st.multiselect("Affiliations", options=sorted(set(ALL_AFFILIATIONS)), default=st.session_state._filter_affils)
    st.session_state._filter_classes = sel_classes
    st.session_state._filter_affils  = sel_affils

    st.markdown("---")
    st.subheader("Actions")
    colA, colB = st.columns(2)
    with colA:
        if st.button("↩️ Undo"): undo_last()
    with colB:
        if st.button("♻️ Full Reset"): reset_draft()

    st.markdown("---")
    st.subheader("Export")
    export_str = export_state()
    st.download_button("Download Draft JSON", data=export_str, file_name="draft_state.json", mime="application/json")

st.markdown("---")

# ====== MIDDLE: Panels (Sequence + Players) ======
left, mid, right = st.columns([1.2, 1, 1])

with left:
    st.subheader("Sequence")
    for i, s in enumerate(DRAFT_SEQUENCE):
        prefix = "➡️" if i == st.session_state.step_idx else "  "
        st.write(f"{prefix} {i+1:02d}. {s.label}")

with mid:
    st.subheader("Player 1")
    st.markdown("**Picks**")
    if st.session_state.picks['P1']:
        for h in st.session_state.picks['P1']: render_hero_chip(h)
    else:
        st.write("—")
    st.markdown("**God**")
    g1 = st.session_state.gods_picked['P1']
    if g1:
        gbytes = load_image_bytes(god_image_path(g1))
        if gbytes: st.image(gbytes, width=48, caption=g1)
        else:      st.write(g1)
    else:
        st.write("—")
    st.markdown("**Bans**")
    st.write(", ".join([b for b in st.session_state.bans if b not in st.session_state.picks['P1'] and b not in st.session_state.picks['P2']]) or "—")

with right:
    st.subheader("Player 2")
    st.markdown("**Picks**")
    if st.session_state.picks['P2']:
        for h in st.session_state.picks['P2']: render_hero_chip(h)
    else:
        st.write("—")
    st.markdown("**God**")
    g2 = st.session_state.gods_picked['P2']
    if g2:
        gbytes = load_image_bytes(god_image_path(g2))
        if gbytes: st.image(gbytes, width=48, caption=g2)
        else:      st.write(g2)
    else:
        st.write("—")

st.markdown("---")

# ====== BOTTOM: Portrait Grid (grays out picked/banned) ======
st.subheader("Hero Portraits")
cols = st.columns(8, gap="small")  # tweak columns per your taste
picked_or_banned = set(st.session_state.picks['P1'] + st.session_state.picks['P2'] + st.session_state.bans)

for i, h in enumerate(st.session_state.heroes):
    with cols[i % len(cols)]:
        b = load_image_bytes(hero_image_path(h))
        if b is None:
            # fallback to placeholder
            st.image(get_placeholder_portrait(), use_column_width=True)
        else:
            if h in picked_or_banned:
                gb = grayscale_bytes(b) or b  # graceful fallback
                st.image(gb, use_column_width=True)
            else:
                st.image(b, use_column_width=True)

# Final summary when complete
if st.session_state.step_idx >= len(DRAFT_SEQUENCE):
    st.markdown("---")
    st.subheader("Summary JSON")
    st.code(export_state(), language='json')
