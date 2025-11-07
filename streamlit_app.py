# streamlit_app.py
# -----------------
# Judgement: Eternal Champions Draft Tool (Picks, Bans, God selection)
# Usage:
#   1) pip install streamlit pillow numpy supabase
#   2) streamlit run streamlit_app.py

import os
import re
import json
import csv
import uuid
from io import StringIO, BytesIO
from dataclasses import dataclass
from typing import List, Dict, Optional

import streamlit as st
import numpy as np
from PIL import Image

from supabase import create_client, Client
import time

# --- Optional Supabase import guard ---
HAS_SUPABASE = False  # default
try:
    from supabase import create_client, Client
    HAS_SUPABASE = True
except Exception as _e:
    create_client = None
    Client = None
    SUPABASE_IMPORT_ERR = str(_e)

# -----------------------
# Defaults (edit freely)
# -----------------------
DEFAULT_HEROES = [
    "Abhothas","Allandir","Aria","Asbrand","Aschell","Bale & Sarna","Barnascus","Bastian","Brok","Carva",
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
Abhothas,"Controller;Shooter","Krognar;Torin"
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
HERO_DB: Dict[str, HeroMeta] = {h: HeroMeta(name=h, classes=[], affiliations=["Avatar"]) for h in DEFAULT_HEROES}
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
    # Match your convention first: "avatar-<god>"
    stem = "avatar-" + _slug_nopunct(name)
    p = _find_image(GOD_IMG_DIR, stem)
    if p: return p
    # Fallback: plain slug
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
        gray = Image.fromarray(np.stack([np.array(im.convert("L"))]*3, axis=-1).astype(np.uint8))
        out = BytesIO()
        gray.save(out, format="PNG")
        return out.getvalue()
    except Exception:
        return None

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

# -------- Supabase adapter --------
def _sb_client() -> "Client":
    if not HAS_SUPABASE:
        st.error(f"Supabase import failed: {SUPABASE_IMPORT_ERR}")
        st.stop()
    url = st.secrets.get("SUPABASE_URL")
    key = st.secrets.get("SUPABASE_ANON_KEY")
    if not url or not key:
        st.error("Supabase secrets missing. Add SUPABASE_URL and SUPABASE_ANON_KEY to Streamlit secrets.")
        st.stop()
    return create_client(url, key)

def sync_now():
    """Manually pull latest state from the lobby and refresh UI."""
    if not st.session_state.get("lobby_id"):
        st.toast("Join a lobby to sync.", icon="ℹ️")
        return
    row = sb_load_draft(st.session_state["lobby_id"])
    if row and row.get("state"):
        remote = row["state"]
        local  = snapshot_state()
        if remote != local:
            apply_snapshot(remote)
            st.toast("Synced latest from lobby.", icon="🔄")
        else:
            st.toast("Already up to date.", icon="✅")
    else:
        # No row yet — seed the draft from current local state
        sb_save_draft(st.session_state["lobby_id"])
        st.toast("Created draft for this lobby.", icon="🆕")
    st.rerun()


@st.cache_resource
def get_sb() -> Client:
    return _sb_client()

def sb_upsert_lobby(code: str) -> Optional[str]:
    sb = get_sb()
    code = code.strip().lower()
    payload = {"code": code}

    # Try the clean upsert path first
    try:
        sb.table("lobbies").upsert(payload, on_conflict="code").execute()
    except Exception as e:
        # Fallback for older clients: insert with upsert flags
        try:
            sb.table("lobbies").insert(payload, upsert=True, on_conflict="code").execute()
        except Exception as e2:
            # Swallow duplicate-key only; re-raise other errors
            msg = f"{e}\n{e2}"
            if "duplicate key value violates unique constraint" not in msg and "23505" not in msg:
                raise

    # Always fetch the id explicitly (no .single()/.maybe_single())
    res = sb.table("lobbies").select("id").eq("code", code).limit(1).execute()
    rows = getattr(res, "data", None)
    return rows[0]["id"] if rows and len(rows) > 0 else None

def snapshot_state() -> dict:
    return {
        "bans": st.session_state.bans,
        "picks": st.session_state.picks,
        "gods": st.session_state.gods_picked,
        "available_heroes": st.session_state.available_heroes,
        "available_gods": st.session_state.available_gods,
        "step_idx": st.session_state.step_idx,
        # Fearless BO3
        "fearless_bo3": st.session_state.fearless_bo3,
        "series_game": st.session_state.series_game,
        "series_banned": sorted(list(st.session_state.series_banned)),
        "version": 1,
    }


def apply_snapshot(s: dict):
    st.session_state.bans              = s.get("bans", [])
    st.session_state.picks             = s.get("picks", {"P1": [], "P2": []})
    st.session_state.gods_picked       = s.get("gods", {"P1": None, "P2": None})
    st.session_state.available_heroes  = s.get("available_heroes", st.session_state.heroes.copy())
    st.session_state.available_gods    = s.get("available_gods", st.session_state.gods.copy())
    st.session_state.step_idx          = s.get("step_idx", 0)
    # Fearless BO3
    st.session_state.fearless_bo3      = bool(s.get("fearless_bo3", False))
    st.session_state.series_game       = int(s.get("series_game", 1))
    st.session_state.series_banned     = set(s.get("series_banned", []))


def sb_save_draft(lobby_id: str):
    sb = get_sb()
    state = snapshot_state()
    sb.table("drafts").upsert(
        {"lobby_id": lobby_id, "state": state, "step_idx": state["step_idx"]},
        on_conflict="lobby_id"
    ).execute()

def sb_load_draft(lobby_id: str) -> Optional[dict]:
    """Return the draft row (dict) for a lobby_id, or None if it doesn't exist."""
    sb = get_sb()
    res = sb.table("drafts")\
            .select("state, updated_at, step_idx")\
            .eq("lobby_id", lobby_id)\
            .limit(1)\
            .execute()
    rows = getattr(res, "data", None)
    return rows[0] if rows and len(rows) > 0 else None

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
    # lobby defaults
    if 'lobby_code' not in st.session_state:
        st.session_state.lobby_code = ""
    if 'lobby_id' not in st.session_state:
        st.session_state.lobby_id = None
    if 'role' not in st.session_state:
        st.session_state.role = 'master'      # 'master' | 'p1' | 'p2'
    #f '_live_sync' not in st.session_state:
    #   st.session_state._live_sync = False   # off until a lobby is joined
        # --- Fearless BO3 defaults ---
    if 'fearless_bo3' not in st.session_state:
        st.session_state.fearless_bo3 = False  # toggle in sidebar
    if 'series_game' not in st.session_state:
        st.session_state.series_game = 1       # 1..3
    if 'series_banned' not in st.session_state:
        st.session_state.series_banned = set() # heroes auto-banned from prior games


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

def _recompute_available_from_bans():
    st.session_state.available_heroes = [h for h in st.session_state.heroes if h not in st.session_state.bans]

def reset_game_from_series():
    """Reset only the current game, preserving Fearless series bans."""
    st.session_state.bans = sorted(list(st.session_state.series_banned)) if st.session_state.fearless_bo3 else []
    _recompute_available_from_bans()
    st.session_state.available_gods   = st.session_state.gods.copy()
    st.session_state.picks            = {'P1': [], 'P2': []}
    st.session_state.gods_picked      = {'P1': None, 'P2': None}
    st.session_state.step_idx         = 0
    st.session_state.history          = []

def start_series_if_needed():
    """If Fearless is on and we're at game 1, ensure game state respects series bans."""
    if st.session_state.fearless_bo3 and st.session_state.series_game == 1 and not st.session_state.history:
        # initial game: no series bans yet, but ensure structure is sound
        reset_game_from_series()

def advance_to_next_game():
    """Move to the next game in a Fearless BO3: add current picks to series bans and reset game."""
    # add both teams' picks to the running banlist
    new_bans = set(st.session_state.picks['P1'] + st.session_state.picks['P2'])
    st.session_state.series_banned |= new_bans
    st.session_state.series_game += 1
    reset_game_from_series()


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
        "remaining_heroes": remaining,
        # Fearless BO3
        "fearless_bo3": st.session_state.fearless_bo3,
        "series_game": st.session_state.series_game,
        "series_banned": sorted(list(st.session_state.series_banned)),
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
        if img_bytes: st.image(img_bytes, width=48)
        else:         st.image(get_placeholder_portrait(), width=48)
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

start_series_if_needed()

role_label = {"master":"Master", "p1":"Player 1", "p2":"Player 2"}[st.session_state.role]
#ync_label = "ON" if st.session_state.get("_live_sync") else "OFF"
st.caption(f"Role: **{role_label}**") # • Live sync: **{sync_label}**")

# --- Poll for lobby changes (compare full state, not just step) ---
if st.session_state.get("lobby_id"):
    now = time.time()
    last_poll = st.session_state.get("_last_poll_ts", 0)
    if now - last_poll > 1.0:
        row = sb_load_draft(st.session_state["lobby_id"])
        st.session_state["_last_poll_ts"] = now
        if row and row.get("state"):
            remote_state = row["state"]
            local_state  = snapshot_state()
            if remote_state != local_state:
                apply_snapshot(remote_state)
                # UI will reflect changes automatically

def _can_act_this_turn(step_player: str) -> bool:
    role = st.session_state.get("role", "master")
    if role == "master":
        return True
    return (role == "p1" and step_player == "P1") or (role == "p2" and step_player == "P2")

# ====== TOP: Current Step Interaction ======
if st.session_state.step_idx < len(DRAFT_SEQUENCE):
    step = DRAFT_SEQUENCE[st.session_state.step_idx]
    can_act = _can_act_this_turn(step.player)
    if not can_act:
        human = {"P1":"Player 1", "P2":"Player 2"}[step.player]
        st.info(f"It's {human}'s turn. Your role is **{st.session_state.role.upper()}**.")

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
        col_ok, col_sync = st.columns([1, 1])
        with col_ok:
            confirm = st.button("Confirm", type="primary", disabled=not can_act, key=f"ok_{st.session_state.step_idx}")
        with col_sync:
            sync_btn = st.button("Sync", key=f"sync_{st.session_state.step_idx}")
        
        if sync_btn:
            sync_now()
        
        if confirm and choice:
            ok = apply_action(step, choice)
            if ok:
                if st.session_state.get("lobby_id"):
                    sb_save_draft(st.session_state["lobby_id"])
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
        col_ok, col_sync = st.columns([1, 1])
        with col_ok:
            confirm = st.button("Confirm God", type="primary", disabled=not can_act, key=f"okgod_{st.session_state.step_idx}")
        with col_sync:
            sync_btn = st.button("Sync", key=f"syncgod_{st.session_state.step_idx}")
            
        if sync_btn:
            sync_now()
            
        if confirm and choice:
            if st.session_state.get('_enforce_unique_gods', True):
                if choice not in st.session_state.available_gods:
                    picked = [g for g in st.session_state.gods_picked.values() if g]
                    st.session_state.available_gods = [g for g in st.session_state.gods if g not in picked]
            ok = apply_action(step, choice)
            if ok:
                if st.session_state.get("lobby_id"):
                    sb_save_draft(st.session_state["lobby_id"])
                st.toast(f"{step.player} picked {choice}", icon="✅")
                st.rerun()

else:
    st.header("Draft Complete ✅")

    # If Fearless BO3 and not at game 3, allow advancing
    if st.session_state.fearless_bo3 and st.session_state.series_game < 3:
        st.success(f"Game {st.session_state.series_game} complete.")
        if st.button("➡️ Start Next Game"):
            advance_to_next_game()
            if st.session_state.get("lobby_id"):
                sb_save_draft(st.session_state["lobby_id"])
            st.rerun()
    elif st.session_state.fearless_bo3 and st.session_state.series_game >= 3:
        st.info("Series complete (Game 3 done). You can reset the series in the sidebar.")

# ====== SIDEBAR ======
with st.sidebar:
    st.header("⚙️ Setup")
    st.caption("Built-in heroes, classes, affiliations, and portraits are loaded.")
    enforce_unique = st.checkbox("Enforce unique gods (no duplicates)", value=True)
    st.session_state._enforce_unique_gods = enforce_unique
        # --- Fearless BO3 toggle ---
    st.markdown("---")
    st.subheader("Series Mode")
    st.session_state.fearless_bo3 = st.checkbox(
        "Fearless BO3",
        value=st.session_state.fearless_bo3,
        help="Play a best-of-3 series where any hero picked in a game is automatically banned for the remainder of the series."
    )

    # Show series status when enabled
    if st.session_state.fearless_bo3:
        st.caption(f"Game {st.session_state.series_game} of 3 • Series bans: {len(st.session_state.series_banned)}")


    # --- Lobby (Supabase) ---
    st.markdown("---")
    st.subheader("Lobby (shared)")
    lobby_code = st.text_input("Lobby code", value=st.session_state.get("lobby_code", ""))

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Join / Create"):
            if lobby_code.strip():
                lobby_id = sb_upsert_lobby(lobby_code)
                if lobby_id:
                    st.session_state["lobby_code"] = lobby_code.strip().lower()
                    st.session_state["lobby_id"] = lobby_id
                    row = sb_load_draft(lobby_id)
                    if not row:
                        sb_save_draft(lobby_id)
                        row = sb_load_draft(lobby_id)
                    if row and row.get("state"):
                        apply_snapshot(row["state"])
                    #else:
                    #    sb_save_draft(lobby_id)
                    st.toast("Lobby ready.", icon="🟢")
                    st.rerun()
    with c2:
        if st.button("Save now"):
            if st.session_state.get("lobby_id"):
                sb_save_draft(st.session_state["lobby_id"])
                st.toast("Saved to lobby.", icon="💾")

        # Add a simple manual sync
        if st.button("Sync now"):
            sync_now()
    with colB:
        if st.button("♻️ Full Reset"):  # full reset (series as well)
            reset_draft()
            st.session_state.fearless_bo3 = st.session_state.fearless_bo3  # no-op, keep toggle state
            st.session_state.series_game = 1
            st.session_state.series_banned = set()
            if st.session_state.get("lobby_id"):
                sb_save_draft(st.session_state["lobby_id"])

    # Optional: reset only the current game (preserve series bans)
    if st.session_state.fearless_bo3 and st.button("↺ Reset Current Game"):
        reset_game_from_series()
        if st.session_state.get("lobby_id"):
            sb_save_draft(st.session_state["lobby_id"])


    # --- Role & Live sync ---
    st.markdown("---")
    st.subheader("Role & Sync")

    st.session_state.role = st.radio(
        "Who are you?",
        options=["master", "p1", "p2"],
        format_func=lambda x: {"master":"Master (both sides)", "p1":"Player 1", "p2":"Player 2"}[x],
        horizontal=False,
        index=["master","p1","p2"].index(st.session_state.role)
    )

    # Live sync makes the app auto-rerun (poll & refresh) ~every 0.8s
    # Safe default: turn on automatically once you join a lobby
    #default_live = bool(st.session_state.get("lobby_id"))
    #if "_live_sync_initialized" not in st.session_state:
    #    st.session_state._live_sync = default_live
    #    st.session_state._live_sync_initialized = True

    #st.session_state._live_sync = st.checkbox(
    #    "🔄 Live sync (auto refresh)",
    #    value=st.session_state._live_sync,
    #    help="Auto-reruns the app about once per second to pick up remote changes."
    #)

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
        if st.button("↩️ Undo"):
            undo_last()
            if st.session_state.get("lobby_id"):
                sb_save_draft(st.session_state["lobby_id"])
    with colB:
        if st.button("♻️ Full Reset"):
            reset_draft()
            if st.session_state.get("lobby_id"):
                sb_save_draft(st.session_state["lobby_id"])

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
        for h in st.session_state.picks['P1']:
            render_hero_chip(h)
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
        for h in st.session_state.picks['P2']:
            render_hero_chip(h)
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
            st.image(get_placeholder_portrait(), use_container_width=True)
        else:
            if h in picked_or_banned:
                gb = grayscale_bytes(b) or b
                st.image(gb, use_container_width=True)
            else:
                st.image(b, use_container_width=True)

# Final summary when complete
if st.session_state.step_idx >= len(DRAFT_SEQUENCE):
    st.markdown("---")
    st.subheader("Summary JSON")
    st.code(export_state(), language='json')

# ---- Live sync auto-rerun (last line of the script) ----
#f st.session_state.get("lobby_id") and st.session_state.get("_live_sync", False):
    # Keep this modest to avoid burning CPU; 0.6–1.0s is a good range.
 #  time.sleep(0.8)  # ms
  # st.experimental_rerun()

## currently disabled