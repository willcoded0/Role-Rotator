# role_moderator.py
# Requires:
#   pip install -U discord.py python-dotenv tzdata aiohttp
#
# Features:
# - Scans the last 24 hours of messages
# - Prefilter with keyword/emoji scoring (cheap)
# - AI scoring (OpenAI) with batching/backoff/budget + cache
#   * If OPENAI_API_KEY is missing or rate-limited, falls back gracefully
# - Highest total gets the role
# - Up to 2 Honorary Mentions (score ≥ threshold) shown with scores
# - Winner announcement includes top 2 flagged messages (with links)
# - Uses .env (or --envfile) for secrets/IDs

import os, json, asyncio, sys, re, datetime as dt, logging
from pathlib import Path
import argparse
import discord
import aiohttp
from dotenv import load_dotenv, dotenv_values

# ---------- Timezone (America/Chicago) ----------
try:
    from zoneinfo import ZoneInfo
    tz = ZoneInfo("Central Time")
except Exception:
    tz = dt.timezone(dt.timedelta(hours=-6))  # fallback (no DST)

# ---------- Load env ----------
parser = argparse.ArgumentParser()
parser.add_argument("--envfile", default=".env", help="Env file to load (default .env)")
args, _ = parser.parse_known_args()

ENV_PATH = Path(__file__).with_name(".env")
load_dotenv(dotenv_path=ENV_PATH, override=True)

TOKEN = os.getenv("DISCORD_TOKEN")
GUILD_ID = int(os.getenv("GUILD_ID", "0") or "0")
ROLE_ID = int(os.getenv("ROLE_ID", "0") or "0")
ANNOUNCE_CHANNEL_ID = int(os.getenv("ANNOUNCE_CHANNEL_ID", "0") or "0")
STATE_FILE = os.getenv("STATE_FILE", "rotation_state.json")

# OpenAI (optional)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
AI_MAX_PER_MSG = 5  # cap per-message score

# ---------- Behavior / schedule ----------
RUN_HOUR, RUN_MINUTE = 12, 0       # 12:00 PM Central daily
RUN_IMMEDIATELY = True             # run once at startup (testing)
LOOKBACK_HOURS = 24
PER_USER_MSG_CAP = 50
HONORARY_THRESHOLD = 10
HONORARY_MAX = 2

# Channel filter (empty = all readable text channels)
CHANNEL_IDS: set[int] = set()

# AI controls
AI_ENABLED = bool(OPENAI_API_KEY)
AI_MESSAGE_BUDGET = int(os.getenv("AI_MESSAGE_BUDGET", "200"))
BATCH_SIZE = int(os.getenv("AI_BATCH_SIZE", "5"))
BASE_SLEEP = float(os.getenv("AI_BASE_SLEEP", "3.0"))
MAX_RETRIES = int(os.getenv("AI_MAX_RETRIES", "6"))
AI_CACHE_FILE = os.getenv("AI_CACHE_FILE", "ai_cache.json")

# ---------- Sanity check ----------
if not TOKEN or not GUILD_ID or not ROLE_ID or not ANNOUNCE_CHANNEL_ID:
    print("Set DISCORD_TOKEN, GUILD_ID, ROLE_ID, and ANNOUNCE_CHANNEL_ID in your env file.")
    print("Loaded env from:", ENV_PATH)
    print("Keys seen:", list(dotenv_values(ENV_PATH).keys()))
    sys.exit(1)

# ---------- Intents & logging ----------
intents = discord.Intents.none()
intents.guilds = True
intents.members = True
intents.message_content = True

discord.utils.setup_logging(level=logging.INFO)
log = logging.getLogger("rotator")

# ---------- Helpers ----------
def load_state():
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            state = json.load(f)
            if isinstance(state.get("last_run_date"), str) and state["last_run_date"].lower() == "null":
                state["last_run_date"] = None
            return state
    except Exception:
        return {"last_run_date": None, "current_holder_id": None}

def save_state(state):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)

def load_cache() -> dict:
    try:
        with open(AI_CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_cache(cache: dict):
    try:
        with open(AI_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
    except Exception:
        pass

def jump_url(guild_id: int, channel_id: int, message_id: int) -> str:
    return f"https://discord.com/channels/{guild_id}/{channel_id}/{message_id}"

def safe_snippet(s: str, limit: int = 200) -> str:
    if not s:
        return ""
    s = s.replace("\n", " ").replace("@", "@\u200b").strip()
    return (s[:limit-3] + "...") if len(s) > limit else s

def can_manage(me: discord.Member, member: discord.Member, role: discord.Role) -> bool:
    return (
        me.guild_permissions.manage_roles and
        me.top_role > role and
        me.top_role > member.top_role
    )

# ---------- Keyword/emoji scoring ----------
PHRASES = [
    "shut up","stupid","dumb","annoying","toxic","trash",
    "no one cares","idiot","loser","worthless",
    "spam","stop spamming","boring","get lost","hate this",
]
EMOJIS = ["🤬","😡","👎","💩"]
REGEXES = [
    re.compile(r"(?!\bnot\b)\b(bad|terrible|awful|hate|worst)\b", re.I),
]

def count_occurrences(text: str, *, nsfw_channel: bool) -> int:
    if not text:
        return 0
    points = 0
    for p in PHRASES:
        points += len(re.findall(re.escape(p), text, flags=re.I))
    for e in EMOJIS:
        points += text.count(e)
    for rgx in REGEXES:
        points += len(rgx.findall(text))
    return max(points, 0)

# ---------- AI scoring ----------
TONE_WEIGHTS = {
    "toxic": 1.3,
    "rude": 1.0,
    "spammy": 1.0,
    "disruptive": 1.2,
}

SYSTEM_PROMPT = (
    "You are a rater that scores Discord messages for these tones: "
    "toxic, rude, spammy, disruptive. "
    "Return JSON with integer scores 0-5 for each tone, no extra text."
)

def _combine_tones(tone_scores: dict, *, nsfw: bool) -> int:
    def clamp(v): return int(max(0, min(AI_MAX_PER_MSG, int(v))))
    ts = {
        "toxic": clamp(tone_scores.get("toxic", 0)),
        "rude": clamp(tone_scores.get("rude", 0)),
        "spammy": clamp(tone_scores.get("spammy", 0)),
        "disruptive": clamp(tone_scores.get("disruptive", 0)),
    }
    s = 0.0
    for k, w in TONE_WEIGHTS.items():
        s += w * ts.get(k, 0)
    return int(max(0, min(AI_MAX_PER_MSG, round(s))))

def _chunk_list(items, size):
    for i in range(0, len(items), size):
        yield items[i:i+size]

async def _post_with_retries(session, url, headers, body):
    delay = BASE_SLEEP
    for _attempt in range(MAX_RETRIES):
        try:
            async with session.post(url, headers=headers, json=body, timeout=75) as r:
                if r.status == 429:
                    ra = r.headers.get("Retry-After")
                    wait = float(ra) if ra else delay
                    await asyncio.sleep(wait)
                    delay = min(delay * 2, 15)
                    continue
                r.raise_for_status()
                return await r.json()
        except aiohttp.ClientResponseError as e:
            if e.status == 429:
                await asyncio.sleep(delay)
                delay = min(delay * 2, 15)
                continue
            raise
    async with session.post(url, headers=headers, json=body, timeout=75) as r:
        r.raise_for_status()
        return await r.json()

async def ai_score_messages_batch(indexed_batch, batch, session) -> dict[int, int]:
    numbered = []
    for i, (text, _nsfw) in indexed_batch:
        t = (text or "").strip()
        if len(t) > 800:
            t = t[:797] + "…"
        numbered.append(f"{i}: {t}")

    user_prompt = (
        "Rate each message (by index) for tones toxic, rude, spammy, disruptive.\n"
        "Return STRICT JSON array; one object per index with integer scores 0-5.\n"
        "Format:\n"
        "[{\"index\":0,\"toxic\":2,\"rude\":1,\"spammy\":0,\"disruptive\":3}, ...]\n\n"
        "Messages:\n" + "\n".join(numbered)
    )

    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    body = {
        "model": AI_MODEL,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
    }

    out: dict[int, int] = {}
    try:
        data = await _post_with_retries(session, url, headers, body)
        content = data["choices"][0]["message"]["content"]
        arr = json.loads(content)
        for obj in arr:
            i = int(obj.get("index", -1))
            if 0 <= i < len(batch):
                score = _combine_tones(obj, nsfw=False)
                out[i] = int(max(0, min(AI_MAX_PER_MSG, score)))
    except Exception as e:
        logging.warning(f"[AI] batch failed: {e}")
    return out

async def ai_score_messages(batch: list[tuple[str, bool]]) -> list[int]:
    if not AI_ENABLED or not batch:
        return [0] * len(batch)

    out = [0] * len(batch)
    indexed = list(enumerate(batch))

    if len(indexed) > AI_MESSAGE_BUDGET:
        indexed = indexed[-AI_MESSAGE_BUDGET:]

    async with aiohttp.ClientSession() as session:
        for chunk in _chunk_list(indexed, BATCH_SIZE):
            scores = await ai_score_messages_batch(chunk, batch, session)
            for i, val in scores.items():
                out[i] = val
            await asyncio.sleep(BASE_SLEEP)
    return out

# ---------- Client ----------
class RoleRotator(discord.Client):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.state = load_state()
        self.bg_task = None
        self.ai_cache = load_cache()

    async def on_ready(self):
        try:
            print(f"Logged in as {self.user} ({self.user.id})")
            guild = self.get_guild(GUILD_ID) or await self.fetch_guild(GUILD_ID)
            me = guild.get_member(self.user.id) or await guild.fetch_member(self.user.id)
            role = guild.get_role(ROLE_ID)
            if not role:
                print(f"[ERROR] ROLE_ID {ROLE_ID} not found in this guild.")
                return

            if RUN_IMMEDIATELY:
                print("[INFO] Running an immediate test rotation…")
                await self.daily_rotation_job()
                now = dt.datetime.now(tz)
                self.state["last_run_date"] = now.date().isoformat()
                save_state(self.state)

            self.bg_task = asyncio.create_task(self.scheduler_loop())
        except Exception as e:
            import traceback
            print("[FATAL] on_ready crashed:", e)
            traceback.print_exc()

    async def scheduler_loop(self):
        while True:
            now = dt.datetime.now(tz)
            should_run = (
                now.hour == RUN_HOUR and
                now.minute == RUN_MINUTE and
                self.state.get("last_run_date") != now.date().isoformat()
            )
            if should_run:
                try:
                    await self.daily_rotation_job()
                    self.state["last_run_date"] = now.date().isoformat()
                    save_state(self.state)
                except Exception as e:
                    print(f"[ERROR] daily_rotation_job failed: {e}")
            await asyncio.sleep(30)

    async def daily_rotation_job(self):
        guild = self.get_guild(GUILD_ID) or await self.fetch_guild(GUILD_ID)
        role = guild.get_role(ROLE_ID)
        if not role:
            print("[ERROR] Configured ROLE_ID not found.")
            return
        print("[INFO] Daily rotation starting…")

        end = dt.datetime.now(tz)
        start = end - dt.timedelta(hours=LOOKBACK_HOURS)

        msg_refs: list[tuple[int, str, float, int, int, bool]] = []

        for ch in guild.text_channels:
            if CHANNEL_IDS and ch.id not in CHANNEL_IDS:
                continue
            perms = ch.permissions_for(guild.me)
            if not (perms.view_channel and perms.read_message_history):
                continue
            try:
                async for msg in ch.history(after=start, before=end, oldest_first=False, limit=None):
                    author = msg.author
                    if not author or getattr(author, "bot", False):
                        continue
                    raw = getattr(msg, "content", "") or ""
                    msg_refs.append((author.id, raw, msg.created_at.timestamp(), ch.id, msg.id, ch.is_nsfw()))
            except (discord.Forbidden, discord.HTTPException):
                pass

        user_total: dict[int, int] = {}
        user_msgs: dict[int, list[dict]] = {}

        ai_batch_inputs: list[tuple[str, bool]] = []
        ai_batch_indices: list[int] = []

        for i, (uid, raw, ts, ch_id, m_id, is_nsfw) in enumerate(msg_refs):
            kw_pts = count_occurrences(raw, nsfw_channel=is_nsfw)
            cached = self.ai_cache.get(str(m_id))
            ai_pts = None
            if cached is not None:
                ai_pts = int(cached)

            if ai_pts is not None:
                pts = max(ai_pts, kw_pts)
            else:
                if kw_pts > 0 and AI_ENABLED:
                    ai_batch_inputs.append((raw, is_nsfw))
                    ai_batch_indices.append(i)
                    pts = None
                else:
                    pts = kw_pts

            if pts is not None and pts > 0:
                entry = {
                    "url": jump_url(guild.id, ch_id, m_id),
                    "content": safe_snippet(raw),
                    "points": pts,
                    "created_at": ts,
                }
                lst = user_msgs.setdefault(uid, [])
                if len(lst) < PER_USER_MSG_CAP:
                    lst.append(entry)
                user_total[uid] = user_total.get(uid, 0) + pts

        if ai_batch_inputs and AI_ENABLED:
            if len(ai_batch_inputs) > AI_MESSAGE_BUDGET:
                ai_batch_inputs = ai_batch_inputs[-AI_MESSAGE_BUDGET:]
                ai_batch_indices = ai_batch_indices[-AI_MESSAGE_BUDGET:]

            ai_scores = await ai_score_messages(ai_batch_inputs)

            for (score, idx_in_refs) in zip(ai_scores, ai_batch_indices):
                uid, raw, ts, ch_id, m_id, is_nsfw = msg_refs[idx_in_refs]
                if score > 0:
                    self.ai_cache[str(m_id)] = int(score)
                    entry = {
                        "url": jump_url(guild.id, ch_id, m_id),
                        "content": safe_snippet(raw),
                        "points": score,
                        "created_at": ts,
                    }
                    lst = user_msgs.setdefault(uid, [])
                    if len(lst) < PER_USER_MSG_CAP:
                        lst.append(entry)
                    user_total[uid] = user_total.get(uid, 0) + score

        save_cache(self.ai_cache)

        if not user_total:
            print("[INFO] No scored messages in the last 24 hours.")
            self.state["current_holder_id"] = None
            save_state(self.state)
            print("[INFO] Rotation complete.")
            return

        members = {m.id: m async for m in guild.fetch_members(limit=None)}
        me = guild.get_member(self.user.id) or await guild.fetch_member(self.user.id)

        sorted_users = sorted(user_total.items(), key=lambda kv: (-kv[1], kv[0]))
        winner_id = None
        winner_total = 0
        for uid, total in sorted_users:
            member = members.get(uid) or await guild.fetch_member(uid)
            if not member:
                continue
            if role in member.roles:
                continue
            if not can_manage(me, member, role):
                continue
            winner_id = uid
            winner_total = total
            break

        if winner_id is None:
            print("[INFO] No manageable winner found.")
            self.state["current_holder_id"] = None
            save_state(self.state)
            print("[INFO] Rotation complete.")
            return

        honorary = [
            (uid, user_total.get(uid, 0))
            for uid, _ in sorted_users
            if uid != winner_id and user_total.get(uid, 0) >= HONORARY_THRESHOLD
        ][:HONORARY_MAX]

        prev_id = self.state.get("current_holder_id")
        if prev_id and prev_id in members:
            try:
                await members[prev_id].remove_roles(role, reason="Daily rotation end")
            except (discord.Forbidden, discord.HTTPException):
                pass

        try:
            for m in list(role.members):
                if m.id != prev_id:
                    await m.remove_roles(role, reason="Rotation cleanup")
        except (discord.Forbidden, discord.HTTPException):
            pass

        try:
            await members[winner_id].add_roles(role, reason="Daily rotation (AI score/keywords)")
            self.state["current_holder_id"] = winner_id
        except (discord.Forbidden, discord.HTTPException):
            self.state["current_holder_id"] = None
            save_state(self.state)
            print("[INFO] Rotation complete.")
            return
