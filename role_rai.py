# role_rotator.py
# Requires:
#   pip install -U discord.py python-dotenv tzdata aiohttp
#
# Features:
# - Scans the last 24 hours of messages
# - Prefilter with keyword/emoji scoring (cheap)
# - AI tone scoring (OpenAI) with batching/backoff/budget + cache
#   * If OPENAI_API_KEY is missing or rate-limited, falls back gracefully
# - Highest total gets the role
# - Up to 2 Honorary Mentions (score ≥ threshold) shown with scores
# - Winner announcement includes top 2 scoring messages (with links)
# - Uses .env (or --envfile) for secrets/IDs

import os, json, asyncio, sys, random, re, datetime as dt, logging
from pathlib import Path
import argparse
import discord
import aiohttp

# ---------- Timezone (America/Chicago) ----------
try:
    from zoneinfo import ZoneInfo
    tz = ZoneInfo("Central Time")
except Exception:
    tz = dt.timezone(dt.timedelta(hours=-6))  # fallback (no DST)

# ---------- Load env ----------
from dotenv import load_dotenv, dotenv_values

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
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # if absent, we fall back to keyword scoring
AI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
AI_MAX_PER_MSG = 5  # cap per-message score

# ---------- Behavior / schedule ----------
RUN_HOUR, RUN_MINUTE = 12, 0       # 12:00 PM Central daily
RUN_IMMEDIATELY = True           # set True to run once at startup (testing)
LOOKBACK_HOURS = 24                # always scan last 24h
PER_USER_MSG_CAP = 50              # store up to 50 scored messages per user
HONORARY_THRESHOLD = 10            # score needed to be an honorary mention
HONORARY_MAX = 2                   # max # of honorary mentions listed
ONLY_COUNT_LEWD_IN_NSFW = False    # True => count lewd-ish cues only in NSFW channels

# Channel filter (empty = all readable text channels)
CHANNEL_IDS: set[int] = set()

# AI controls
AI_ENABLED = bool(OPENAI_API_KEY)
AI_MESSAGE_BUDGET = int(os.getenv("AI_MESSAGE_BUDGET", "200"))  # max msgs/day sent to AI
BATCH_SIZE = int(os.getenv("AI_BATCH_SIZE", "5"))               # msgs per API request
BASE_SLEEP = float(os.getenv("AI_BASE_SLEEP", "3.0"))           # seconds between batches
MAX_RETRIES = int(os.getenv("AI_MAX_RETRIES", "6"))             # retries on 429
AI_CACHE_FILE = os.getenv("AI_CACHE_FILE", "ai_cache.json")     # message_id -> score

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
intents.message_content = True  # enable in Dev Portal

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

# ---------- Keyword/emoji scoring (prefilter & fallback) ----------
PHRASES = [
    # hedges / hesitancy
    "sort of","idk","i dunno","i guess","i think",
    "i’m not sure","im not sure","i am not sure",
    "if that’s okay","if thats okay","if that is okay","if you want","if you’d like","if youd like",
    # apologies
    "sorry","apologize","apologies","my bad",
    # vulnerable
    "i feel","i’m anxious","im anxious","i’m nervous","im nervous",
    "i’m insecure","im insecure","i’m scared","im scared","i’m overwhelmed","im overwhelmed",
    # submissive / deferential
    "you decide","up to you","whatever you prefer","i don’t mind","i dont mind","tell me what to do",
    "please","pls",
    # lewd-ish (light)
    "nsfw","lewd","horny","thirsty","down bad","spicy","risqué","risque","dm me","slide in","sussy",
]
LEWD_SET = {"nsfw","lewd","horny","thirsty","down bad","spicy","risqué","risque","dm me","slide in","sussy"}

EMOJIS = ["🥺","🙏","🙇","🙈","😏","😈","🍑","🍆","🌶️","💦","👀"]

REGEXES = [
    re.compile(r"\.\.\.+"),               # ellipses
    re.compile(r"\b(simp|thirst|thirsty|horny|lewd|nsfw|down\s*bad|spicy)\b", re.I),
]

def count_occurrences(text: str, *, nsfw_channel: bool) -> int:
    """Return fallback points for a single message; +1 per phrase/emoji/regex match."""
    if not text:
        return 0
    points = 0
    # phrases
    for p in PHRASES:
        if ONLY_COUNT_LEWD_IN_NSFW and (p in LEWD_SET) and (not nsfw_channel):
            continue
        points += len(re.findall(re.escape(p), text, flags=re.I))
    # emojis
    for e in EMOJIS:
        points += text.count(e)
    # regex cues
    for rgx in REGEXES:
        if ONLY_COUNT_LEWD_IN_NSFW and rgx is REGEXES[-1] and not nsfw_channel:
            continue
        points += len(rgx.findall(text))
    return max(points, 0)

# ---------- AI scoring (OpenAI) with batching/backoff/budget/cache ----------
TONE_WEIGHTS = {
    "vulnerable": 1.0,
    "shy": 1.0,
    "submissive": 1.3,
    "lewd": 1.2,
}

SYSTEM_PROMPT = (
    "You are a rater that scores Discord messages for these tones: "
    "vulnerable, shy, submissive, lewd. "
    "Return JSON with integer scores 0-5 for each tone, no extra text."
)

def _combine_tones(tone_scores: dict, *, nsfw: bool) -> int:
    """Weighted sum, 0..AI_MAX_PER_MSG, optionally zero out lewd if not NSFW."""
    def clamp(v): return int(max(0, min(AI_MAX_PER_MSG, int(v))))
    ts = {
        "vulnerable": clamp(tone_scores.get("vulnerable", 0)),
        "shy": clamp(tone_scores.get("shy", 0)),
        "submissive": clamp(tone_scores.get("submissive", 0)),
        "lewd": clamp(tone_scores.get("lewd", 0)),
    }
    if ONLY_COUNT_LEWD_IN_NSFW and not nsfw:
        ts["lewd"] = 0
    s = 0.0
    for k, w in TONE_WEIGHTS.items():
        s += w * ts.get(k, 0)
    return int(max(0, min(AI_MAX_PER_MSG, round(s))))

def _chunk_list(items, size):
    for i in range(0, len(items), size):
        yield items[i:i+size]

async def _post_with_retries(session, url, headers, body):
    """POST with basic retry on 429, respecting Retry-After if present."""
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
    # final attempt
    async with session.post(url, headers=headers, json=body, timeout=75) as r:
        r.raise_for_status()
        return await r.json()

async def ai_score_messages_batch(indexed_batch, batch, session) -> dict[int, int]:
    """
    Score one chunk (indexed subset). Returns dict[index -> score].
    """
    # Build prompt with numbered messages
    numbered = []
    for i, (text, _nsfw) in indexed_batch:
        t = (text or "").strip()
        if len(t) > 800:
            t = t[:797] + "…"
        numbered.append(f"{i}: {t}")

    user_prompt = (
        "Rate each message (by index) for tones vulnerable, shy, submissive, lewd.\n"
        "Return STRICT JSON array; one object per index with integer scores 0-5.\n"
        "Format:\n"
        "[{\"index\":0,\"vulnerable\":2,\"shy\":1,\"submissive\":3,\"lewd\":0}, ...]\n\n"
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
                nsfw = batch[i][1]
                score = _combine_tones(obj, nsfw=nsfw)
                out[i] = int(max(0, min(AI_MAX_PER_MSG, score)))
    except Exception as e:
        logging.warning(f"[AI] batch failed: {e}")
        # leave out empty -> fallback later
    return out

async def ai_score_messages(batch: list[tuple[str, bool]]) -> list[int]:
    """
    Batch-score messages with OpenAI. Each message gets a 0..AI_MAX_PER_MSG int.
    `batch`: list of (text, is_nsfw)
    """
    if not AI_ENABLED or not batch:
        return [0] * len(batch)

    out = [0] * len(batch)
    indexed = list(enumerate(batch))  # [(i, (text, nsfw)), ...]

    # Budget newest N
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
        self.ai_cache = load_cache()  # {str(message_id): int_score}

    async def on_ready(self):
        try:
            print(f"Logged in as {self.user} ({self.user.id})")
            guild = self.get_guild(GUILD_ID) or await self.fetch_guild(GUILD_ID)
            me = guild.get_member(self.user.id) or await guild.fetch_member(self.user.id)
            role = guild.get_role(ROLE_ID)
            if not role:
                print(f"[ERROR] ROLE_ID {ROLE_ID} not found in this guild.")
                return

            print(f"[DEBUG] bot_top={me.top_role} (pos {me.top_role.position}), "
                  f"target_role={role} (pos {role.position})")
            if not me.top_role > role:
                print("[WARN] Bot's highest role must be ABOVE the rotating role (drag it higher).")

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

        # ----- time window -----
        end = dt.datetime.now(tz)
        start = end - dt.timedelta(hours=LOOKBACK_HOURS)

        # ----- collect messages -----
        # Store: (uid, content, ts, ch_id, msg_id, nsfw)
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
                continue

        # ----- prefilter with keyword scoring; build AI batch & totals -----
        user_total: dict[int, int] = {}
        user_msgs: dict[int, list[dict]] = {}

        # Build AI batch from messages that had at least some signal via keywords
        ai_batch_inputs: list[tuple[str, bool]] = []
        ai_batch_indices: list[int] = []  # index into msg_refs

        for i, (uid, raw, ts, ch_id, m_id, is_nsfw) in enumerate(msg_refs):
            kw_pts = count_occurrences(raw, nsfw_channel=is_nsfw)

            # cache: if we already have an AI score for this message_id, use it
            cached = self.ai_cache.get(str(m_id))
            ai_pts = None
            if cached is not None:
                ai_pts = int(cached)

            # decide final points for this message
            if ai_pts is not None:
                pts = max(ai_pts, kw_pts)  # prefer AI, but don't be worse than keywords
            else:
                if kw_pts > 0 and AI_ENABLED:
                    # candidate for AI batch; we will add later after AI call
                    ai_batch_inputs.append((raw, is_nsfw))
                    ai_batch_indices.append(i)
                    pts = None  # pending
                else:
                    pts = kw_pts  # zero or small signal, no AI

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

        # ----- run AI on the selected subset (budgeted & batched) -----
        if ai_batch_inputs and AI_ENABLED:
            # Apply budget: keep newest N
            if len(ai_batch_inputs) > AI_MESSAGE_BUDGET:
                ai_batch_inputs = ai_batch_inputs[-AI_MESSAGE_BUDGET:]
                ai_batch_indices = ai_batch_indices[-AI_MESSAGE_BUDGET:]

            ai_scores = await ai_score_messages(ai_batch_inputs)  # list[int]

            # integrate AI results back into totals, update cache
            for (score, idx_in_refs) in zip(ai_scores, ai_batch_indices):
                uid, raw, ts, ch_id, m_id, is_nsfw = msg_refs[idx_in_refs]
                if score > 0:
                    self.ai_cache[str(m_id)] = int(score)  # cache
                    # If this message was previously counted with kw_pts (it wasn't; we deferred),
                    # simply add now:
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

        # Save cache now that the run finished scoring
        save_cache(self.ai_cache)

        if not user_total:
            print("[INFO] No scored messages in the last 24 hours.")
            self.state["current_holder_id"] = None
            save_state(self.state)
            print("[INFO] Rotation complete.")
            return

        # ----- choose highest-scoring manageable member -----
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
                print(f"[SKIP] cannot manage {member.display_name} (total={total})")
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

        # Honorary mentions (exclude winner), max 2
        honorary = [
            (uid, user_total.get(uid, 0))
            for uid, _ in sorted_users
            if uid != winner_id and user_total.get(uid, 0) >= HONORARY_THRESHOLD
        ][:HONORARY_MAX]

        # ----- remove role from previous holder & cleanup -----
        prev_id = self.state.get("current_holder_id")
        if prev_id and prev_id in members:
            try:
                await members[prev_id].remove_roles(role, reason="Daily rotation end")
                print(f"[INFO] Removed role from {members[prev_id].display_name}")
            except (discord.Forbidden, discord.HTTPException) as e:
                print(f"[WARN] Could not remove role from previous holder: {e}")

        # remove strays
        try:
            for m in list(role.members):
                if m.id != prev_id:
                    await m.remove_roles(role, reason="Rotation cleanup")
                    print(f"[INFO] Cleaned stray role on {m.display_name}")
        except (discord.Forbidden, discord.HTTPException):
            pass

        # ----- assign to winner -----
        try:
            await members[winner_id].add_roles(role, reason="Daily rotation (AI score/keywords)")
            self.state["current_holder_id"] = winner_id
            print(f"[INFO] Assigned role to {members[winner_id].display_name} (total={winner_total})")
        except discord.Forbidden:
            print("[ERROR] Lack permissions to add role.")
            self.state["current_holder_id"] = None
            save_state(self.state)
            print("[INFO] Rotation complete.")
            return
        except discord.HTTPException:
            print("[ERROR] HTTP error adding role.")
            self.state["current_holder_id"] = None
            save_state(self.state)
            print("[INFO] Rotation complete.")
            return

        # ----- announce -----
        announce_ch = guild.get_channel(ANNOUNCE_CHANNEL_ID)
        if announce_ch:
            try:
                now = dt.datetime.now(tz)
                yesterday = (now - dt.timedelta(days=1)).strftime("%b %d, %Y")

                # winner's top 2 messages by points (then newest first)
                top_msgs = sorted(
                    user_msgs.get(winner_id, []),
                    key=lambda m: (-m["points"], -m["created_at"])
                )[:2]

                quotes = ""
                for m in top_msgs:
                    if m.get("content"):
                        quotes += f"\n> {m['content']}\n{m['url']}"
                    else:
                        quotes += f"\n{m['url']}"

                # honorary mentions (mention + score)
                honorary_line = ""
                if honorary:
                    parts = []
                    for uid, total in honorary:
                        member = members.get(uid) or await guild.fetch_member(uid)
                        if member:
                            parts.append(f"{member.mention} (**{total}**)")
                    if parts:
                        honorary_line = "\n\n**Honorary Mentions:** " + ", ".join(parts)

                msg = (
                    f"{members[winner_id].mention} was the most submissive {role.mention} "
                    f" with **{winner_total}** vulnerable actions on "
                    f"{yesterday} 👀 "
                    f"You have until {RUN_HOUR:02d}:{RUN_MINUTE:02d} PM Central "
                    f"tomorrow to give your speech as {role.mention}."
                    f"\n\n**Notable Promiscuous Quotes**{quotes}"
                    f"{honorary_line}"
                )

                await announce_ch.send(msg)

            except discord.Forbidden:
                print("[WARN] Cannot send message in announce channel (missing Send Messages).")
            except discord.HTTPException as e:
                print(f"[WARN] Failed to send announce message: {e}")
        else:
            print(f"[WARN] Announce channel {ANNOUNCE_CHANNEL_ID} not found.")

        save_state(self.state)
        print("[INFO] Rotation complete.")

# ---------- main ----------
client = RoleRotator(intents=intents)

async def main():
    async with client:
        await client.start(TOKEN)

if __name__ == "__main__":
    asyncio.run(main())
