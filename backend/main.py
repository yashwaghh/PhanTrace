from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, String, Integer, Float, DateTime, ForeignKey, Text, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime
from uuid import uuid4
import os
import json
import hashlib
import numpy as np
from dotenv import load_dotenv
from pydantic import BaseModel as PydanticBase
from PIL import Image
from fastapi.staticfiles import StaticFiles

load_dotenv()

app = FastAPI(
    title="PhanTrace API",
    description="AI-Powered Sports Media Protection",
    version="1.0.0"
)

os.makedirs("uploads", exist_ok=True)
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============ DATABASE SETUP ============

DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ============ DATABASE MODELS ============

class Content(Base):
    __tablename__ = "content"

    id           = Column(String, primary_key=True, default=lambda: str(uuid4()))
    file_id      = Column(String, unique=True)
    filename     = Column(String)
    file_path    = Column(String)
    file_size_mb = Column(Float)
    owner        = Column(String)
    event        = Column(String)
    team         = Column(String)
    sport        = Column(String)
    uploaded_by  = Column(String)
    created_at   = Column(DateTime, default=datetime.utcnow)


class Fingerprint(Base):
    __tablename__ = "fingerprints"

    id             = Column(String, primary_key=True)
    content_id     = Column(String, index=True)
    visual_hash    = Column(String)           # single-frame pHash (kept for backwards compat)
    audio_hash     = Column(String)
    hash_signature = Column(String)
    # ── Phase 3 additions ──────────────────────────────────────────
    frame_hashes   = Column(Text, nullable=True)   # JSON array of per-frame pHashes
    frame_count    = Column(Integer, nullable=True) # how many frames were sampled
    video_dna      = Column(String, nullable=True)  # single fingerprint derived from all frames
    # ───────────────────────────────────────────────────────────────
    created_at     = Column(DateTime, default=datetime.utcnow)


class Detection(Base):
    __tablename__ = "detections"

    id             = Column(String, primary_key=True, default=lambda: str(uuid4()))
    fingerprint_id = Column(String, ForeignKey("fingerprints.id"))
    platform       = Column(String)
    url            = Column(String)
    title          = Column(String)
    confidence     = Column(Float)
    threat_level   = Column(String)
    thumbnail      = Column(String)
    source         = Column(String)
    detected_at    = Column(DateTime, default=datetime.utcnow)
    created_at     = Column(DateTime, default=datetime.utcnow)
    status = Column(String, default="active")  
    region = Column(String, default="Unknown") 


class KnownContent(Base):
    """Registry of known copyrighted content hashes."""
    __tablename__ = "known_content"

    id             = Column(String, primary_key=True, default=lambda: str(uuid4()))
    content_source = Column(String)
    source_url     = Column(String, unique=True)
    source_title   = Column(String)
    visual_hash    = Column(String)
    audio_hash     = Column(String)
    hash_signature = Column(String)
    # ── Phase 3 addition ──────────────────────────────────────────
    frame_hashes   = Column(Text, nullable=True)   # JSON array — for video-vs-video matching
    video_dna      = Column(String, nullable=True)
    # ─────────────────────────────────────────────────────────────
    content_type   = Column(String)
    threat_level   = Column(String, default="high")
    added_at       = Column(DateTime, default=datetime.utcnow)

class Strike(Base):
    __tablename__ = "strikes"
 
    id               = Column(String, primary_key=True, default=lambda: str(uuid4()))
    detection_id     = Column(String, ForeignKey("detections.id"), nullable=True)
    title            = Column(String)
    account          = Column(String)
    platform         = Column(String)
    status           = Column(String, default="draft")   # draft | sent | acknowledged | removed | appealed | restored
    jurisdiction     = Column(String, default="US")      # US | EU | IN | GB | FR | DE | BR | OTHER
    notice_type      = Column(String, default="DMCA")    # DMCA | DSA | IT_ACT | GENERIC
    sent_date        = Column(DateTime, nullable=True)
    removed_date     = Column(DateTime, nullable=True)
    violation_count  = Column(Integer, default=1)
    is_repeat_offender = Column(Integer, default=0)      # 0 = False, 1 = True  (SQLite-safe bool)
    created_at       = Column(DateTime, default=datetime.utcnow)
    updated_at       = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

Base.metadata.create_all(bind=engine)

# ============ HASH HELPERS ============

def hamming_distance(hash1: str, hash2: str) -> int:
    """Bit-level Hamming distance between two hex pHash strings."""
    if not hash1 or not hash2:
        return 64
    try:
        xor = int(hash1, 16) ^ int(hash2, 16)
        return bin(xor).count("1")
    except ValueError:
        return 64


def visual_similarity(hash1: str, hash2: str, hash_bits: int = 64) -> float:
    """Similarity score 0.0–100.0 (100 = identical)."""
    dist = hamming_distance(hash1, hash2)
    return round((1 - dist / hash_bits) * 100, 2)


SIMILARITY_THRESHOLDS = {
    "exact":    95.0,
    "likely":   85.0,
    "possible": 70.0,
}


def classify_match(score: float) -> str:
    if score >= SIMILARITY_THRESHOLDS["exact"]:
        return "exact_match"
    elif score >= SIMILARITY_THRESHOLDS["likely"]:
        return "likely_match"
    elif score >= SIMILARITY_THRESHOLDS["possible"]:
        return "possible_match"
    return "no_match"


# ============ PHASE 3: VIDEO FINGERPRINTING ============

def generate_phash(image_path: str, hash_size: int = 8) -> str:
    """
    Generate a perceptual hash (pHash) from an image file.
    Returns a 16-char hex string.
    """
    try:
        img    = Image.open(image_path).convert("L")
        img    = img.resize((hash_size + 1, hash_size))
        pixels = np.array(img)
        diff   = pixels[:, 1:] > pixels[:, :-1]
        bits   = "".join(["1" if d else "0" for d in diff.flatten()])
        return hashlib.md5(bits.encode()).hexdigest()[:16]
    except Exception as e:
        return f"error_{str(e)[:8]}"


def generate_phash_from_array(frame: np.ndarray, hash_size: int = 8) -> str:
    """
    Generate pHash directly from a numpy frame array (avoids writing temp files).
    frame: H x W x 3 uint8 array
    """
    try:
        img    = Image.fromarray(frame.astype("uint8")).convert("L")
        img    = img.resize((hash_size + 1, hash_size))
        pixels = np.array(img)
        diff   = pixels[:, 1:] > pixels[:, :-1]
        bits   = "".join(["1" if d else "0" for d in diff.flatten()])
        return hashlib.md5(bits.encode()).hexdigest()[:16]
    except Exception as e:
        return f"error_{str(e)[:8]}"


def generate_video_dna(frame_hashes: list[str]) -> str:
    """
    Derive a single video DNA string from a list of per-frame pHashes.
    Strategy: XOR all frame hashes together, then MD5 the result.
    This is stable — re-ordering frames changes it, but minor re-encodes don't.
    """
    if not frame_hashes:
        return "no_frames"
    try:
        # XOR all frame hashes into one integer
        combined = 0
        for h in frame_hashes:
            if h and not h.startswith("error"):
                combined ^= int(h, 16)
        # Also include frame count to prevent trivially matching single frames
        combined ^= len(frame_hashes)
        hex_combined = format(combined, "016x")
        return hashlib.md5(hex_combined.encode()).hexdigest()[:16]
    except Exception:
        return hashlib.md5("".join(frame_hashes).encode()).hexdigest()[:16]


def extract_video_fingerprint(
    video_path: str,
    sample_interval: float = 0.5,
    max_frames: int = 120,
) -> dict:
    """
    Phase 3 core function.

    Samples frames every `sample_interval` seconds across the full video,
    generates a pHash per frame, and produces a video DNA string.

    Returns:
        {
            "frame_hashes": [...],   # list of 16-char hex strings
            "frame_count":  int,
            "video_dna":    str,     # single fingerprint for the whole video
            "duration_s":   float,
            "error":        str | None
        }
    """
    try:
        from moviepy.editor import VideoFileClip

        video    = VideoFileClip(video_path)
        duration = video.duration  # seconds

        if duration is None or duration <= 0:
            video.close()
            return {"frame_hashes": [], "frame_count": 0, "video_dna": "no_duration", "duration_s": 0, "error": "Could not read video duration"}

        # Build sample timestamps
        timestamps = []
        t = 0.0
        while t < duration and len(timestamps) < max_frames:
            timestamps.append(round(t, 2))
            t += sample_interval

        frame_hashes = []
        errors       = 0

        for ts in timestamps:
            try:
                frame = video.get_frame(ts)          # numpy H x W x 3
                h     = generate_phash_from_array(frame)
                frame_hashes.append(h)
                if h.startswith("error"):
                    errors += 1
            except Exception as e:
                frame_hashes.append(f"error_{str(e)[:8]}")
                errors += 1

        video.close()

        # Remove error frames before computing DNA
        valid_hashes = [h for h in frame_hashes if not h.startswith("error")]
        video_dna    = generate_video_dna(valid_hashes)

        return {
            "frame_hashes": frame_hashes,
            "frame_count":  len(frame_hashes),
            "video_dna":    video_dna,
            "duration_s":   round(duration, 2),
            "error_frames": errors,
            "error":        None,
        }

    except ImportError:
        return {"frame_hashes": [], "frame_count": 0, "video_dna": "moviepy_missing", "duration_s": 0, "error": "moviepy not installed"}
    except Exception as e:
        return {"frame_hashes": [], "frame_count": 0, "video_dna": "extraction_error", "duration_s": 0, "error": str(e)}


def compare_video_fingerprints(
    hashes_a: list[str],
    hashes_b: list[str],
    window:   int = 5,
) -> dict:
    """
    Phase 3 comparison: sliding window matching between two frame-hash arrays.

    For each window of `window` frames in hashes_b, slide it across hashes_a
    and find the position with the highest average similarity.

    Returns:
        {
            "overall_similarity": float,   # 0–100
            "best_window_score":  float,
            "match_type":         str,
            "matched_segments":   list      # where in video_a the stolen segment lives
        }
    """
    if not hashes_a or not hashes_b:
        return {"overall_similarity": 0.0, "best_window_score": 0.0, "match_type": "no_match", "matched_segments": []}

    # Filter out error hashes
    a = [h for h in hashes_a if h and not h.startswith("error")]
    b = [h for h in hashes_b if h and not h.startswith("error")]

    if not a or not b:
        return {"overall_similarity": 0.0, "best_window_score": 0.0, "match_type": "no_match", "matched_segments": []}

    # ── Full pairwise similarity matrix ─────────────────────────────────────
    # For speed we sample at most 60 frames from each
    a_sample = a[::max(1, len(a) // 60)]
    b_sample = b[::max(1, len(b) // 60)]

    scores = []
    for ha in a_sample:
        row = [visual_similarity(ha, hb) for hb in b_sample]
        scores.append(row)

    matrix         = np.array(scores)
    overall_score  = float(np.mean(matrix))

    # ── Sliding window: find best matching segment ───────────────────────────
    win        = min(window, len(b_sample), len(a_sample))
    best_score = 0.0
    segments   = []

    if win > 0:
        for i in range(len(a_sample) - win + 1):
            for j in range(len(b_sample) - win + 1):
                window_scores = [matrix[i + k][j + k] for k in range(win)]
                avg            = float(np.mean(window_scores))
                if avg > best_score:
                    best_score = avg
                if avg >= SIMILARITY_THRESHOLDS["possible"]:
                    segments.append({
                        "source_start_frame": i,
                        "suspect_start_frame": j,
                        "window_size": win,
                        "avg_similarity": round(avg, 2),
                    })

    # Deduplicate segments that overlap heavily
    segments.sort(key=lambda x: x["avg_similarity"], reverse=True)
    unique_segments = []
    for seg in segments:
        overlap = any(
            abs(seg["source_start_frame"] - s["source_start_frame"]) < win
            for s in unique_segments
        )
        if not overlap:
            unique_segments.append(seg)
        if len(unique_segments) >= 5:
            break

    final_score = round(max(overall_score, best_score * 0.7), 2)

    return {
        "overall_similarity": round(overall_score, 2),
        "best_window_score":  round(best_score, 2),
        "final_score":        final_score,
        "match_type":         classify_match(final_score),
        "matched_segments":   unique_segments,
    }


# ============ AUDIO FINGERPRINT ============

def generate_audio_fingerprint(audio_path: str) -> str:
    try:
        stats = os.stat(audio_path)
        return hashlib.md5(f"{stats.st_size}_{stats.st_mtime}".encode()).hexdigest()[:16]
    except Exception as e:
        return f"error_{str(e)[:8]}"


def combine_hashes(visual: str, audio: str) -> str:
    return hashlib.md5(f"{visual}_{audio}".encode()).hexdigest()[:16]


# ============ DETECTION SERVICE ============

def _extract_platform(url: str) -> str:
    url_lower = url.lower()
    if "youtube.com" in url_lower or "youtu.be" in url_lower:
        return "youtube"
    elif "twitter.com" in url_lower or "x.com" in url_lower:
        return "twitter"
    elif "instagram.com" in url_lower:
        return "instagram"
    elif "tiktok.com" in url_lower:
        return "tiktok"
    elif "reddit.com" in url_lower:
        return "reddit"
    elif "facebook.com" in url_lower or "fb.com" in url_lower:
        return "facebook"
    else:
        try:
            from urllib.parse import urlparse
            return urlparse(url).netloc.replace("www.", "")
        except Exception:
            return "web"

JURISDICTION_META = {
    "US": {
        "notice_type": "DMCA",
        "full_name":   "Digital Millennium Copyright Act",
        "citation":    "17 U.S.C. § 512",
        "deadline":    "Expeditious removal required",
    },
    "EU": {
        "notice_type": "DSA",
        "full_name":   "Digital Services Act",
        "citation":    "Regulation (EU) 2022/2065",
        "deadline":    "Content must be assessed within 24 hours for illegal content",
    },
    "IN": {
        "notice_type": "IT_ACT",
        "full_name":   "Information Technology Act, 2000",
        "citation":    "Section 79 – Intermediary Guidelines",
        "deadline":    "Removal within 36 hours for unlawful content",
    },
    "GB": {
        "notice_type": "DMCA",
        "full_name":   "Copyright, Designs and Patents Act 1988",
        "citation":    "CDPA 1988 + Online Safety Act 2023",
        "deadline":    "Expeditious removal expected",
    },
    "FR": {
        "notice_type": "DSA",
        "full_name":   "Digital Services Act (EU)",
        "citation":    "Regulation (EU) 2022/2065",
        "deadline":    "Content must be assessed within 24 hours",
    },
    "DE": {
        "notice_type": "DSA",
        "full_name":   "Digital Services Act (EU) + NetzDG",
        "citation":    "Regulation (EU) 2022/2065 & NetzDG § 3",
        "deadline":    "Removal within 24 hours for manifestly unlawful content",
    },
    "BR": {
        "notice_type": "GENERIC",
        "full_name":   "Lei do Marco Civil da Internet",
        "citation":    "Lei nº 12.965/2014, Art. 19",
        "deadline":    "Removal required after judicial order or notice",
    },
}
 
def get_jurisdiction_meta(jurisdiction: str) -> dict:
    return JURISDICTION_META.get(jurisdiction, {
        "notice_type": "GENERIC",
        "full_name":   "International Copyright Convention",
        "citation":    "Berne Convention for the Protection of Literary and Artistic Works",
        "deadline":    "Prompt removal expected",
    })

def search_youtube(query: str, limit: int = 5) -> list:
    try:
        import yt_dlp
        detections = []
        ydl_opts   = {"quiet": True, "no_warnings": True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            results = ydl.extract_info(f"ytsearch{limit}:{query}", download=False)
            if results:
                for video in results.get("entries", []):
                    detections.append({
                        "platform":     "youtube",
                        "url":          video.get("webpage_url", ""),
                        "title":        video.get("title", "Unknown"),
                        "confidence":   85.0,
                        "threat_level": "high",
                        "source":       "youtube_search",
                    })
        return detections
    except Exception as e:
        print(f"YouTube search error: {e}")
        return []


def search_reverse_image(image_path: str, limit: int = 5) -> list:
    try:
        from serpapi import GoogleSearch
        import base64

        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        ext      = image_path.split(".")[-1].lower()
        mime     = "image/jpeg" if ext in ["jpg", "jpeg"] else f"image/{ext}"
        data_url = f"data:{mime};base64,{image_data}"

        params = {
            "engine":  "google_lens",
            "url":     data_url,
            "api_key": os.getenv("SERPAPI_KEY"),
        }

        results    = GoogleSearch(params).get_dict()
        detections = []

        for match in results.get("visual_matches", [])[:limit]:
            source_url = match.get("link", "")
            if not source_url:
                continue
            position   = match.get("position", limit)
            confidence = max(40.0, round(95.0 - (position - 1) * 10, 2))
            detections.append({
                "platform":     _extract_platform(source_url),
                "url":          source_url,
                "title":        match.get("title", "Unknown"),
                "confidence":   confidence,
                "threat_level": "high" if confidence >= 80 else "medium",
                "thumbnail":    match.get("thumbnail", ""),
                "source":       "reverse_image_search",
            })

        return detections
    except Exception as e:
        print(f"Reverse image search error: {e}")
        return []


def detect_content(fingerprint_id: str, file_path: str, filename: str) -> list:
    """
    Full detection pipeline:
    Phase 1 — YouTube text search
    Phase 2 — Reverse image search (SerpAPI)
    Phase 3 — Video fingerprint comparison against KnownContent registry
    """
    all_detections = []
    seen_urls      = set()
    file_ext       = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    # Phase 1: YouTube
    query = filename.rsplit(".", 1)[0] if "." in filename else filename
    for det in search_youtube(query, limit=5):
        if det["url"] not in seen_urls:
            seen_urls.add(det["url"])
            all_detections.append(det)

    # Phase 2: Reverse image search
    if file_ext in ["jpg", "jpeg", "png", "gif", "webp"]:
        for det in search_reverse_image(file_path, limit=5):
            if det["url"] not in seen_urls:
                seen_urls.add(det["url"])
                all_detections.append(det)

    elif file_ext in ["mp4", "mov", "avi"]:
        # Extract one frame for reverse image search
        result     = extract_video_fingerprint(file_path, sample_interval=1.0, max_frames=1)
        frame_path = "/tmp/phantrace_frame_search.jpg"
        try:
            from moviepy.editor import VideoFileClip
            v     = VideoFileClip(file_path)
            frame = v.get_frame(min(1.0, v.duration / 2))
            v.close()
            Image.fromarray(frame.astype("uint8")).save(frame_path)
            for det in search_reverse_image(frame_path, limit=5):
                if det["url"] not in seen_urls:
                    seen_urls.add(det["url"])
                    all_detections.append(det)
        except Exception as e:
            print(f"Frame extraction for reverse search failed: {e}")
        finally:
            if os.path.exists(frame_path):
                os.remove(frame_path)

    all_detections.sort(key=lambda x: x.get("confidence", 0), reverse=True)
    return all_detections


# ============ DATABASE DEPENDENCY ============

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ============ PYDANTIC SCHEMAS ============

class KnownContentCreate(PydanticBase):
    content_source: str
    source_url:     str
    source_title:   str | None = None
    visual_hash:    str | None = None
    audio_hash:     str | None = None
    hash_signature: str | None = None
    content_type:   str = "video"
    threat_level:   str = "high"


class HashCheckRequest(PydanticBase):
    visual_hash: str
    top_k:       int = 5


class VideoCompareRequest(PydanticBase):
    fingerprint_id_a: str   # the original (protected) content
    fingerprint_id_b: str   # the suspect content

class StrikeCreate(PydanticBase):
    detection_id:      str | None = None
    title:             str
    account:           str
    platform:          str
    jurisdiction:      str = "US"
    notice_type:       str = "DMCA"
    violation_count:   int = 1
    is_repeat_offender: bool = False
 
 
class StrikeStatusUpdate(PydanticBase):
    status: str   # draft | sent | acknowledged | removed | appealed | restored

# ============ UPLOAD CONFIG ============

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
ALLOWED_EXTENSIONS = {"mp4", "mov", "avi", "mp3", "jpg", "jpeg", "png"}
MAX_FILE_SIZE      = 500 * 1024 * 1024  # 500 MB


# ============ API ENDPOINTS ============

@app.get("/")
def read_root():
    return {"message": "PhanTrace API is running! 🛡️", "version": "1.0.0"}


@app.get("/health")
def health_check():
    return {"status": "ok", "service": "PhanTrace Backend"}


@app.get("/db-test")
def test_database():
    try:
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        db.close()
        return {"status": "✅ Database connected!", "database": "Supabase PostgreSQL"}
    except Exception as e:
        return {"status": "❌ Database connection failed", "error": str(e)}


# ── Upload ────────────────────────────────────────────────────────────────────

@app.post("/api/upload")
async def upload_content(
    file:  UploadFile = File(...),
    owner: str = Form(...),
    event: str = Form(...),
    team:  str = Form(...),
    sport: str = Form(...),
    db:    Session = Depends(get_db),
):
    try:
        file_ext = file.filename.split(".")[-1].lower()
        if file_ext not in ALLOWED_EXTENSIONS:
            return {"error": f"File type .{file_ext} not allowed.", "status": "error"}

        contents = await file.read()
        if len(contents) > MAX_FILE_SIZE:
            return {"error": "File too large. Max 500 MB.", "status": "error"}

        unique_id = str(uuid4())
        file_path = os.path.join(UPLOAD_DIR, f"{unique_id}.{file_ext}")
        with open(file_path, "wb") as f:
            f.write(contents)

        new_content = Content(
            id           = unique_id,
            filename     = file.filename,
            file_path    = file_path,
            file_size_mb = round(len(contents) / (1024 * 1024), 2),
            owner        = owner,
            event        = event,
            team         = team,
            sport        = sport,
            uploaded_by  = "User",
        )
        db.add(new_content)
        db.commit()
        db.refresh(new_content)

        return {
            "status":       "success",
            "file_id":      unique_id,
            "filename":     file.filename,
            "file_path":    file_path,
            "file_size_mb": round(len(contents) / (1024 * 1024), 2),
            "file_type":    file_ext,
            "owner":        owner,
            "event":        event,
            "team":         team,
            "sport":        sport,
            "message":      "File uploaded successfully. Ready for fingerprinting.",
        }

    except Exception as e:
        return {"status": "error", "error": str(e)}


# ── Fingerprint (Phase 1 + 3) ─────────────────────────────────────────────────

@app.post("/api/fingerprint/{file_id}")
async def generate_fingerprint(file_id: str, db: Session = Depends(get_db)):
    """
    Generate fingerprint for uploaded content.
    - Images:  single pHash
    - Audio:   audio fingerprint
    - Videos:  Phase 3 multi-frame DNA + single-frame pHash for backwards compat
    Also auto-checks against KnownContent registry.
    """
    try:
        content = db.query(Content).filter(Content.id == file_id).first()
        if not content:
            return {"status": "error", "error": "File not found in database"}

        # Find file on disk
        file_path = None
        for fname in os.listdir(UPLOAD_DIR):
            if fname.startswith(file_id):
                file_path = os.path.join(UPLOAD_DIR, fname)
                break

        if not file_path or not os.path.exists(file_path):
            return {"status": "error", "error": "File not found on disk"}

        file_ext     = file_path.split(".")[-1].lower()
        visual_hash  = ""
        audio_hash   = ""
        frame_hashes = []
        frame_count  = 0
        video_dna    = None
        video_meta   = {}

        # ── Image ────────────────────────────────────────────────
        if file_ext in ["jpg", "jpeg", "png"]:
            visual_hash = generate_phash(file_path)
            audio_hash  = "N/A"

        # ── Audio ────────────────────────────────────────────────
        elif file_ext == "mp3":
            visual_hash = "N/A"
            audio_hash  = generate_audio_fingerprint(file_path)

        # ── Video: Phase 3 multi-frame fingerprinting ────────────
        elif file_ext in ["mp4", "mov", "avi"]:
            print(f"[Phase 3] Starting video fingerprinting for {file_path}")
            vfp = extract_video_fingerprint(
                video_path       = file_path,
                sample_interval  = 0.5,   # sample every 0.5 seconds
                max_frames       = 120,   # cap at 120 frames (~60s video at 0.5s interval)
            )

            if vfp["error"]:
                print(f"[Phase 3] Warning: {vfp['error']}")

            frame_hashes = vfp["frame_hashes"]
            frame_count  = vfp["frame_count"]
            video_dna    = vfp["video_dna"]
            video_meta   = {
                "duration_s":   vfp.get("duration_s", 0),
                "error_frames": vfp.get("error_frames", 0),
            }

            # Use first valid frame hash as visual_hash for backwards compat
            valid = [h for h in frame_hashes if not h.startswith("error")]
            visual_hash = valid[0] if valid else "N/A"
            audio_hash  = generate_audio_fingerprint(file_path)

            print(f"[Phase 3] Done: {frame_count} frames, DNA={video_dna}")

        combined_hash = combine_hashes(
            visual_hash if visual_hash != "N/A" else "0" * 16,
            audio_hash  if audio_hash  != "N/A" else "0" * 16,
        )

        # Save fingerprint
        fingerprint = Fingerprint(
            id             = str(uuid4()),
            content_id     = file_id,
            visual_hash    = visual_hash,
            audio_hash     = audio_hash,
            hash_signature = combined_hash,
            frame_hashes   = json.dumps(frame_hashes) if frame_hashes else None,
            frame_count    = frame_count if frame_count > 0 else None,
            video_dna      = video_dna,
        )
        db.add(fingerprint)
        db.commit()
        db.refresh(fingerprint)

        # ── Auto hash-check against KnownContent ────────────────
        hash_matches = []

        # Check video DNA first (fastest, single string comparison)
        if video_dna and video_dna not in ("no_frames", "no_duration", "extraction_error", "moviepy_missing"):
            known_videos = db.query(KnownContent).filter(
                KnownContent.video_dna.isnot(None),
                KnownContent.content_type == "video"
            ).all()
            for item in known_videos:
                if item.video_dna == video_dna:
                    score = 98.0  # exact DNA match
                    auto_det = Detection(
                        fingerprint_id = fingerprint.id,
                        platform       = item.content_source or "known_registry",
                        url            = item.source_url,
                        title          = item.source_title or "Video DNA Match",
                        confidence     = score,
                        threat_level   = item.threat_level,
                        source         = "video_dna_match",
                    )
                    db.add(auto_det)
                    hash_matches.append({
                        "source_url":   item.source_url,
                        "source_title": item.source_title,
                        "similarity":   score,
                        "match_type":   "exact_match",
                        "method":       "video_dna",
                        "threat_level": item.threat_level,
                    })

        # Check frame-level similarity if no DNA match yet
        if not hash_matches and frame_hashes:
            known_with_frames = db.query(KnownContent).filter(
                KnownContent.frame_hashes.isnot(None)
            ).all()
            for item in known_with_frames:
                try:
                    known_frames = json.loads(item.frame_hashes)
                    result       = compare_video_fingerprints(frame_hashes, known_frames)
                    if result["match_type"] != "no_match":
                        auto_det = Detection(
                            fingerprint_id = fingerprint.id,
                            platform       = item.content_source or "known_registry",
                            url            = item.source_url,
                            title          = item.source_title or "Video Frame Match",
                            confidence     = result["final_score"],
                            threat_level   = item.threat_level,
                            source         = "video_frame_comparison",
                        )
                        db.add(auto_det)
                        hash_matches.append({
                            "source_url":       item.source_url,
                            "source_title":     item.source_title,
                            "similarity":       result["final_score"],
                            "match_type":       result["match_type"],
                            "matched_segments": result["matched_segments"],
                            "method":           "frame_comparison",
                            "threat_level":     item.threat_level,
                        })
                except Exception as e:
                    print(f"Frame comparison error for {item.id}: {e}")
                    continue

        # Fall back to single visual_hash check for images
        if not hash_matches and visual_hash and visual_hash not in ("N/A", "") and not visual_hash.startswith("error"):
            known_items = db.query(KnownContent).filter(
                KnownContent.visual_hash.isnot(None)
            ).all()
            for item in known_items:
                score      = visual_similarity(visual_hash, item.visual_hash)
                match_type = classify_match(score)
                if match_type != "no_match":
                    auto_det = Detection(
                        fingerprint_id = fingerprint.id,
                        platform       = item.content_source or "known_registry",
                        url            = item.source_url,
                        title          = item.source_title or "Hash Match",
                        confidence     = score,
                        threat_level   = item.threat_level,
                        source         = "hash_comparison",
                    )
                    db.add(auto_det)
                    hash_matches.append({
                        "source_url":   item.source_url,
                        "source_title": item.source_title,
                        "similarity":   score,
                        "match_type":   match_type,
                        "method":       "visual_hash",
                        "threat_level": item.threat_level,
                    })

        if hash_matches:
            db.commit()

        return {
            "status":         "success",
            "fingerprint_id": fingerprint.id,
            "content_id":     file_id,
            "visual_hash":    visual_hash,
            "audio_hash":     audio_hash,
            "hash_signature": combined_hash,
            # Phase 3
            "frame_count":    frame_count,
            "video_dna":      video_dna,
            "video_meta":     video_meta,
            # Matches
            "hash_matches":   hash_matches,
            "match_count":    len(hash_matches),
            "message":        "Fingerprint generated successfully",
        }

    except Exception as e:
        return {"status": "error", "error": str(e)}


# ── Get ALL detections (for Evidence Locker page) ──────────────────────────

@app.get("/api/detections")
def get_all_detections(db: Session = Depends(get_db)):
    """Return all detections from all fingerprints"""
    try:
        detections = db.query(Detection).order_by(Detection.detected_at.desc()).all()
        
        return {
            "status": "success",
            "detections": [
                {
                    "id": d.id,
                    "title": d.title,
                    "platform": d.platform,
                    "account": d.source if d.source else "Unknown",
                    "confidence": d.confidence,
                    "threat": d.threat_level.upper() if d.threat_level else "MEDIUM",
                    "time": d.detected_at.isoformat() if d.detected_at else "Unknown",
                    "region": "Unknown",  # You'll need to add region to Detection model
                    "status": "active",  # You'll need to add status to Detection model
                    "contentID": d.id,
                    "uploadDate": d.created_at.isoformat() if d.created_at else "Unknown",
                }
                for d in detections
            ]
        }
    except Exception as e:
        return {"status": "error", "error": str(e), "detections": []}
    
# ── Detection (Phase 1 + 2 external search) ───────────────────────────────────

@app.post("/api/detections/{fingerprint_id}")
async def start_detection(fingerprint_id: str, db: Session = Depends(get_db)):
    """Run external detection: YouTube search + reverse image search."""
    try:
        fingerprint = db.query(Fingerprint).filter(Fingerprint.id == fingerprint_id).first()
        if not fingerprint:
            return {"status": "error", "error": "Fingerprint not found"}

        content = db.query(Content).filter(Content.id == fingerprint.content_id).first()
        if not content:
            return {"status": "error", "error": "Content not found"}

        detections = detect_content(fingerprint_id, content.file_path, content.filename)

        for det in detections:
            db.add(Detection(
                fingerprint_id = fingerprint_id,
                platform       = det["platform"],
                url            = det["url"],
                title          = det["title"],
                confidence     = det["confidence"],
                threat_level   = det["threat_level"],
                thumbnail      = det.get("thumbnail", ""),
                source         = det.get("source", ""),
            ))
        db.commit()

        return {
            "status":           "success",
            "fingerprint_id":   fingerprint_id,
            "detections_found": len(detections),
            "detections":       detections,
            "message":          f"Found {len(detections)} potential matches",
        }

    except Exception as e:
        return {"status": "error", "error": str(e)}


# ── Phase 3: Direct video comparison endpoint ─────────────────────────────────

@app.post("/api/compare-videos", tags=["Phase 3"])
def compare_videos(req: VideoCompareRequest, db: Session = Depends(get_db)):
    """
    Compare two already-fingerprinted videos using frame-level sliding window.
    fingerprint_id_a = your original protected content
    fingerprint_id_b = the suspect content
    """
    fp_a = db.query(Fingerprint).filter(Fingerprint.id == req.fingerprint_id_a).first()
    fp_b = db.query(Fingerprint).filter(Fingerprint.id == req.fingerprint_id_b).first()

    if not fp_a:
        raise HTTPException(status_code=404, detail=f"Fingerprint A not found: {req.fingerprint_id_a}")
    if not fp_b:
        raise HTTPException(status_code=404, detail=f"Fingerprint B not found: {req.fingerprint_id_b}")

    hashes_a = json.loads(fp_a.frame_hashes) if fp_a.frame_hashes else []
    hashes_b = json.loads(fp_b.frame_hashes) if fp_b.frame_hashes else []

    if not hashes_a or not hashes_b:
        return {
            "status":  "error",
            "message": "One or both fingerprints have no frame data. Were these videos fingerprinted with Phase 3?",
        }

    result = compare_video_fingerprints(hashes_a, hashes_b)

    # Auto-save as detection if match found
    if result["match_type"] != "no_match":
        content_b = db.query(Content).filter(Content.id == fp_b.content_id).first()
        db.add(Detection(
            fingerprint_id = fp_a.id,
            platform       = "direct_compare",
            url            = content_b.file_path if content_b else "unknown",
            title          = content_b.filename if content_b else "Unknown file",
            confidence     = result["final_score"],
            threat_level   = "high" if result["final_score"] >= 85 else "medium",
            source         = "video_frame_comparison",
        ))
        db.commit()

    return {
        "status":             "success",
        "fingerprint_id_a":   req.fingerprint_id_a,
        "fingerprint_id_b":   req.fingerprint_id_b,
        "video_dna_a":        fp_a.video_dna,
        "video_dna_b":        fp_b.video_dna,
        "dna_match":          fp_a.video_dna == fp_b.video_dna if fp_a.video_dna and fp_b.video_dna else False,
        "frame_count_a":      fp_a.frame_count,
        "frame_count_b":      fp_b.frame_count,
        **result,
    }

# ── Get detections with detection ID ─────────────────────────────────────────────────
@app.get("/api/detections/detail/{detection_id}")
def get_detection_detail(detection_id: str, db: Session = Depends(get_db)):
    """Return single detection with full details"""
    try:
        detection = db.query(Detection).filter(Detection.id == detection_id).first()
        
        if not detection:
            return {"status": "error", "error": "Detection not found"}
        
        return {
            "status": "success",
            "detection": {
                "id": detection.id,
                "title": detection.title or "Unknown",
                "platform": detection.platform or "Unknown",
                "account": detection.source or "Unknown",
                "confidence": detection.confidence or 0,
                "threat": detection.threat_level.upper() if detection.threat_level else "MEDIUM",
                "time": detection.detected_at.isoformat() if detection.detected_at else "Unknown",
                "region": detection.region or "Unknown",
                "status": detection.status or "active",
                "contentID": detection.id,
                "uploadDate": detection.created_at.isoformat() if detection.created_at else "Unknown",
                "url": detection.url or "",
                "detectionSource": detection.source or "Unknown",
            }
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}
    
# ── Get detections for a file ─────────────────────────────────────────────────

@app.get("/api/detections/{file_id}")
def get_detections(file_id: str, db: Session = Depends(get_db)):
    fingerprint = db.query(Fingerprint).filter(Fingerprint.content_id == file_id).first()
    if not fingerprint:
        return {"status": "error", "error": "No fingerprint found for this file"}

    detections = db.query(Detection).filter(
        Detection.fingerprint_id == fingerprint.id
    ).order_by(Detection.detected_at.desc()).all()

    return {
        "status":     "success",
        "file_id":    file_id,
        "detections": [
            {
                "id":           d.id,
                "platform":     d.platform,
                "url":          d.url,
                "title":        d.title,
                "confidence":   d.confidence,
                "threat_level": d.threat_level,
                "thumbnail":    d.thumbnail,
                "source":       d.source,
                "detected_at":  d.detected_at.isoformat() if d.detected_at else None,
            }
            for d in detections
        ],
    }


# ── Dashboard ─────────────────────────────────────────────────────────────────

@app.get("/api/dashboard")
def get_dashboard(db: Session = Depends(get_db)):
    total_content    = db.query(Content).count()
    total_detections = db.query(Detection).count()
    high_threat      = db.query(Detection).filter(Detection.threat_level == "high").count()
    return {
        "status":           "success",
        "total_content":    total_content,
        "total_detections": total_detections,
        "high_threat":      high_threat,
    }


# ── Known Content Registry ────────────────────────────────────────────────────

@app.post("/api/known-content", tags=["Registry"])
def register_known_content(payload: KnownContentCreate, db: Session = Depends(get_db)):
    existing = db.query(KnownContent).filter(KnownContent.source_url == payload.source_url).first()
    if existing:
        raise HTTPException(status_code=409, detail="URL already registered")
    entry = KnownContent(**payload.dict())
    db.add(entry)
    db.commit()
    db.refresh(entry)
    return {"status": "registered", "id": entry.id}


@app.get("/api/known-content", tags=["Registry"])
def list_known_content(db: Session = Depends(get_db)):
    items = db.query(KnownContent).order_by(KnownContent.added_at.desc()).all()
    return {
        "total": len(items),
        "items": [
            {
                "id":              i.id,
                "content_source":  i.content_source,
                "source_url":      i.source_url,
                "source_title":    i.source_title,
                "content_type":    i.content_type,
                "threat_level":    i.threat_level,
                "has_visual_hash": bool(i.visual_hash),
                "has_video_dna":   bool(i.video_dna),
                "frame_count":     len(json.loads(i.frame_hashes)) if i.frame_hashes else 0,
                "added_at":        i.added_at.isoformat() if i.added_at else None,
            }
            for i in items
        ],
    }


@app.post("/api/check-hash", tags=["Detection"])
def check_hash(req: HashCheckRequest, db: Session = Depends(get_db)):
    known   = db.query(KnownContent).filter(KnownContent.visual_hash.isnot(None)).all()
    results = []
    for item in known:
        score      = visual_similarity(req.visual_hash, item.visual_hash)
        match_type = classify_match(score)
        if match_type != "no_match":
            results.append({
                "id":           item.id,
                "source_url":   item.source_url,
                "source_title": item.source_title,
                "threat_level": item.threat_level,
                "similarity":   score,
                "match_type":   match_type,
            })
    results.sort(key=lambda x: x["similarity"], reverse=True)
    return {
        "query_hash":    req.visual_hash,
        "total_matches": len(results),
        "matches":       results[: req.top_k],
    }

@app.get("/api/strikes")
def list_strikes(db: Session = Depends(get_db)):
    """Return all strikes, newest first."""
    try:
        strikes = db.query(Strike).order_by(Strike.created_at.desc()).all()
        return {
            "status": "success",
            "strikes": [
                {
                    "id":               s.id,
                    "detectionId":      s.detection_id or "",
                    "title":            s.title,
                    "account":          s.account,
                    "platform":         s.platform,
                    "status":           s.status,
                    "jurisdiction":     s.jurisdiction,
                    "noticeType":       s.notice_type,
                    "sentDate":         s.sent_date.isoformat() if s.sent_date else None,
                    "removedDate":      s.removed_date.isoformat() if s.removed_date else None,
                    "violationCount":   s.violation_count,
                    "isRepeatOffender": bool(s.is_repeat_offender),
                    "createdAt":        s.created_at.isoformat() if s.created_at else None,
                }
                for s in strikes
            ],
        }
    except Exception as e:
        return {"status": "error", "error": str(e), "strikes": []}
 
 
# ── Create a strike ───────────────────────────────────────────────────────────
 
@app.post("/api/strikes")
def create_strike(payload: StrikeCreate, db: Session = Depends(get_db)):
    """
    Create a new strike record.
    If violation_count >= 3, auto-marks as repeat offender and escalates notice type.
    """
    try:
        # 🔴 Prevent duplicate strike for same detection
        if payload.detection_id:
            existing = db.query(Strike).filter(Strike.detection_id == payload.detection_id).first()
            if existing:
                raise HTTPException(status_code=400, detail="Strike already exists for this detection")
        is_repeat = payload.is_repeat_offender or payload.violation_count >= 3
 
        # Auto-infer notice type from jurisdiction if not explicitly passed
        meta        = get_jurisdiction_meta(payload.jurisdiction)
        notice_type = payload.notice_type if payload.notice_type != "GENERIC" else meta["notice_type"]
 
        strike = Strike(
            detection_id      = payload.detection_id,
            title             = payload.title,
            account           = payload.account,
            platform          = payload.platform,
            jurisdiction      = payload.jurisdiction,
            notice_type       = notice_type,
            violation_count   = payload.violation_count,
            is_repeat_offender = 1 if is_repeat else 0,
            status            = "draft",
        )
        db.add(strike)
        db.commit()
        db.refresh(strike)
 
        return {
            "status": "success",
            "strike": {
                "id":               strike.id,
                "detectionId":      strike.detection_id or "",
                "title":            strike.title,
                "account":          strike.account,
                "platform":         strike.platform,
                "status":           strike.status,
                "jurisdiction":     strike.jurisdiction,
                "noticeType":       strike.notice_type,
                "violationCount":   strike.violation_count,
                "isRepeatOffender": bool(strike.is_repeat_offender),
                "createdAt":        strike.created_at.isoformat(),
            },
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}
 
 
# ── Update strike status ──────────────────────────────────────────────────────
 
@app.patch("/api/strikes/{strike_id}/status")
def update_strike_status(strike_id: str, payload: StrikeStatusUpdate, db: Session = Depends(get_db)):
    """
    Update a strike's status.
    Automatically sets sentDate when status → 'sent'
    Automatically sets removedDate when status → 'removed'
    """
    try:
        strike = db.query(Strike).filter(Strike.id == strike_id).first()
        if not strike:
            raise HTTPException(status_code=404, detail="Strike not found")
 
        valid_statuses = {"draft", "sent", "acknowledged", "removed", "appealed", "restored"}
        if payload.status not in valid_statuses:
            raise HTTPException(status_code=400, detail=f"Invalid status. Must be one of: {valid_statuses}")
 
        strike.status = payload.status
 
        if payload.status == "sent" and not strike.sent_date:
            strike.sent_date = datetime.utcnow()
 
        if payload.status == "removed" and not strike.removed_date:
            strike.removed_date = datetime.utcnow()
 
        # If the same account re-offended after removal, bump violation count
        if payload.status == "appealed":
            strike.violation_count = (strike.violation_count or 1) + 1
            if strike.violation_count >= 3:
                strike.is_repeat_offender = 1
 
        db.commit()
        db.refresh(strike)
 
        return {
            "status": "success",
            "strike": {
                "id":               strike.id,
                "status":           strike.status,
                "sentDate":         strike.sent_date.isoformat() if strike.sent_date else None,
                "removedDate":      strike.removed_date.isoformat() if strike.removed_date else None,
                "violationCount":   strike.violation_count,
                "isRepeatOffender": bool(strike.is_repeat_offender),
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        return {"status": "error", "error": str(e)}
 
 
# ── Generate DMCA / DSA / IT Act notice text ──────────────────────────────────
 
@app.get("/api/strikes/{strike_id}/notice")
def get_strike_notice(strike_id: str, db: Session = Depends(get_db)):
    """
    Return structured notice data for a strike.
    The frontend uses this to populate the PDF (jsPDF) — no PDF generation on the server.
    """
    try:
        strike = db.query(Strike).filter(Strike.id == strike_id).first()
        if not strike:
            raise HTTPException(status_code=404, detail="Strike not found")
 
        meta = get_jurisdiction_meta(strike.jurisdiction)
 
        notice_body = f"""This is a formal {meta['notice_type']} notice issued under {meta['full_name']} ({meta['citation']}).
 
The content identified below infringes the intellectual property rights of the rights holder and must be removed in accordance with applicable law.
 
DEADLINE: {meta['deadline']}.
 
INFRINGING CONTENT DETAILS:
  Strike ID       : {strike.id}
  Content Title   : {strike.title}
  Infringing Account: {strike.account}
  Platform        : {strike.platform}
  Violation Count : {strike.violation_count}x
  Repeat Offender : {"YES — Legal escalation may apply." if strike.is_repeat_offender else "No prior violations on record."}
 
RIGHTS HOLDER STATEMENT:
I, the undersigned, state under penalty of perjury that I am authorized to act on behalf of the rights holder of the work described herein, and that the use of the described material is not authorized by the rights holder, its agent, or applicable law.
 
COUNTER-NOTICE:
The account holder has the right to file a counter-notice if they believe this takedown was issued in error, subject to the penalties for false statements under {meta['citation']}.
 
Generated by PhanTrace Legal Enforcement System
Date: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}
"""
 
        return {
            "status":       "success",
            "notice_type":  meta["notice_type"],
            "full_name":    meta["full_name"],
            "citation":     meta["citation"],
            "deadline":     meta["deadline"],
            "notice_body":  notice_body,
            "strike": {
                "id":               strike.id,
                "detectionId":      strike.detection_id or "",
                "title":            strike.title,
                "account":          strike.account,
                "platform":         strike.platform,
                "jurisdiction":     strike.jurisdiction,
                "violationCount":   strike.violation_count,
                "isRepeatOffender": bool(strike.is_repeat_offender),
                "noticeType":       strike.notice_type,
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        return {"status": "error", "error": str(e)}
 
 
# ── Batch send strikes ────────────────────────────────────────────────────────
 
@app.post("/api/strikes/batch/send")
def batch_send_strikes(payload: dict, db: Session = Depends(get_db)):
    """
    Send multiple strikes at once.
    Body: { "strike_ids": ["STR-001", "STR-002", ...] }
    """
    try:
        strike_ids = payload.get("strike_ids", [])
        if not strike_ids:
            return {"status": "error", "error": "No strike IDs provided"}
 
        updated = []
        for sid in strike_ids:
            strike = db.query(Strike).filter(Strike.id == sid).first()
            if strike and strike.status == "draft":
                strike.status    = "sent"
                strike.sent_date = datetime.utcnow()
                updated.append(sid)
 
        db.commit()
        return {"status": "success", "updated_count": len(updated), "updated_ids": updated}
    except Exception as e:
        return {"status": "error", "error": str(e)}
 
 
# ── Seed demo strikes (for pitch / testing) ────────────────────────────────────
 
@app.post("/api/strikes/seed-demo")
def seed_demo_strikes(db: Session = Depends(get_db)):
    """
    Seed 6 realistic demo strikes so the Strike Command page has real data.
    Safe to call multiple times — skips if strikes already exist.
    """
    try:
        existing = db.query(Strike).count()
        if existing >= 6:
            return {"status": "skipped", "message": f"{existing} strikes already exist"}
 
        demo_data = [
            {
                "title": "IPL 2024 MI vs CSK",
                "account": "@sportclips99",
                "platform": "YouTube",
                "status": "removed",
                "jurisdiction": "IN",
                "notice_type": "IT_ACT",
                "violation_count": 5,
                "is_repeat_offender": 1,
                "sent_date": datetime.utcnow(),
                "removed_date": datetime.utcnow(),
            },
            {
                "title": "FIFA World Cup Goals",
                "account": "@goalhighlights",
                "platform": "Instagram",
                "status": "sent",
                "jurisdiction": "BR",
                "notice_type": "GENERIC",
                "violation_count": 2,
                "is_repeat_offender": 0,
                "sent_date": datetime.utcnow(),
            },
            {
                "title": "NBA Finals Game 7",
                "account": "u/sportsreels",
                "platform": "Reddit",
                "status": "acknowledged",
                "jurisdiction": "US",
                "notice_type": "DMCA",
                "violation_count": 1,
                "is_repeat_offender": 0,
                "sent_date": datetime.utcnow(),
            },
            {
                "title": "Wimbledon 2024 Final",
                "account": "@tennisclips_hd",
                "platform": "Twitter",
                "status": "removed",
                "jurisdiction": "GB",
                "notice_type": "DMCA",
                "violation_count": 8,
                "is_repeat_offender": 1,
                "sent_date": datetime.utcnow(),
                "removed_date": datetime.utcnow(),
            },
            {
                "title": "Formula 1 Monaco GP",
                "account": "@f1fanzone",
                "platform": "YouTube",
                "status": "appealed",
                "jurisdiction": "FR",
                "notice_type": "DSA",
                "violation_count": 3,
                "is_repeat_offender": 0,
                "sent_date": datetime.utcnow(),
            },
            {
                "title": "Premier League Goals",
                "account": "@plgoals2024",
                "platform": "Instagram",
                "status": "restored",
                "jurisdiction": "DE",
                "notice_type": "DSA",
                "violation_count": 2,
                "is_repeat_offender": 0,
                "sent_date": datetime.utcnow(),
                "removed_date": datetime.utcnow(),
            },
        ]
 
        created = []
        for d in demo_data:
            strike = Strike(**d)
            db.add(strike)
            created.append(d["title"])
 
        db.commit()
        return {"status": "success", "created": len(created), "titles": created}
    except Exception as e:
        db.rollback()
        return {"status": "error", "error": str(e)}