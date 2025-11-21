# core/layer2_hrl/ace_playbook.py
"""
ACE (Accumulated, Curated, Evolving) Playbook Implementation
Provides dynamic strategy management with Generator → Reflector → Curator workflow.
"""

import json
import sqlite3
import hashlib
import time
import asyncio
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Callable
from enum import Enum
from pathlib import Path
import threading

class BulletSection(Enum):
    DEFAULTS = "defaults"
    STRATEGIES = "strategies"
    PITFALLS = "pitfalls"
    TOOLS = "tools"
    EXAMPLES = "examples"

@dataclass
class PlaybookBullet:
    """Single playbook entry with scoring metadata."""
    id: str
    content: str
    section: str = "strategies"
    helpful_count: int = 0
    harmful_count: int = 0
    tags: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    source: str = "manual"  # manual, generated, reflected

    @property
    def score(self) -> float:
        total = self.helpful_count + self.harmful_count
        if total == 0:
            return 0.5
        return self.helpful_count / total

    @property
    def confidence(self) -> float:
        """Higher confidence with more feedback."""
        total = self.helpful_count + self.harmful_count
        return min(1.0, total / 10.0)

    def to_dict(self) -> Dict:
        d = asdict(self)
        d['score'] = self.score
        d['confidence'] = self.confidence
        return d

class ACEPlaybook:
    """
    Main playbook class with persistence, retrieval, and evolution.
    Thread-safe for concurrent access.
    """

    def __init__(self, db_path: str = "ace_playbook.db"):
        self.db_path = db_path
        self._lock = threading.RLock()
        self._init_db()
        self._version = self._get_version()

    def _init_db(self):
        """Initialize SQLite database with schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS bullets (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    section TEXT DEFAULT 'strategies',
                    helpful_count INTEGER DEFAULT 0,
                    harmful_count INTEGER DEFAULT 0,
                    tags TEXT DEFAULT '[]',
                    created_at REAL,
                    last_used REAL,
                    source TEXT DEFAULT 'manual'
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_section ON bullets(section)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_score ON bullets(helpful_count, harmful_count)
            """)
            # Initialize version if not exists
            conn.execute(
                "INSERT OR IGNORE INTO metadata (key, value) VALUES ('version', '0')"
            )

    def _get_version(self) -> int:
        with sqlite3.connect(self.db_path) as conn:
            result = conn.execute(
                "SELECT value FROM metadata WHERE key = 'version'"
            ).fetchone()
            return int(result[0]) if result else 0

    def _increment_version(self, conn):
        self._version += 1
        conn.execute(
            "UPDATE metadata SET value = ? WHERE key = 'version'",
            (str(self._version),)
        )

    async def add_bullet(
        self,
        content: str,
        section: str = "strategies",
        tags: List[str] = None,
        source: str = "manual"
    ) -> str:
        """Add a new bullet to the playbook."""
        bullet_id = hashlib.md5(f"{content}{time.time()}".encode()).hexdigest()[:12]
        now = time.time()

        def db_call():
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute(
                        """INSERT INTO bullets
                           (id, content, section, tags, created_at, last_used, source)
                           VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        (bullet_id, content, section, json.dumps(tags or []), now, now, source)
                    )
                    self._increment_version(conn)
            return bullet_id

        return await asyncio.to_thread(db_call)

    async def get_bullet(self, bullet_id: str) -> Optional[PlaybookBullet]:
        """Retrieve a specific bullet by ID."""

        def db_call():
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute(
                    "SELECT * FROM bullets WHERE id = ?", (bullet_id,)
                ).fetchone()

                if row:
                    return PlaybookBullet(
                        id=row['id'],
                        content=row['content'],
                        section=row['section'],
                        helpful_count=row['helpful_count'],
                        harmful_count=row['harmful_count'],
                        tags=json.loads(row['tags']),
                        created_at=row['created_at'],
                        last_used=row['last_used'],
                        source=row['source']
                    )
            return None

        return await asyncio.to_thread(db_call)

    async def retrieve(
        self,
        query: str,
        section: str = None,
        top_k: int = 5,
        min_score: float = 0.0
    ) -> List[PlaybookBullet]:
        """
        Retrieve relevant bullets using keyword matching and scoring.
        In production, replace with embedding-based retrieval.
        """
        query_terms = set(query.lower().split())

        def db_call():
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row

                sql = "SELECT * FROM bullets"
                params = []

                if section:
                    sql += " WHERE section = ?"
                    params.append(section)

                return conn.execute(sql, params).fetchall()

        rows = await asyncio.to_thread(db_call)

        scored_bullets = []
        for row in rows:
            bullet = PlaybookBullet(
                id=row['id'],
                content=row['content'],
                section=row['section'],
                helpful_count=row['helpful_count'],
                harmful_count=row['harmful_count'],
                tags=json.loads(row['tags']),
                created_at=row['created_at'],
                last_used=row['last_used'],
                source=row['source']
            )

            if bullet.score < min_score:
                continue

            # Simple keyword relevance scoring
            content_terms = set(bullet.content.lower().split())
            tag_terms = set(t.lower() for t in bullet.tags)

            keyword_overlap = len(query_terms & (content_terms | tag_terms))
            relevance = keyword_overlap * (0.5 + 0.5 * bullet.score)

            if relevance > 0:
                scored_bullets.append((relevance, bullet))

        # Sort by relevance and return top_k
        scored_bullets.sort(key=lambda x: x[0], reverse=True)
        return [b for _, b in scored_bullets[:top_k]]

    async def mark_helpful(self, bullet_id: str):
        """Mark a bullet as helpful (positive feedback)."""

        def db_call():
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute(
                        "UPDATE bullets SET helpful_count = helpful_count + 1, last_used = ? WHERE id = ?",
                        (time.time(), bullet_id)
                    )
                    self._increment_version(conn)

        await asyncio.to_thread(db_call)

    async def mark_harmful(self, bullet_id: str):
        """Mark a bullet as harmful (negative feedback)."""

        def db_call():
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute(
                        "UPDATE bullets SET harmful_count = harmful_count + 1, last_used = ? WHERE id = ?",
                        (time.time(), bullet_id)
                    )
                    self._increment_version(conn)

        await asyncio.to_thread(db_call)

    async def prune(self, score_threshold: float = 0.3, min_feedback: int = 5) -> int:
        """Remove consistently harmful bullets."""

        def db_call():
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    # Find bullets to prune
                    result = conn.execute("""
                        SELECT id FROM bullets
                        WHERE (helpful_count + harmful_count) >= ?
                        AND CAST(helpful_count AS REAL) / (helpful_count + harmful_count) < ?
                    """, (min_feedback, score_threshold)).fetchall()

                    pruned_ids = [r[0] for r in result]

                    if pruned_ids:
                        conn.execute(
                            f"DELETE FROM bullets WHERE id IN ({','.join('?' * len(pruned_ids))})",
                            pruned_ids
                        )
                        self._increment_version(conn)

                    return len(pruned_ids)

        return await asyncio.to_thread(db_call)

    async def get_section(self, section: str) -> List[PlaybookBullet]:
        """Get all bullets in a section."""

        def db_call():
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    "SELECT * FROM bullets WHERE section = ? ORDER BY helpful_count DESC",
                    (section,)
                ).fetchall()

                return [
                    PlaybookBullet(
                        id=row['id'],
                        content=row['content'],
                        section=row['section'],
                        helpful_count=row['helpful_count'],
                        harmful_count=row['harmful_count'],
                        tags=json.loads(row['tags']),
                        created_at=row['created_at'],
                        last_used=row['last_used'],
                        source=row['source']
                    )
                    for row in rows
                ]

        return await asyncio.to_thread(db_call)

    async def export(self) -> Dict:
        """Export entire playbook as dictionary."""

        def db_call():
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute("SELECT * FROM bullets").fetchall()

                bullets = []
                for row in rows:
                    bullet = PlaybookBullet(
                        id=row['id'],
                        content=row['content'],
                        section=row['section'],
                        helpful_count=row['helpful_count'],
                        harmful_count=row['harmful_count'],
                        tags=json.loads(row['tags']),
                        created_at=row['created_at'],
                        last_used=row['last_used'],
                        source=row['source']
                    )
                    bullets.append(bullet.to_dict())

                return {
                    "version": self._version,
                    "exported_at": time.time(),
                    "bullet_count": len(bullets),
                    "bullets": bullets
                }

        return await asyncio.to_thread(db_call)

    async def import_bullets(self, data: Dict, merge: bool = True):
        """Import bullets from exported data."""
        bullets = data.get("bullets", [])

        def db_call():
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    for b in bullets:
                        if merge:
                            conn.execute("""
                                INSERT OR REPLACE INTO bullets
                                (id, content, section, helpful_count, harmful_count, tags, created_at, last_used, source)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """, (
                                b['id'], b['content'], b['section'],
                                b['helpful_count'], b['harmful_count'],
                                json.dumps(b.get('tags', [])),
                                b.get('created_at', time.time()),
                                b.get('last_used', time.time()),
                                b.get('source', 'imported')
                            ))
                        else:
                            conn.execute("""
                                INSERT OR IGNORE INTO bullets
                                (id, content, section, helpful_count, harmful_count, tags, created_at, last_used, source)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """, (
                                b['id'], b['content'], b['section'],
                                b['helpful_count'], b['harmful_count'],
                                json.dumps(b.get('tags', [])),
                                b.get('created_at', time.time()),
                                b.get('last_used', time.time()),
                                b.get('source', 'imported')
                            ))
                    self._increment_version(conn)

        await asyncio.to_thread(db_call)


class ACECurator:
    """
    Curator component for playbook evolution.
    Handles deduplication, refinement, and growth.
    """

    def __init__(self, playbook: ACEPlaybook, similarity_threshold: float = 0.85):
        self.playbook = playbook
        self.similarity_threshold = similarity_threshold

    def _jaccard_similarity(self, text1: str, text2: str) -> float:
        """Simple Jaccard similarity for deduplication."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        return intersection / union if union > 0 else 0

    def find_duplicates(self) -> List[tuple]:
        """Find potential duplicate bullets."""
        all_bullets = []
        for section in BulletSection:
            all_bullets.extend(self.playbook.get_section(section.value))

        duplicates = []
        for i, b1 in enumerate(all_bullets):
            for b2 in all_bullets[i+1:]:
                sim = self._jaccard_similarity(b1.content, b2.content)
                if sim >= self.similarity_threshold:
                    duplicates.append((b1.id, b2.id, sim))

        return duplicates

    def update_from_execution(self, result: Dict) -> Optional[str]:
        """
        Update playbook based on task execution result.
        Called by HRL Manager after task completion.
        """
        if result.get("success"):
            insight = result.get("insight")
            if insight and len(insight) > 10:
                return self.playbook.add_bullet(
                    content=insight,
                    section="strategies",
                    tags=result.get("tags", []),
                    source="reflected"
                )
        else:
            error = result.get("error")
            if error:
                return self.playbook.add_bullet(
                    content=f"AVOID: {error}",
                    section="pitfalls",
                    tags=["error", "avoid"],
                    source="reflected"
                )
        return None

    def grow_and_refine(self, generator_fn: Callable[[str], str] = None):
        """
        Grow playbook by generating new bullets from existing ones.
        Requires an LLM generator function.
        """
        if not generator_fn:
            return

        # Get high-scoring bullets as seeds
        strategies = self.playbook.get_section("strategies")
        high_scorers = [b for b in strategies if b.score > 0.7 and b.confidence > 0.5]

        for bullet in high_scorers[:5]:  # Limit to top 5
            prompt = f"Based on this strategy: '{bullet.content}', generate a related but distinct strategy."
            new_content = generator_fn(prompt)

            if new_content and not self._is_duplicate(new_content):
                self.playbook.add_bullet(
                    content=new_content,
                    section="strategies",
                    tags=bullet.tags + ["generated"],
                    source="generated"
                )

    def _is_duplicate(self, content: str) -> bool:
        """Check if content is duplicate of existing bullet."""
        for section in BulletSection:
            for bullet in self.playbook.get_section(section.value):
                if self._jaccard_similarity(content, bullet.content) >= self.similarity_threshold:
                    return True
        return False


# --- Seed data for initial playbook ---

SEED_BULLETS = [
    # Defaults
    {"content": "Always validate input before processing", "section": "defaults", "tags": ["validation", "security"]},
    {"content": "Log all critical operations with timestamps", "section": "defaults", "tags": ["logging", "audit"]},
    {"content": "Use circuit breakers for external service calls", "section": "defaults", "tags": ["resilience"]},

    # Strategies
    {"content": "Decompose complex tasks into smaller subtasks for HRL workers", "section": "strategies", "tags": ["hrl", "decomposition"]},
    {"content": "Cache frequently accessed playbook bullets for performance", "section": "strategies", "tags": ["performance", "caching"]},
    {"content": "Use exponential backoff for retry attempts", "section": "strategies", "tags": ["resilience", "retry"]},
    {"content": "Prioritize tasks based on business impact and urgency", "section": "strategies", "tags": ["prioritization"]},

    # Pitfalls
    {"content": "AVOID: Infinite retry loops without circuit breaker", "section": "pitfalls", "tags": ["resilience", "avoid"]},
    {"content": "AVOID: Hardcoding API keys or secrets in code", "section": "pitfalls", "tags": ["security", "avoid"]},
    {"content": "AVOID: Blocking operations in async handlers", "section": "pitfalls", "tags": ["async", "avoid"]},

    # Tools
    {"content": "Use Redis for distributed caching across agents", "section": "tools", "tags": ["redis", "caching"]},
    {"content": "Use Prometheus + Grafana for metrics and monitoring", "section": "tools", "tags": ["monitoring"]},
]


async def seed_playbook(playbook: ACEPlaybook):
    """Seed playbook with initial bullets."""
    for bullet_data in SEED_BULLETS:
        await playbook.add_bullet(
            content=bullet_data["content"],
            section=bullet_data["section"],
            tags=bullet_data.get("tags", []),
            source="seed"
        )

async def main():
    # Demo usage
    playbook = ACEPlaybook("demo_playbook.db")
    await seed_playbook(playbook)

    print("=== ACE Playbook Demo ===\n")

    # Retrieve relevant bullets
    results = await playbook.retrieve("resilience retry external", top_k=3)
    print("Query: 'resilience retry external'")
    for b in results:
        print(f"  [{b.score:.2f}] {b.content}")

    # Add feedback
    if results:
        await playbook.mark_helpful(results[0].id)
        print(f"\nMarked '{results[0].id}' as helpful")

    # Export
    export = await playbook.export()
    print(f"\nPlaybook version: {export['version']}, Bullets: {export['bullet_count']}")

if __name__ == "__main__":
    asyncio.run(main())
