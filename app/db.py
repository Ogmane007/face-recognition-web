import sqlite3
import numpy as np
from typing import List, Tuple, Optional

class FaceDB:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _init(self):
        with self._connect() as con:
            con.execute("""
            CREATE TABLE IF NOT EXISTS faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                embedding BLOB NOT NULL
            );
            """)
            con.execute("CREATE INDEX IF NOT EXISTS idx_faces_name ON faces(name);")
            con.commit()

    @staticmethod
    def _to_blob(vec: np.ndarray) -> bytes:
        vec = np.asarray(vec, dtype=np.float32)
        return vec.tobytes()

    @staticmethod
    def _from_blob(blob: bytes) -> np.ndarray:
        return np.frombuffer(blob, dtype=np.float32)

    def add(self, name: str, embedding: np.ndarray) -> None:
        blob = self._to_blob(embedding)
        with self._connect() as con:
            con.execute("INSERT INTO faces(name, embedding) VALUES(?, ?);", (name, blob))
            con.commit()

    def delete_user(self, name: str) -> int:
        with self._connect() as con:
            cur = con.execute("DELETE FROM faces WHERE name = ?;", (name,))
            con.commit()
            return cur.rowcount

    def list_users(self) -> List[str]:
        with self._connect() as con:
            cur = con.execute("SELECT DISTINCT name FROM faces ORDER BY name;")
            return [r[0] for r in cur.fetchall()]

    def fetch_all(self) -> List[Tuple[str, np.ndarray]]:
        with self._connect() as con:
            cur = con.execute("SELECT name, embedding FROM faces;")
            return [(name, self._from_blob(blob)) for name, blob in cur.fetchall()]

    def count_samples(self, name: str) -> int:
        with self._connect() as con:
            cur = con.execute("SELECT COUNT(*) FROM faces WHERE name=?;", (name,))
            return int(cur.fetchone()[0])
