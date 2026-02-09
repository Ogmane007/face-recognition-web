from app.config import settings
from app.db import FaceDB

name = input("Enter name to delete: ").strip()
db = FaceDB(settings.db_path)
n = db.delete_user(name)
print(f"Deleted {n} samples." if n else "User not found.")
