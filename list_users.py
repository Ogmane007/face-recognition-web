from app.config import settings
from app.db import FaceDB

db = FaceDB(settings.db_path)
users = db.list_users()
if not users:
    print("No users enrolled.")
else:
    for u in users:
        print(f"{u} (samples: {db.count_samples(u)})")
