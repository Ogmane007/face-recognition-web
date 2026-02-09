import cv2
from app.config import settings
from app.db import FaceDB
from app.face_engine import FaceEngine
from app.camera import Camera

def main():
    name = input("Enter person name to enroll: ").strip()
    if not name:
        raise SystemExit("Name cannot be empty.")

    db = FaceDB(settings.db_path)
    engine = FaceEngine(settings.model_name)
    cam = Camera(settings.camera_index)

    try:
        cam.open()
    except Exception as e:
        raise SystemExit(f"Could not open camera: {e}")
    
    print("\nEnrollment started.")
    print("Tips: good lighting, face centered, remove cap/mask.")
    print("Press SPACE to capture. Press Q to quit.\n")

    try:
        while True:
            frame = cam.read()
            if frame is None:
                print("Failed to read frame from camera.")
                break
                
            view = frame.copy()
            cv2.putText(view, f"Enroll: {name}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(view, "SPACE=capture  Q=quit", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imshow("Enroll", view)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q')):
                break
            if key == 32:  # space
                emb = engine.get_best_face_embedding(frame)
                if emb is None:
                    print("No face detected. Move closer / improve lighting.")
                    continue
                db.upsert(name, emb)
                print(f"Saved sample for '{name}'. Total samples: {db.count_samples(name)}")
    finally:
        cam.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()