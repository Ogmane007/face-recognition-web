import cv2
from app.config import settings
from app.db import FaceDB
from app.face_engine import FaceEngine

# ----------------------------
# Tracking + Stability Helpers
# ----------------------------

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def iou(b1, b2):
    """
    Intersection-over-Union for bounding boxes: (x1,y1,x2,y2)
    """
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter = inter_w * inter_h

    a1 = max(0, b1[2] - b1[0]) * max(0, b1[3] - b1[1])
    a2 = max(0, b2[2] - b2[0]) * max(0, b2[3] - b2[1])
    union = a1 + a2 - inter + 1e-9

    return inter / union

class Track:
    def __init__(self, bbox):
        self.bbox = bbox
        self.miss = 0

        self.locked_name = None
        self.locked_score = 0.0

        self.candidate_name = None
        self.streak = 0
        self.last_score = 0.0

    def update_bbox(self, bbox):
        self.bbox = bbox
        self.miss = 0

    def mark_missed(self):
        self.miss += 1

    def update_identity(self, name, score, lock_frames=5):
        """
        Lock identity only after `lock_frames` consecutive frames.
        UNKNOWN resets candidate streak and does not lock.
        """
        self.last_score = score

        if name == "UNKNOWN":
            # Do not keep streak on unknown; reduces false locks
            self.candidate_name = None
            self.streak = 0
            return

        # If already locked, keep it unless you want re-lock logic
        if self.locked_name is not None:
            return

        # Update streak
        if name == self.candidate_name:
            self.streak += 1
        else:
            self.candidate_name = name
            self.streak = 1

        if self.streak >= lock_frames:
            self.locked_name = self.candidate_name
            self.locked_score = score

def assign_tracks(tracks, detections, iou_thresh=0.30):
    """
    Greedy assignment: match each detection to best IoU track, if above threshold.
    Returns list of (track_id, det_idx) assignments and list of unmatched det indices.
    """
    if not tracks:
        return [], list(range(len(detections)))

    track_ids = list(tracks.keys())
    used_tracks = set()
    used_dets = set()
    assignments = []

    # Build all candidate pairs with IoU
    pairs = []
    for tid in track_ids:
        for di, det in enumerate(detections):
            pairs.append((iou(tracks[tid].bbox, det), tid, di))

    # Sort by IoU desc and greedily match
    pairs.sort(reverse=True, key=lambda x: x[0])
    for score, tid, di in pairs:
        if score < iou_thresh:
            break
        if tid in used_tracks or di in used_dets:
            continue
        used_tracks.add(tid)
        used_dets.add(di)
        assignments.append((tid, di))

    unmatched = [i for i in range(len(detections)) if i not in used_dets]
    return assignments, unmatched

# ----------------------------
# Main
# ----------------------------

def main():
    db = FaceDB(settings.db_path)
    known = db.fetch_all()

    if not known:
        print("No enrolled users found. Run enroll.py first.")
        return

    engine = FaceEngine(settings.model_name)

    cap = cv2.VideoCapture(settings.camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera index {settings.camera_index}. "
            "Try 0/1/2 in app/config.py"
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Tracking state
    tracks = {}          # track_id -> Track
    next_id = 1

    # Tunables
    LOCK_FRAMES = 5      # <- your requirement
    IOU_THRESH = 0.30
    MAX_MISSES = 10      # frames to keep track alive when face disappears

    print("Recognition running (stable lock over 5 frames). Press Q to quit.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Failed to read frame.")
                break

            faces = engine.detect_faces(frame)

            view = frame.copy()
            h, w = view.shape[:2]

            # Build detection bboxes list (clamped)
            det_bboxes = []
            for f in faces:
                x1, y1, x2, y2 = [int(v) for v in f.bbox]
                x1 = clamp(x1, 0, w - 1)
                x2 = clamp(x2, 0, w - 1)
                y1 = clamp(y1, 0, h - 1)
                y2 = clamp(y2, 0, h - 1)
                det_bboxes.append((x1, y1, x2, y2))

            # Assign detections to existing tracks
            assignments, unmatched_det_idxs = assign_tracks(tracks, det_bboxes, IOU_THRESH)

            # Mark all tracks missed by default, then reset for matched ones
            matched_track_ids = set(tid for tid, _ in assignments)
            for tid, tr in list(tracks.items()):
                if tid not in matched_track_ids:
                    tr.mark_missed()
                    if tr.miss > MAX_MISSES:
                        del tracks[tid]

            # Update matched tracks + identity
            for tid, di in assignments:
                tr = tracks[tid]
                tr.update_bbox(det_bboxes[di])

                # Recognize this face this frame
                emb = faces[di].embedding
                name, score = engine.recognize_one(emb, known, settings.match_threshold)
                tr.update_identity(name, score, lock_frames=LOCK_FRAMES)

            # Create new tracks for unmatched detections
            for di in unmatched_det_idxs:
                tr = Track(det_bboxes[di])

                emb = faces[di].embedding
                name, score = engine.recognize_one(emb, known, settings.match_threshold)
                tr.update_identity(name, score, lock_frames=LOCK_FRAMES)

                tracks[next_id] = tr
                next_id += 1

            # Draw UI
            cv2.putText(view, "Q=quit", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(view, f"Faces detected: {len(faces)} | Tracks: {len(tracks)}",
                        (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Draw tracks
            for tid, tr in tracks.items():
                x1, y1, x2, y2 = tr.bbox

                locked = tr.locked_name is not None
                if locked:
                    display_name = tr.locked_name
                    display_score = tr.locked_score
                    status = "LOCKED"
                    color = (0, 255, 0)
                else:
                    # If not locked, show candidate + streak (or UNKNOWN)
                    if tr.candidate_name:
                        display_name = tr.candidate_name
                        display_score = tr.last_score
                        status = f"STABLE {tr.streak}/{LOCK_FRAMES}"
                        color = (0, 255, 255)  # yellow while stabilizing
                    else:
                        display_name = "UNKNOWN"
                        display_score = tr.last_score
                        status = "SEARCHING"
                        color = (0, 0, 255)

                cv2.rectangle(view, (x1, y1), (x2, y2), color, 2)

                label = f"{display_name}  {display_score:.2f}  [{status}]"
                (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                box_y1 = max(0, y1 - th - baseline - 6)
                box_y2 = y1
                box_x2 = min(w - 1, x1 + tw + 10)

                cv2.rectangle(view, (x1, box_y1), (box_x2, box_y2), color, -1)
                cv2.putText(view, label, (x1 + 5, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            cv2.imshow("Face Recognition", view)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q')):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()