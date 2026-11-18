"""Gap filling: one FrameHands per source frame from the strided, tracked detections."""

from __future__ import annotations

from egovision.schema import FrameHands, Hand


def _lerp(a: list[float], b: list[float], w: float) -> list[float]:
    return [x + (y - x) * w for x, y in zip(a, b, strict=True)]


def _interpolate_hand(a: Hand, b: Hand, w: float) -> Hand:
    return Hand(
        id=a.id, side=a.side, score=min(a.score, b.score), interp=True, source="interpolated",
        bbox=_lerp(a.bbox, b.bbox, w),
        kp2d=[_lerp(p, q, w) for p, q in zip(a.kp2d, b.kp2d, strict=True)],
    )


def bridge_dropouts(frames: list[FrameHands], max_gap: int) -> list[FrameHands]:
    """Fill missing tracks between short, same-ID detections on processed frames."""
    previous: dict[int, tuple[int, Hand]] = {}
    additions: dict[int, list[Hand]] = {}
    for index, frame in enumerate(frames):
        for hand in frame.hands:
            if hand.id is None:
                continue
            prior = previous.get(hand.id)
            if prior is not None:
                left, start_hand = prior
                gap = frame.frame - frames[left].frame
                if gap <= max_gap and index - left > 1:
                    for missing in range(left + 1, index):
                        weight = (frames[missing].t - frames[left].t) / (frame.t - frames[left].t)
                        additions.setdefault(missing, []).append(_interpolate_hand(start_hand, hand, weight))
            previous[hand.id] = (index, hand)
    return [
        frame.model_copy(update={"hands": frame.hands + additions[index]}) if index in additions else frame
        for index, frame in enumerate(frames)
    ]


def fill_gaps(processed: list[FrameHands], n_frames: int, fps: float, max_gap: int) -> list[FrameHands]:
    """Return one FrameHands per source frame in [0, n_frames).

    Frames between two processed frames get a hand (marked interp=True) only when the same id is
    present on both sides and the gap is <= max_gap source frames. Otherwise they stay empty:
    we never invent a hand we did not see on both sides.
    """
    by_frame = {fh.frame: fh for fh in processed}
    keys = sorted(by_frame)
    out: list[FrameHands] = []
    k = 0
    for i in range(n_frames):
        if i in by_frame:
            out.append(by_frame[i])
            continue
        while k + 1 < len(keys) and keys[k + 1] < i:
            k += 1
        prev_i = keys[k] if keys and keys[k] < i else None
        next_i = keys[k + 1] if k + 1 < len(keys) and keys[k + 1] > i else None
        hands: list[Hand] = []
        if prev_i is not None and next_i is not None and next_i - prev_i <= max_gap:
            w = (i - prev_i) / (next_i - prev_i)
            nxt = {h.id: h for h in by_frame[next_i].hands if h.id is not None}
            for a in by_frame[prev_i].hands:
                b = nxt.get(a.id) if a.id is not None else None
                if b is None:
                    continue
                hands.append(_interpolate_hand(a, b, w))
        out.append(FrameHands(frame=i, t=round(i / fps, 4), hands=hands))
    return out
