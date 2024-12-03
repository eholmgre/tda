from typing import List

from .deletor import Deletor
from ..track import Track
from ..util.track_writer import TrackWriter


class MissBasedDeletor(Deletor):
    def delete_tracks(self, missed_tracks: List[Track], all_tracks: List[Track], frame_time: float) -> None:

        for t in missed_tracks:
            revisit = t.meas_hist[-1].sensor_revisit
            # if the track only has one hit and we've already missed - it's probably clutter so delete
            if len(t.meas_hist) == 1:
                all_tracks.remove(t)

            elif len(t.meas_hist) == 2:
                # if the track has two hits, let it live for one revisit before deleting
                if frame_time - t.filter.update_time > revisit:
                    all_tracks.remove(t)

            elif len(t.meas_hist) <= 5:
                # if the track has 3-5 hits, let it live for two revisits before deleting
                if frame_time - t.filter.update_time > 2 * revisit:
                    all_tracks.remove(t)

            else:
                # let mature tracks live for 4 revisits before deleting
                if frame_time - t.filter.update_time > 4 * revisit:
                    all_tracks.remove(t)
