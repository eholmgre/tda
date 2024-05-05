from typing import List

from tda.tracker.track import Track

from .deletor import Deletor

class MissBasedDeletor(Deletor):
    def delete_tracks(self, missed_tracks: List[Track], all_tracks: List[Track], frame_time: float) -> None:

        for t in missed_tracks:
            # if the track only has one hit and we've already missed - it's probably clutter so delete
            if len(t.meas_hist) == 1:
                all_tracks.remove(t)

            elif len(t.meas_hist) == 2:
                # if the track has two hits, let it live for 5 s befor deleting
                if frame_time - t.filter.update_time > 5:
                    all_tracks.remove(t)

            elif len(t.meas_hist) <= 5:
                # if the track has 3-5 hits, let it live for 10 s befor deleting
                if frame_time - t.filter.update_time > 10:
                    all_tracks.remove(t)

            else:
                # let mature tracks live for 20 s before deleting
                if frame_time - t.filter.update_time > 20:
                    all_tracks.remove(t)
