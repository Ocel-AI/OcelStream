import uuid

class SpotManager:
    def __init__(self, max_spots):
        self.max_spots = max_spots
        self.used_spots = set()
        self.released_spots = set()
        self.next_spot = 0
        self.uuid_map = {}  # Map from spot to UUID

    def acquire(self):
        if self.released_spots:
            spot = self.released_spots.pop()
            is_fresh = False
        elif self.next_spot < self.max_spots:
            spot = self.next_spot
            self.next_spot += 1
            is_fresh = True
        else:
            return None, None, None  # No available spot

        self.used_spots.add(spot)

        if spot not in self.uuid_map:
            self.uuid_map[spot] = str(uuid.uuid4())  # Assign new UUID if not already present

        return spot, self.uuid_map[spot], is_fresh

    def release(self, spot):
        if spot in self.used_spots:
            self.used_spots.remove(spot)
            self.released_spots.add(spot)

    def get_used(self):
        return sorted(self.used_spots)

    def get_available(self):
        return list(self.released_spots) + list(range(self.next_spot, self.max_spots))

    def release_all(self):
        self.released_spots.update(self.used_spots)
        self.used_spots.clear()
        self.next_spot = 0

    def get_uuid(self, spot):
        return self.uuid_map.get(spot)
    
    def get_spot_by_uuid(self, uid):
        for spot, uuid in self.uuid_map.items():
            if uuid == uid:
                return spot
        return None


