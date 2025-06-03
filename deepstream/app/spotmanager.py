class SpotManager:
    def __init__(self, max_spots):
        self.max_spots = max_spots
        self.used_spots = set()
        self.released_spots = set()
        self.next_spot = 0

    def acquire(self):
        if self.released_spots:
            spot = self.released_spots.pop()
            is_fresh = False
        elif self.next_spot < self.max_spots:
            spot = self.next_spot
            self.next_spot += 1
            is_fresh = True
        else:
            return None, None  # No available spot
        self.used_spots.add(spot)
        return spot, is_fresh

    def release(self, spot):
        if spot in self.used_spots:
            self.used_spots.remove(spot)
            self.released_spots.add(spot)

    def get_used(self):
        return sorted(self.used_spots)

    def get_available(self):
        return list(self.released_spots) + list(range(self.next_spot, self.max_spots))


if __name__ == "__main__":
    manager = SpotManager(5)
    print("Acquiring spots:")
    for _ in range(7):
        spot, is_fresh = manager.acquire()
        print(f"Acquired spot: {spot}, Fresh: {is_fresh}")

    print("\nUsed spots:", manager.get_used())
    print("Available spots:", manager.get_available())

    print("\nReleasing spot 2")
    manager.release(2)
    print("Used spots after release:", manager.get_used())
    print("Available spots after release:", manager.get_available())

    spot, is_fresh = manager.acquire()
    print(f"Acquired spot after release: {spot}, Fresh: {is_fresh}")
