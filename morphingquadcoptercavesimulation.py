"""


Grid convention:
    0 = free space (drone can be here)
    1 = wall (solid rock)
    2 = obstacle (rubble, debris - inside otherwise-free passages)

Coordinates: grid[row, col] where row = y (down), col = x (right).
When we plot with imshow + origin='lower', row 0 is at the bottom,
matching a normal x/y coordinate system for the drone's (x, y) position.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random

FREE, WALL, OBSTACLE = 0, 1, 2


# ---------------------------------------------------------------------------
# 1. CAVE GENERATION
# ---------------------------------------------------------------------------
def generate_cave(width=130, height=85, seed=None):
    """
    Carve a cave into a grid that starts entirely as WALL.

    Approach: a "tunneling walker" that moves forward, carves a corridor
    of some width around itself, occasionally turns, and occasionally
    stops to carve a large circular chamber. This gives varying passage
    widths, turns, and open rooms -- the things that would actually make
    your 3 drone morphologies (H/O/T) matter -- instead of uniform
    random rectangles.
    """
    rng = random.Random(seed)
    grid = np.ones((height, width), dtype=np.uint8) * WALL

    def carve_disc(cx, cy, radius):
        """Carve a filled circle of FREE space centered at (cx, cy)."""
        y0, y1 = max(0, int(cy - radius)), min(height, int(cy + radius) + 1)
        x0, x1 = max(0, int(cx - radius)), min(width, int(cx + radius) + 1)
        for y in range(y0, y1):
            for x in range(x0, x1):
                if (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2:
                    grid[y, x] = FREE

    # Start roughly in the middle-left, heading right.
    x, y = width * 0.15, height * 0.5
    heading = 0.0  # radians

    passage_min_r, passage_max_r = 1.3, 2.6  # corridor "radius" (half-width)
    steps = 0
    max_steps = 1100
    margin = 8  # start nudging away from the edge within this many cells
    next_chamber_in = rng.randint(60, 110)
    next_turn_in = rng.randint(10, 20)
    radius = rng.uniform(passage_min_r, passage_max_r)

    while steps < max_steps:
        carve_disc(x, y, radius)

        # Occasionally open into a chamber (kept modest in size and rare,
        # otherwise repeated carves overlap into one giant open cavern
        # instead of distinct rooms)
        next_chamber_in -= 1
        if next_chamber_in <= 0:
            carve_disc(x, y, rng.uniform(3.5, 5))
            next_chamber_in = rng.randint(70, 130)

        # Continuous small heading drift (a gentle random walk in
        # direction) plus occasional bigger turns -- gives natural-looking
        # bends instead of sharp corners or a straight line.
        heading += rng.uniform(-0.12, 0.12)
        next_turn_in -= 1
        if next_turn_in <= 0:
            heading += rng.uniform(-0.9, 0.9)
            radius = rng.uniform(passage_min_r, passage_max_r)
            next_turn_in = rng.randint(10, 25)

        # Gently push the heading away from nearby edges (proportional
        # nudge, not a hard snap) so the walker stays in-bounds without
        # looping back over its own path.
        push_x = push_y = 0.0
        if x < margin:
            push_x = (margin - x) / margin
        elif x > width - margin:
            push_x = -(x - (width - margin)) / margin
        if y < margin:
            push_y = (margin - y) / margin
        elif y > height - margin:
            push_y = -(y - (height - margin)) / margin
        if push_x or push_y:
            edge_heading = np.arctan2(push_y, push_x)
            heading = 0.75 * heading + 0.25 * edge_heading

        x += np.cos(heading) * 1.0
        y += np.sin(heading) * 1.0
        steps += 1

    # Scatter obstacle (rubble) cells inside free space, away from walls,
    # so passages remain traversable but not empty.
    free_cells = np.argwhere(grid == FREE)
    n_obstacles = len(free_cells) // 40
    placed = 0
    attempts = 0
    while placed < n_obstacles and attempts < n_obstacles * 20:
        attempts += 1
        ry, rx = free_cells[rng.randrange(len(free_cells))]
        # only place if the immediate 4-neighborhood is free (avoid
        # sealing off narrow single-width corridors)
        y0, y1 = max(0, ry - 1), min(height, ry + 2)
        x0, x1 = max(0, rx - 1), min(width, rx + 2)
        if np.all(grid[y0:y1, x0:x1] != WALL) and grid[ry, rx] == FREE:
            grid[ry, rx] = OBSTACLE
            placed += 1

    start = (width * 0.15, height * 0.5)
    return grid, start


# ---------------------------------------------------------------------------
# 2. DRONE MORPHOLOGIES (footprint) + DISTANCE SENSOR SIMULATION
# ---------------------------------------------------------------------------
from matplotlib.path import Path

# Real sensor characteristics (loosely modeled on a short-range ToF sensor
# like a VL53L1X). These are placeholders -- swap in real datasheet numbers
# once you've picked hardware.
SENSOR_MAX_RANGE = 6.0      # grid units; finite, not "basically unlimited"
SENSOR_MIN_RANGE = 0.15     # blind zone close to the sensor
SENSOR_FOV_DEG = 25.0       # cone half-angle... total cone width, see below
SENSOR_N_SUBRAYS = 5        # sub-rays sampled across the cone
SENSOR_NOISE_BASE = 0.015   # fixed noise floor (grid units)
SENSOR_NOISE_PROP = 0.02    # extra noise proportional to distance


class DroneConfig:
    """
    One drone morphology: a body footprint (for collision checking) plus
    where the distance sensors are mounted on that body.

    `footprint` is a polygon in the BODY FRAME: a list of (x, y) points
    with the drone facing +x (heading = 0), origin at the drone's center.
    `sensor_angles` are directions (radians, body frame) each sensor
    points -- the mount POSITION is computed automatically as where that
    angle exits the footprint polygon, so sensors always sit on the body
    surface, not floating at the centroid.

    NOTE: the footprint numbers below are placeholders sized relative to
    the simulated cave's passage widths (~1.3-2.6 unit radius), not real
    drone dimensions. Swap these for your actual mechanical envelope once
    you have it -- nothing else in the sim needs to change.
    """

    def __init__(self, name, footprint, sensor_angles):
        self.name = name
        self.footprint = footprint
        self.sensor_angles = sensor_angles
        self.sensor_mounts = [
            self._mount_point(angle) for angle in sensor_angles
        ]

    def _mount_point(self, angle, step=0.03):
        """Find where a ray from the body-frame origin along `angle`
        exits the footprint polygon -- that's where the sensor sits."""
        path = Path(self.footprint)
        max_r = max(np.hypot(px, py) for px, py in self.footprint) * 1.5
        dx, dy = np.cos(angle), np.sin(angle)
        r = 0.0
        last_inside = (0.0, 0.0)
        while r < max_r:
            p = (dx * r, dy * r)
            if not path.contains_point(p):
                return last_inside
            last_inside = p
            r += step
        return last_inside


# Placeholder H / O / T footprints. Bounding sizes are deliberately varied
# so the 3 configs will later matter for which passages they fit through.
CONFIGS = {
    # O: compact, roughly square/folded -- smallest footprint, best for
    # tight passages.
    "O": DroneConfig(
        name="O",
        footprint=[(-0.45, -0.45), (0.45, -0.45), (0.45, 0.45), (-0.45, 0.45)],
        sensor_angles=[2 * np.pi * i / 6 for i in range(6)],
    ),
    # H: nominal flat quad shape, arms spread -- wider footprint, more
    # stable, better sensor spread for open chambers.
    "H": DroneConfig(
        name="H",
        footprint=[(-0.7, -0.5), (0.7, -0.5), (0.7, 0.5), (-0.7, 0.5)],
        sensor_angles=[2 * np.pi * i / 6 for i in range(6)],
    ),
    # T: elongated along one axis (arms folded in line) -- long and
    # narrow, for squeezing through long slot-shaped gaps.
    "T": DroneConfig(
        name="T",
        footprint=[(-0.95, -0.25), (0.95, -0.25), (0.95, 0.25), (-0.95, 0.25)],
        sensor_angles=[2 * np.pi * i / 6 for i in range(6)],
    ),
}


def cast_ray(grid, x, y, angle, max_range=SENSOR_MAX_RANGE, step=0.2):
    """
    March a ray forward in small steps from (x, y) at `angle` (radians,
    world frame) until it hits a WALL/OBSTACLE cell or reaches max_range.
    Returns (distance, (hit_x, hit_y), hit) where `hit` is False if the
    ray reached max_range without hitting anything (a real sensor would
    report "no detection", not a confident distance).

    This "small-step marching" approach is not the most efficient way to
    ray cast (a true grid-traversal / DDA algorithm skips cell-to-cell
    instead of sampling), but it's simple to read and debug, and fast
    enough at this grid size. Worth revisiting if you scale up later.
    """
    height, width = grid.shape
    dx, dy = np.cos(angle), np.sin(angle)
    dist = 0.0
    while dist < max_range:
        px, py = x + dx * dist, y + dy * dist
        col, row = int(px), int(py)
        if row < 0 or row >= height or col < 0 or col >= width:
            return dist, (px, py), True  # ran off the map edge -- treat as a hit
        if grid[row, col] != FREE:
            return dist, (px, py), True
        dist += step
    return max_range, (x + dx * max_range, y + dy * max_range), False


def sense_cone(grid, x, y, angle, rng, max_range=SENSOR_MAX_RANGE,
               fov_deg=SENSOR_FOV_DEG, n_subrays=SENSOR_N_SUBRAYS):
    """
    Simulate one real sensor reading: sample several sub-rays across the
    sensor's field of view and return the NEAREST hit (a real ToF sensor
    reports whatever reflects soonest inside its cone, not one exact
    line). Then add distance-dependent noise and enforce the minimum
    range, matching how a real short-range ToF sensor behaves.

    Returns a dict: distance, hit point, and `valid` (False = no
    detection within range, e.g. pointed into a big open chamber).
    """
    half_fov = np.radians(fov_deg) / 2
    sub_angles = np.linspace(angle - half_fov, angle + half_fov, n_subrays)

    best = None
    for a in sub_angles:
        dist, hit_pt, hit = cast_ray(grid, x, y, a, max_range)
        if hit and (best is None or dist < best[0]):
            best = (dist, hit_pt)

    if best is None:
        return {"distance": max_range, "point": None, "valid": False}

    true_dist, hit_pt = best
    noise_std = SENSOR_NOISE_BASE + SENSOR_NOISE_PROP * true_dist
    noisy_dist = true_dist + rng.gauss(0, noise_std)
    noisy_dist = max(SENSOR_MIN_RANGE, min(max_range, noisy_dist))
    return {"distance": noisy_dist, "point": hit_pt, "valid": True}


class Drone:
    """
    Position/heading + a body footprint (from `config`) + distance
    sensors mounted on that footprint.

    Sensor mounts are stored RELATIVE to the drone's own heading (body
    frame), so as the drone rotates, both the mount positions and the
    directions they point rotate with it. This matters later: when you
    fuse in IMU heading, you'll rotate these by the IMU's heading
    estimate to know where each ray actually is in the world frame.
    """

    def __init__(self, x, y, heading=0.0, config=None, seed=None):
        self.x = x
        self.y = y
        self.heading = heading
        self.config = config or CONFIGS["H"]
        self.rng = random.Random(seed)

    def _world_footprint(self):
        """Rotate + translate the body-frame footprint into world coords."""
        c, s = np.cos(self.heading), np.sin(self.heading)
        return [
            (self.x + px * c - py * s, self.y + px * s + py * c)
            for px, py in self.config.footprint
        ]

    def sense(self, grid):
        """Return one reading dict (see sense_cone) per sensor, using the
        sensor's actual mount point and outward-pointing direction on
        the rotated body, not the drone's center."""
        c, s = np.cos(self.heading), np.sin(self.heading)
        readings = []
        for (mx, my), angle in zip(self.config.sensor_mounts, self.config.sensor_angles):
            wx = self.x + mx * c - my * s
            wy = self.y + mx * s + my * c
            world_angle = self.heading + angle
            readings.append(sense_cone(grid, wx, wy, world_angle, self.rng))
        return readings

    def can_move_to(self, grid, x, y):
        """Footprint collision check: rasterize the rotated/translated
        body polygon and confirm every cell it covers is FREE."""
        height, width = grid.shape
        c, s = np.cos(self.heading), np.sin(self.heading)
        world_poly = [
            (x + px * c - py * s, y + px * s + py * c)
            for px, py in self.config.footprint
        ]
        path = Path(world_poly)
        xs = [p[0] for p in world_poly]
        ys = [p[1] for p in world_poly]
        x0, x1 = max(0, int(np.floor(min(xs)))), min(width, int(np.ceil(max(xs))) + 1)
        y0, y1 = max(0, int(np.floor(min(ys)))), min(height, int(np.ceil(max(ys))) + 1)
        if x0 <= 0 or x1 >= width or y0 <= 0 or y1 >= height:
            # touching or past the map edge
            if min(xs) < 0 or max(xs) >= width or min(ys) < 0 or max(ys) >= height:
                return False
        # sample a grid of points across the bounding box, check the ones
        # inside the polygon land on FREE cells
        xx, yy = np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(y0, y1) + 0.5)
        pts = np.column_stack([xx.ravel(), yy.ravel()])
        inside = path.contains_points(pts)
        for (px, py), is_inside in zip(pts, inside):
            if is_inside and grid[int(py), int(px)] != FREE:
                return False
        return True


# ---------------------------------------------------------------------------
# 3. VISUALIZATION
# ---------------------------------------------------------------------------
def draw_scene(ax, grid, drone, readings):
    ax.clear()
    cmap = plt.matplotlib.colors.ListedColormap(["white", "#3b3b3b", "#b06a2c"])
    ax.imshow(grid, cmap=cmap, origin="lower", vmin=0, vmax=2)

    # sensor rays: solid = valid detection, dashed/faint = no detection
    # (sensor pointed into open space beyond max range)
    c, s = np.cos(drone.heading), np.sin(drone.heading)
    for (mx, my), angle_offset, reading in zip(
        drone.config.sensor_mounts, drone.config.sensor_angles, readings
    ):
        wx = drone.x + mx * c - my * s
        wy = drone.y + mx * s + my * c
        if reading["valid"]:
            hx, hy = reading["point"]
            ax.plot([wx, hx], [wy, hy], color="#e63946", linewidth=1.2, alpha=0.85)
            ax.plot(hx, hy, "o", color="#e63946", markersize=3)
        else:
            angle = drone.heading + angle_offset
            ex = wx + np.cos(angle) * reading["distance"]
            ey = wy + np.sin(angle) * reading["distance"]
            ax.plot([wx, ex], [wy, ey], color="#e63946", linewidth=1.0, alpha=0.3, linestyle="--")

    # drone body: actual footprint polygon, not a placeholder shape
    world_poly = drone._world_footprint()
    body = patches.Polygon(world_poly, closed=True, color="#1d3557", alpha=0.9)
    ax.add_patch(body)
    # heading tick so orientation is readable at a glance
    ax.plot([drone.x, drone.x + c * 0.6], [drone.y, drone.y + s * 0.6], color="white", linewidth=1.5)

    ax.set_xlim(0, grid.shape[1])
    ax.set_ylim(0, grid.shape[0])
    ax.set_aspect("equal")
    ax.set_title(f"Cave sim [{drone.config.name}] - arrows to move/turn, q to quit")


def render_static_preview(grid, drone, out_path):
    """Non-interactive: save one frame as PNG (for headless environments)."""
    fig, ax = plt.subplots(figsize=(10, 7))
    readings = drone.sense(grid)
    draw_scene(ax, grid, drone, readings)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. OCCUPANCY GRID MAPPING
#    Built ONLY from the drone's sensor readings + its own pose estimate --
#    this code never reads the ground-truth cave grid.
# ---------------------------------------------------------------------------
LOG_ODDS_OCC = 0.85    # evidence added to a cell when a ray hits it
LOG_ODDS_FREE = -0.35  # evidence added to a cell a ray passes through
LOG_ODDS_MIN, LOG_ODDS_MAX = -4.0, 4.0  # clamp so no cell gets "too certain"


class OccupancyGrid:
    """
    Belief map built ONLY from sensor readings + the drone's own
    (believed) pose. Each cell holds a log-odds value: 0 = unknown
    (p=0.5), positive = more likely occupied, negative = more likely
    free. Log-odds is used instead of raw probability because repeated
    independent observations simply ADD in log-odds space, instead of
    needing Bayes' rule multiplication on every single update.
    """

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.log_odds = np.zeros((height, width), dtype=np.float32)

    def probability_map(self):
        """Convert log-odds back to probability (0..1), for display."""
        return 1.0 / (1.0 + np.exp(-self.log_odds))

    def _update_ray(self, ox, oy, angle, reading, max_range, step=0.3):
        """
        Inverse sensor model for one ray: everything between the sensor
        and the hit point is evidence of FREE space (something passed
        through it without stopping); the cell at the hit point is
        evidence of OCCUPIED. A no-detection reading just marks the
        whole ray up to max_range as free -- we know nothing was there
        within range, but we don't know what's beyond it.
        """
        dx, dy = np.cos(angle), np.sin(angle)
        free_up_to = reading["distance"] - step if reading["valid"] else reading["distance"]
        d = 0.0
        while d < free_up_to:
            px, py = ox + dx * d, oy + dy * d
            col, row = int(px), int(py)
            if 0 <= row < self.height and 0 <= col < self.width:
                self.log_odds[row, col] = np.clip(
                    self.log_odds[row, col] + LOG_ODDS_FREE, LOG_ODDS_MIN, LOG_ODDS_MAX
                )
            d += step

        if reading["valid"]:
            hx, hy = ox + dx * reading["distance"], oy + dy * reading["distance"]
            col, row = int(hx), int(hy)
            if 0 <= row < self.height and 0 <= col < self.width:
                self.log_odds[row, col] = np.clip(
                    self.log_odds[row, col] + LOG_ODDS_OCC, LOG_ODDS_MIN, LOG_ODDS_MAX
                )

    def update_from_pose(self, x, y, heading, config, readings):
        """
        Update the map from one full sensor sweep, using an EXPLICIT pose
        (x, y, heading) rather than reading it off a Drone object. This is
        the hook where localization error enters the picture: feed this
        the true pose and you get an accurate map (as before); feed it a
        drifting pose ESTIMATE instead, and every ray gets planted at the
        wrong place on the map even though the raw sensor readings
        themselves are unaffected -- the sensor doesn't get noisier, your
        belief about where you were standing does.
        """
        c, s = np.cos(heading), np.sin(heading)
        for (mx, my), angle_offset, reading in zip(
            config.sensor_mounts, config.sensor_angles, readings
        ):
            wx = x + mx * c - my * s
            wy = y + mx * s + my * c
            world_angle = heading + angle_offset
            self._update_ray(wx, wy, world_angle, reading, SENSOR_MAX_RANGE)

    def update_from_drone(self, drone, readings):
        """Convenience wrapper: update using the drone's TRUE pose. Only
        appropriate when you deliberately want a ground-truth map (e.g.
        validating the mapper itself, as we did in the previous step) --
        for anything meant to reflect what the drone could actually know,
        use update_from_pose with a pose estimate instead."""
        self.update_from_pose(drone.x, drone.y, drone.heading, drone.config, readings)


# ---------------------------------------------------------------------------
# 5. IMU-BASED POSE ESTIMATION (dead reckoning with noise + drift)
#    The mapper will be fed THIS pose, never the drone's true one.
# ---------------------------------------------------------------------------
# Two distinct error sources, because they behave very differently over time:
#   - White noise: random, independent each step, zero-mean. Roughly
#     cancels out over a long path (grows like sqrt(time)).
#   - Bias drift: a slowly-wandering non-zero offset (temperature drift,
#     imperfect calibration). Does NOT cancel -- it accumulates, and is
#     the dominant error source in real IMU dead-reckoning over time.
HEADING_NOISE_STD = 0.01     # rad, per-step random heading-rate error
HEADING_BIAS_WALK = 0.00005  # rad, std of how much the heading bias wanders per step
SPEED_NOISE_PROP = 0.02      # fraction of movement distance, per-step random error
SPEED_BIAS_WALK = 0.001      # units, std of how much the speed bias wanders per step


class PoseEstimator:
    """
    Dead-reckoned pose ESTIMATE: integrates noisy, drifting measurements
    of "how far did I just move / turn" instead of ever knowing the true
    position. This is what real IMU-based localization looks like with
    no external correction (no GPS, no visual/lidar odometry, no loop
    closure) -- exactly the situation underground.
    """

    def __init__(self, x, y, heading, seed=None):
        self.x, self.y, self.heading = x, y, heading
        self.rng = random.Random(seed)
        self.heading_bias = 0.0
        self.speed_bias = 0.0

    def update(self, true_forward, true_dheading):
        """Feed in the REALIZED true motion for this step (what actually
        happened physically); integrates a noisy, biased measurement of
        it into the pose estimate. Call this every step, including when
        true_forward is 0 (turning in place still drifts the heading)."""
        # biases wander slowly on their own -- this IS what "drift" means
        self.heading_bias += self.rng.gauss(0, HEADING_BIAS_WALK)
        self.speed_bias += self.rng.gauss(0, SPEED_BIAS_WALK)

        measured_dheading = (
            true_dheading + self.heading_bias + self.rng.gauss(0, HEADING_NOISE_STD)
        )
        measured_forward = (
            true_forward * (1 + self.rng.gauss(0, SPEED_NOISE_PROP)) + self.speed_bias
        )

        # dead reckoning: integrate the noisy measurement, not the truth
        self.heading += measured_dheading
        self.x += np.cos(self.heading) * measured_forward
        self.y += np.sin(self.heading) * measured_forward


# ---------------------------------------------------------------------------
# 6. SCAN MATCHING (pose correction from real sensor readings + the map
#    built so far -- NEVER the ground-truth cave grid)
# ---------------------------------------------------------------------------
# A cell counts as "confidently occupied" for prediction purposes once its
# belief probability crosses this threshold. Unknown/unexplored cells
# (probability near 0.5) are treated as passable -- we don't know what's
# there, so a predicted ray shouldn't stop on them.
SCAN_MATCH_OCC_THRESHOLD = 0.65
MIN_COMPARABLE_SENSORS = 2  # below this, there's not enough signal to trust a match


def cast_ray_on_belief(occ_mask, x, y, angle, max_range, step=0.2):
    """Same marching-ray technique as cast_ray, but against the belief
    map (a boolean occupied/not array) instead of the real cave grid --
    this is what the drone PREDICTS it would see at a candidate pose,
    given only what it has already mapped."""
    height, width = occ_mask.shape
    dx, dy = np.cos(angle), np.sin(angle)
    dist = 0.0
    while dist < max_range:
        px, py = x + dx * dist, y + dy * dist
        col, row = int(px), int(py)
        if row < 0 or row >= height or col < 0 or col >= width:
            return {"distance": dist, "valid": True}
        if occ_mask[row, col]:
            return {"distance": dist, "valid": True}
        dist += step
    return {"distance": max_range, "valid": False}


def predict_readings(occ_mask, x, y, heading, config, max_range=SENSOR_MAX_RANGE):
    """What the sensors WOULD read if the drone were at (x, y, heading),
    according to the map built so far. Compared against what the sensors
    ACTUALLY read, this disagreement is the correction signal."""
    c, s = np.cos(heading), np.sin(heading)
    predicted = []
    for (mx, my), angle_offset in zip(config.sensor_mounts, config.sensor_angles):
        wx = x + mx * c - my * s
        wy = y + mx * s + my * c
        world_angle = heading + angle_offset
        predicted.append(cast_ray_on_belief(occ_mask, wx, wy, world_angle, max_range))
    return predicted


def _match_score(predicted, actual):
    """Mean squared distance error over sensors where BOTH the prediction
    and the actual reading are valid hits -- a no-detection on either
    side isn't a usable comparison, so it's skipped rather than
    penalized (we simply don't know enough there)."""
    errors = [
        (p["distance"] - a["distance"]) ** 2
        for p, a in zip(predicted, actual)
        if p["valid"] and a["valid"]
    ]
    if len(errors) < MIN_COMPARABLE_SENSORS:
        return None, len(errors)
    return sum(errors) / len(errors), len(errors)


def scan_match(occ_grid, x_guess, y_guess, heading_guess, config, actual_readings,
               search_xy=1.2, search_theta=0.3, n_steps=5):
    """
    Local brute-force search: try small (dx, dy, dtheta) offsets around
    the current pose guess, predict sensor readings at each candidate
    against the belief map, and keep whichever candidate's predicted
    readings best match what the sensors actually reported.

    This is deliberately simple (a grid search, not a real optimizer) --
    the point right now is to see whether using the real map + real
    sensor data can correct pose at all, not to make it fast or elegant.

    Returns (best_x, best_y, best_heading, n_comparable) if a good-enough
    match was found, or None if there wasn't enough overlap between the
    prediction and the actual readings to trust a correction (e.g. early
    on, when little of the map has been built yet).
    """
    occ_mask = occ_grid.probability_map() > SCAN_MATCH_OCC_THRESHOLD

    offsets_xy = np.linspace(-search_xy, search_xy, n_steps)
    offsets_th = np.linspace(-search_theta, search_theta, n_steps)

    best = None  # (score, x, y, heading, n_comparable)
    for dx in offsets_xy:
        for dy in offsets_xy:
            for dth in offsets_th:
                cx, cy, cth = x_guess + dx, y_guess + dy, heading_guess + dth
                predicted = predict_readings(occ_mask, cx, cy, cth, config)
                score, n_comparable = _match_score(predicted, actual_readings)
                if score is None:
                    continue
                if best is None or score < best[0]:
                    best = (score, cx, cy, cth, n_comparable)

    if best is None:
        return None
    _, bx, by, bth, n_comparable = best
    return bx, by, bth, n_comparable


# ---------------------------------------------------------------------------
# 7. KALMAN FILTER: fuse IMU prediction with scan-match correction
# ---------------------------------------------------------------------------
# We saw directly (previous step) why applying a scan match as a flat
# pose overwrite can HURT: it can't tell "this axis is well-constrained
# by the current scan" from "this axis is ambiguous" (e.g. along a
# featureless corridor). The fix is to track uncertainty explicitly, per
# dimension, and let a low-uncertainty measurement pull hard while a
# high-uncertainty one is nearly ignored -- a scalar Kalman update,
# applied independently to x, y, and heading (a diagonal-covariance
# simplification: no x/y/heading cross-correlation tracked, which loses
# some information a full matrix filter would keep, but keeps every
# number here directly inspectable).
MEAS_UNCERTAINTY_SCALE = 0.05  # converts match-score curvature -> measurement variance
MIN_CURVATURE = 1e-3           # floor so a perfectly flat direction gets large-but-finite variance


class KalmanPoseFilter:
    """
    Wraps a PoseEstimator (the noisy IMU dead-reckoning model) as its
    motion model, and adds explicit per-dimension uncertainty tracking
    (var_x, var_y, var_heading) so scan-match corrections can be
    weighted by how much they're actually worth trusting, instead of
    applied as a flat overwrite.
    """

    def __init__(self, x, y, heading, seed=None, var_x=0.05, var_y=0.05, var_heading=0.01):
        self.pose_estimator = PoseEstimator(x, y, heading, seed=seed)
        self.x, self.y, self.heading = x, y, heading
        self.var_x, self.var_y, self.var_heading = var_x, var_y, var_heading
        # process noise: how much LESS certain each prediction step makes
        # us, before any correction. Approximate and hand-tuned to the
        # drift behavior measured in the previous step -- a real system
        # would derive this from the IMU's datasheet noise figures.
        self.q_xy = 0.01
        self.q_heading = 0.0006

    def predict(self, true_forward, true_dheading):
        """Advance the pose using the IMU motion model, and grow the
        uncertainty -- every prediction makes us less sure, until a
        correction pulls it back down."""
        self.pose_estimator.update(true_forward, true_dheading)
        self.x = self.pose_estimator.x
        self.y = self.pose_estimator.y
        self.heading = self.pose_estimator.heading
        self.var_x += self.q_xy
        self.var_y += self.q_xy
        self.var_heading += self.q_heading

    def correct(self, occ_grid, config, actual_readings, eps_xy=0.3, eps_theta=0.05):
        """
        Scan-match against the map, then estimate per-dimension
        measurement uncertainty from the LOCAL CURVATURE of the match
        score around the best-fit pose: nudge the candidate pose along
        each axis independently and see how much the score worsens. A
        sharp increase = well-constrained = low uncertainty = trust it.
        A flat response = ambiguous = high uncertainty = ignore it.

        Fuses that measurement with the current prediction per-dimension
        via a scalar Kalman update, and re-syncs the internal IMU
        integrator to the corrected pose (so future predictions continue
        from the fused estimate, not the uncorrected drift).

        Returns a dict with the actual Kalman gain used per dimension
        (0 = fully ignored the correction, 1 = fully trusted it), or
        None if no usable scan match was found.
        """
        result = scan_match(occ_grid, self.x, self.y, self.heading, config, actual_readings)
        if result is None:
            return None
        mx, my, mtheta, n_comparable = result

        occ_mask = occ_grid.probability_map() > SCAN_MATCH_OCC_THRESHOLD

        def score_at(x, y, theta):
            predicted = predict_readings(occ_mask, x, y, theta, config)
            score, _ = _match_score(predicted, actual_readings)
            return score if score is not None else 1e6  # heavy penalty: no usable comparison

        s0 = score_at(mx, my, mtheta)
        curv_x = (score_at(mx + eps_xy, my, mtheta) + score_at(mx - eps_xy, my, mtheta) - 2 * s0) / eps_xy ** 2
        curv_y = (score_at(mx, my + eps_xy, mtheta) + score_at(mx, my - eps_xy, mtheta) - 2 * s0) / eps_xy ** 2
        curv_h = (score_at(mx, my, mtheta + eps_theta) + score_at(mx, my, mtheta - eps_theta) - 2 * s0) / eps_theta ** 2

        meas_var_x = MEAS_UNCERTAINTY_SCALE / max(curv_x, MIN_CURVATURE)
        meas_var_y = MEAS_UNCERTAINTY_SCALE / max(curv_y, MIN_CURVATURE)
        meas_var_h = MEAS_UNCERTAINTY_SCALE / max(curv_h, MIN_CURVATURE)

        gains = {}
        for dim, meas_val, meas_var in [("x", mx, meas_var_x), ("y", my, meas_var_y),
                                         ("heading", mtheta, meas_var_h)]:
            pred_val = getattr(self, dim)
            pred_var = getattr(self, "var_" + dim if dim != "heading" else "var_heading")
            k = pred_var / (pred_var + meas_var)  # Kalman gain for this dimension
            fused_val = pred_val + k * (meas_val - pred_val)
            fused_var = (1 - k) * pred_var
            gains[dim] = k
            setattr(self, dim, fused_val)
            setattr(self, "var_" + dim if dim != "heading" else "var_heading", fused_var)

        # re-sync the IMU integrator so future predictions build on the
        # CORRECTED pose, not the uncorrected dead-reckoning trajectory
        self.pose_estimator.x = self.x
        self.pose_estimator.y = self.y
        self.pose_estimator.heading = self.heading

        return {"gains": gains, "n_comparable": n_comparable, "match_pose": (mx, my, mtheta)}


def draw_map_comparison(ax_truth, ax_map, grid, drone, readings, occ_grid, pose_est=None):
    """Side-by-side: ground truth (with drone + rays) vs the estimated
    occupancy map built purely from sensor data. If pose_est is given,
    both panels also mark the drifting pose ESTIMATE (orange) next to
    the drone's true position (blue) -- the gap between them is exactly
    the localization error the map is being built with."""
    draw_scene(ax_truth, grid, drone, readings)
    ax_truth.set_title("Ground truth (for our reference only)")

    ax_map.clear()
    prob = occ_grid.probability_map()
    # white=free, black=occupied, gray=unknown -- standard occupancy grid look
    ax_map.imshow(1 - prob, cmap="gray", origin="lower", vmin=0, vmax=1)

    if pose_est is not None:
        ax_truth.plot(pose_est.x, pose_est.y, "x", color="#f4a300", markersize=9, mew=2,
                       label="pose estimate")
        ax_truth.legend(loc="upper right", fontsize=8)
        ax_map.plot(pose_est.x, pose_est.y, "x", color="#f4a300", markersize=9, mew=2)
    else:
        ax_map.plot(drone.x, drone.y, "o", color="#e63946", markersize=5)

    ax_map.set_xlim(0, grid.shape[1])
    ax_map.set_ylim(0, grid.shape[0])
    ax_map.set_aspect("equal")
    ax_map.set_title("Estimated map (from sensors + pose ESTIMATE only)")


def run_interactive(grid, drone, build_map=True, use_pose_drift=True):
    """Interactive keyboard control. When build_map=True, shows the
    ground-truth cave (left) next to the occupancy map being built
    purely from sensor readings (right). When use_pose_drift=True (the
    realistic case), the map is built from a drifting IMU pose ESTIMATE
    instead of the drone's true position -- the orange x marks where
    the drone THINKS it is."""
    occ_grid = OccupancyGrid(grid.shape[1], grid.shape[0]) if build_map else None
    pose_est = PoseEstimator(drone.x, drone.y, drone.heading) if use_pose_drift else None
    move_step, turn_step = 0.8, np.pi / 12

    if build_map:
        fig, (ax_truth, ax_map) = plt.subplots(1, 2, figsize=(16, 7))
    else:
        fig, ax_truth = plt.subplots(figsize=(10, 7))
        ax_map = None

    def redraw():
        readings = drone.sense(grid)
        if build_map:
            if use_pose_drift:
                occ_grid.update_from_pose(pose_est.x, pose_est.y, pose_est.heading,
                                           drone.config, readings)
            else:
                occ_grid.update_from_drone(drone, readings)
            draw_map_comparison(ax_truth, ax_map, grid, drone, readings, occ_grid, pose_est)
        else:
            draw_scene(ax_truth, grid, drone, readings)
        fig.canvas.draw_idle()

    def on_key(event):
        nx, ny = drone.x, drone.y
        true_forward, true_dheading = 0.0, 0.0
        if event.key == "up":
            nx += np.cos(drone.heading) * move_step
            ny += np.sin(drone.heading) * move_step
        elif event.key == "down":
            nx -= np.cos(drone.heading) * move_step
            ny -= np.sin(drone.heading) * move_step
        elif event.key == "left":
            drone.heading += turn_step
            true_dheading = turn_step
        elif event.key == "right":
            drone.heading -= turn_step
            true_dheading = -turn_step
        elif event.key == "q":
            plt.close(fig)
            return

        if event.key in ("up", "down"):
            if drone.can_move_to(grid, nx, ny):
                realized = move_step if event.key == "up" else -move_step
                drone.x, drone.y = nx, ny
                true_forward = realized
            # if blocked, true_forward stays 0 -- the realized motion,
            # not the commanded one, is what should drive the estimate

        if use_pose_drift:
            pose_est.update(true_forward, true_dheading)

        redraw()

    fig.canvas.mpl_connect("key_press_event", on_key)
    redraw()
    plt.show()


if __name__ == "__main__":
    cave, (start_x, start_y) = generate_cave(seed=7)
    drone = Drone(x=start_x, y=start_y, heading=0.0, config=CONFIGS["H"], seed=1)
    run_interactive(cave, drone, build_map=True, use_pose_drift=True)