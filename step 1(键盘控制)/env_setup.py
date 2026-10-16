import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle, Polygon, Ellipse
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


A = np.array([0, -0.425, -0.3922, 0, 0, 0], dtype=float)
D = np.array([0.1625, 0, 0, 0.1333, 0.0997, 0.0996], dtype=float)
ALPHA = np.array([np.pi / 2, 0, 0, np.pi / 2, -np.pi / 2, 0])
GRIP_LENGTH = 0.15  # illustrative wrist flange to fingertip distance, m
CYL_HEIGHT = 0.100  # illustrative sample size; replace with measurements if known
CYL_DIAMETER = 0.052
PICK = np.array([-0.59, -0.237, CYL_HEIGHT / 2])
STAND_XY = np.array([-0.47, 0.43])
STAND_TOP_Z = 0.32
RECOGNITION = np.array([*STAND_XY, STAND_TOP_Z + CYL_HEIGHT / 2])
STATIC_SAMPLE = np.array([-0.73, -0.38, CYL_HEIGHT / 2])
DOWN = np.diag([1.0, -1.0, -1.0])  # tool local +Z points down


def dh(theta, a, d, alpha):
    c, s, ca, sa = np.cos(theta), np.sin(theta), np.cos(alpha), np.sin(alpha)
    return np.array([[c, -s * ca, s * sa, a * c],
                     [s, c * ca, -c * sa, a * s],
                     [0, sa, ca, d], [0, 0, 0, 1]], dtype=float)


def fk(q):
    transforms = [np.eye(4)]
    for i in range(6):
        transforms.append(transforms[-1] @ dh(q[i], A[i], D[i], ALPHA[i]))
    return transforms


def point(t, xyz):
    return t[:3, :3] @ np.asarray(xyz) + t[:3, 3]


def tip_pose(q):
    t = fk(q)[-1]
    return point(t, [0, 0, GRIP_LENGTH]), t[:3, :3]


def solve_pose(target_tip, seed):
    def residual(q):
        p, r = tip_pose(q)
        dr = Rotation.from_matrix(DOWN @ r.T).as_rotvec()
        return np.r_[9 * (p - target_tip), 1.3 * dr, 0.012 * (q - seed)]

    seeds = [seed, seed + np.array([0, .25, -.4, .2, 0, 0]),
             np.array([.5, -1.65, 1.75, -1.7, -1.5, 0])]
    candidates = [least_squares(residual, s, max_nfev=110, ftol=1e-7,
                                xtol=1e-7, gtol=1e-7) for s in seeds]
    best = min(candidates, key=lambda x: np.linalg.norm(residual(x.x)))
    p, r = tip_pose(best.x)
    pos_err = np.linalg.norm(p - target_tip)
    angle_err = Rotation.from_matrix(DOWN @ r.T).magnitude()
    if pos_err > .025 or angle_err > .24:
        raise RuntimeError(f"IK did not converge: position={pos_err:.3f} m, "
                           f"rotation={angle_err:.3f} rad")
    return best.x


def smooth(t):
    return t * t * (3 - 2 * t)


def build_motion(fps):
    # Rack -> raised oval recognition surface -> same rack hole.
    targets = [PICK + [0, 0, .29], PICK + [0, 0, .035],
               PICK + [0, 0, .35], RECOGNITION + [0, 0, .25],
               RECOGNITION + [0, 0, .035], RECOGNITION + [0, 0, .20],
               RECOGNITION + [0, 0, .035], RECOGNITION + [0, 0, .25],
               PICK + [0, 0, .29], PICK + [0, 0, .035],
               PICK + [0, 0, .29]]
    seed = np.array([.5, -1.65, 1.75, -1.7, -1.5, 0.])
    q_waypoints = []
    for xyz in targets:
        seed = solve_pose(xyz, seed)
        q_waypoints.append(seed.copy())

    segments = [
        (0, 0, .4, "Ready", "rack", "open"),
        (0, 1, .7, "Approach sample", "rack", "open"),
        (1, 1, .35, "Grip sample", "rack", "close"),
        (1, 2, .65, "Lift sample", "attached", "closed"),
        (2, 3, .8, "Move to inspection", "attached", "closed"),
        (3, 4, .65, "Place on recognition point", "attached", "closed"),
        (4, 4, .35, "Release for inspection", "stand", "open"),
        (4, 5, .55, "Camera inspection", "stand", "open"),
        (5, 6, .55, "Approach inspected sample", "stand", "open"),
        (6, 6, .35, "Grip inspected sample", "stand", "close"),
        (6, 7, .55, "Lift from platform", "attached", "closed"),
        (7, 8, .8, "Return to rack", "attached", "closed"),
        (8, 9, .65, "Lower into original hole", "attached", "closed"),
        (9, 9, .35, "Release in rack", "rack", "open"),
        (9, 10, .65, "Retract", "rack", "open"),
        (10, 10, .45, "Return complete", "rack", "open"),
    ]
    frames = []
    for start, end, seconds, label, location, grip_state in segments:
        n = max(2, round(seconds * fps))
        for j in range(n):
            u = smooth(j / (n - 1))
            q = q_waypoints[start] * (1 - u) + q_waypoints[end] * u
            if grip_state == "close":
                gap = .095 * (1 - u) + .052 * u
            elif grip_state == "open" and label.startswith("Release"):
                gap = .052 * (1 - u) + .095 * u
            else:
                gap = .052 if grip_state == "closed" else .095
            if location == "attached":
                # The sample is represented by its center, just below finger ends.
                box = tip_pose(q)[0] + [0, 0, -.035]
            elif location == "stand":
                box = RECOGNITION
            else:
                box = PICK
            frames.append((q, box, gap, label))
    return frames


def project(xyz):
    x, y, z = np.asarray(xyz)
    return np.array([-x - .58 * y, -.31 * x + .37 * y + 1.43 * z])


def draw(frame, ax):
    q, box, gap, label = frame
    ax.cla()
    ax.set_facecolor("#f6f9fc")
    ax.figure.patch.set_facecolor("#ffffff")
    transforms = fk(q)
    joints = np.array([t[:3, 3] for t in transforms])
    screen = np.array([project(j) for j in joints])

    # Table and return tray are drawn before the robot, so the tool stays clear.
    corners = np.array([[-.91,-.68,0],[.34,-.68,0],[.34,.85,0],[-.91,.85,0]])
    ax.add_patch(Polygon([project(p) for p in corners], closed=True,
                         facecolor="#e9eff4", edgecolor="#bacbd9", lw=1.8))
    tray = np.array([[-.82,-.47,.008],[-.10,-.47,.008],[-.10,.22,.008],[-.82,.22,.008]])
    ax.add_patch(Polygon([project(p) for p in tray], closed=True,
                         facecolor="#f8fbfd", edgecolor="#aabecb", lw=3))
    for x in np.linspace(-.73, -.17, 5):
        for y in np.linspace(-.38, .18, 5):
            hole = project([x, y, .011])
            ax.add_patch(Ellipse(hole, .070, .028, angle=14,
                                 facecolor="#a7b9c3", edgecolor="#dae5ea", lw=1))
    # The two printed stand components are represented as an assembled
    # pedestal; the supplied gcode.3mf files contain previews, not a mesh.
    sx, sy = STAND_XY
    stand_base = [project([sx+dx,sy+dy,.006]) for dx,dy in
                  [(-.11,-.11),(.11,-.11),(.11,.11),(-.11,.11)]]
    ax.add_patch(Polygon(stand_base, closed=True, facecolor="#d5dfe5",
                         edgecolor="#93a9b7", lw=2))
    post_bottom = project([sx,sy,.01])
    post_top = project([sx,sy,STAND_TOP_Z])
    ax.plot([post_bottom[0],post_top[0]],[post_bottom[1],post_top[1]],
            color="#9aafb9", lw=28, solid_capstyle="butt", zorder=2)
    ax.plot([post_bottom[0],post_top[0]],[post_bottom[1],post_top[1]],
            color="#f6f8fa", lw=23, solid_capstyle="butt", zorder=3)
    oval = [project([sx+.24*np.cos(a),sy+.15*np.sin(a),STAND_TOP_Z])
            for a in np.linspace(0,2*np.pi,64)]
    ax.add_patch(Polygon([p + [0,-.018] for p in oval], closed=True,
                         facecolor="#bccbd3", edgecolor="#8fa8b8", lw=1.6, zorder=3))
    ax.add_patch(Polygon(oval, closed=True, facecolor="#e0eaf0",
                         edgecolor="#708fa2", lw=2.6, zorder=4))
    marker = project([sx,sy,STAND_TOP_Z+.004])
    ax.add_patch(Ellipse(marker,.065,.027,angle=-12,fill=False,
                         edgecolor="#19a5bb",lw=2.3,zorder=5))

    def draw_cylinder(center):
        lower = project([center[0],center[1],center[2]-CYL_HEIGHT/2])
        upper = project([center[0],center[1],center[2]+CYL_HEIGHT/2])
        rad = CYL_DIAMETER * .60
        if np.allclose(center, RECOGNITION, atol=.001):
            ax.add_patch(Ellipse(lower, 2.25*rad, .024,
                                 facecolor="#768e9c", alpha=.42, zorder=5))
        ax.add_patch(Polygon([lower+[-rad,0],lower+[rad,0],
                              upper+[rad,0],upper+[-rad,0]], closed=True,
                             facecolor="#e4edf2", edgecolor="#657f8e", lw=1.8,
                             zorder=7))
        ax.add_patch(Ellipse(lower, 2*rad, .017, facecolor="#d5e1e8",
                             edgecolor="#657f8e", lw=1.2, zorder=7))
        ax.add_patch(Ellipse(upper, 2*rad, .020, facecolor="#ffffff",
                             edgecolor="#829ba9", lw=1.5, zorder=8))

    draw_cylinder(STATIC_SAMPLE)
    draw_cylinder(box)
    base_xy = screen[0]
    ax.add_patch(Circle(base_xy, .073, facecolor="#225e80",
                        edgecolor="#183f59", lw=2, zorder=8))
    ax.add_patch(Circle(base_xy, .052, facecolor="#81c8e8",
                        edgecolor="#ffffff", lw=1.4, zorder=9))
    for i in range(6):
        s, e = screen[i], screen[i+1]
        ax.plot([s[0],e[0]], [s[1],e[1]], color="#245571",
                lw=25 if i in (1,2) else 19, solid_capstyle="round", zorder=10+i)
        ax.plot([s[0],e[0]], [s[1],e[1]], color="#edf5f8",
                lw=20 if i in (1,2) else 14, solid_capstyle="round", zorder=11+i)
    for i in range(1,7):
        ax.add_patch(Circle(screen[i], .040 if i < 4 else .029,
                            facecolor="#73bada", edgecolor="#236282",
                            lw=1.7, zorder=23+i))

    wrist = transforms[-1]
    base = point(wrist, [0, 0, .018])
    tip = point(wrist, [0, 0, GRIP_LENGTH])
    b2, tip2 = project(base), project(tip)
    axis = tip2 - b2
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    lateral = np.array([-axis[1], axis[0]])
    palm = tip2 - axis * .125
    shoulder = palm - axis * .035
    ax.plot([b2[0],shoulder[0]], [b2[1],shoulder[1]], color="#264e68",
            lw=14, solid_capstyle="round", zorder=32)
    # Schematic OnRobot RG2: a housing with two opposing, moving fingers.
    housing = [shoulder + lateral*.071,
               shoulder - lateral*.071,
               palm - lateral*.073,
               palm + lateral*.073]
    ax.add_patch(Polygon(housing, closed=True, facecolor="#e6eef2",
                         edgecolor="#2a485d", lw=2.5, zorder=34))
    ax.plot(*np.array([palm-lateral*.073,palm+lateral*.073]).T,
            color="#4e7185", lw=4, solid_capstyle="round", zorder=35)
    jaw_gap = gap * .92
    for side in [-1, 1]:
        pivot = palm + lateral * side * jaw_gap/2
        jaw_start = tip2 - axis * .083 + lateral * side * jaw_gap/2
        jaw_end = tip2 + lateral * side * jaw_gap/2
        ax.add_patch(Circle(pivot,.010,facecolor="#537f98",
                            edgecolor="#243d4e",lw=1,zorder=36))
        ax.plot([jaw_start[0],jaw_end[0]],[jaw_start[1],jaw_end[1]],
                color="#222e36", lw=8, solid_capstyle="round", zorder=37)
        pad_end = jaw_end - lateral*side*.010
        ax.plot([jaw_end[0],pad_end[0]],[jaw_end[1],pad_end[1]],
                color="#161d22",lw=5,solid_capstyle="round",zorder=38)
    # Gemini 305-inspired 42 x 42 x 23 mm dark housing, dual front lenses.
    cam_center = [.082, 0, .040]
    body = [point(wrist, [cam_center[0]+dx, cam_center[1]+dy, cam_center[2]])
            for dx,dy in [(-.021,-.021),(.021,-.021),(.021,.021),(-.021,.021)]]
    body_2d = [project(p) for p in body]
    ax.add_patch(Polygon(body_2d, closed=True, facecolor="#26323e",
                         edgecolor="#101d2a", lw=2, zorder=40))
    for dx in [-.009,.009]:
        lens = project(point(wrist, [cam_center[0]+dx,0,cam_center[2]+.013]))
        ax.add_patch(Circle(lens, .009, facecolor="#111827",
                            edgecolor="#8ba6b9", lw=1, zorder=41))
    cam_ray = project(point(wrist, [.082, 0, .17]))
    ax.plot([np.mean(np.array(body_2d)[:,0]),cam_ray[0]],
            [np.mean(np.array(body_2d)[:,1]),cam_ray[1]],
            color="#24a7b2", lw=1.7, ls="--", zorder=31)

    ax.set_xlim(-.55, 1.34)
    ax.set_ylim(-.36, 1.62)
    ax.set_aspect("equal", adjustable="box")
    ax.set_position([.08,.09,.84,.70])
    ax.set_axis_off()
    ax.text(.02, 1.16, "UR5e  |  RG2  |  Gemini 305", transform=ax.transAxes,
            fontsize=17, weight="bold", color="#253f54", va="top", clip_on=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="ur5e_pick_place_v3.gif",
                        help="Output GIF path")
    parser.add_argument("--fps", type=int, default=9)
    args = parser.parse_args()
    if args.fps < 2 or args.fps > 30:
        parser.error("--fps must be between 2 and 30")
    frames = build_motion(args.fps)
    fig = plt.figure(figsize=(10.2, 7.5), dpi=100)
    ax = fig.add_subplot(111)
    anim = FuncAnimation(fig, lambda i: draw(frames[i], ax),
                         frames=len(frames), interval=1000 / args.fps,
                         repeat=True)
    output = Path(args.out).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(output), writer=PillowWriter(fps=args.fps))
    plt.close(fig)
    print(f"Saved {output} ({len(frames)} frames, {len(frames)/args.fps:.1f} s)")

if __name__ == "__main__":
    main()
