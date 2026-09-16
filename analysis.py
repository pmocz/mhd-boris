#!/usr/bin/env python3
# Analysis of Boris Simulation results

import os

import matplotlib.pyplot as plt
import numpy as np

REF = "inf"  # c_limit label of the un-corrected (ideal MHD) reference run

# c_limit sweeps for each problem, coarsest correction first
SWEEPS = {
    1: ["2.4", "2.2", "2.0", "1.8", "1.6"],
    2: ["32.0", "16.0", "8.0", "4.0", "2.0", "1.0", "0.5"],
    4: ["16.0", "12.0", "8.0", "6.0"],
}

TITLES = {
    1: "Orszag-Tang",
    2: "Alfven wave",
    4: "Orszag-Tang + guide field",
}


def load(prob_id, field, c_limit):
    return np.load(f"output/p{prob_id}_data_{field}_{c_limit}.npy")


def strip(prob_id, field, c_limits, fname, label):
    """Show the reference run and each Boris run side by side."""
    panels = [load(prob_id, field, c) for c in c_limits]
    plt.figure(figsize=(2.2 * len(panels), 3.0))
    plt.imshow(np.hstack(panels), aspect=1, cmap="jet")
    plt.colorbar()
    plt.title(label)
    plt.xlabel("c_limit = " + ", ".join(c_limits))
    plt.yticks([])
    plt.xticks([])
    plt.tight_layout()
    plt.savefig(fname, dpi=120)
    plt.close()


def main():
    if not os.path.isdir("output"):
        raise SystemExit("no output/ directory -- run the simulations first")

    summary = {}

    for prob_id, c_limits in SWEEPS.items():
        prefix = f"p{prob_id}_"
        all_c = [REF] + c_limits

        try:
            dt = {c: load(prob_id, "dt", c) for c in all_c}
        except FileNotFoundError as e:
            print(f"skipping problem {prob_id}: {e}")
            continue

        # ------------------------------------------------------------------
        # what the Boris correction bought, and what it cost
        # ------------------------------------------------------------------
        speedup = {c: len(dt[REF]) / len(dt[c]) for c in all_c}
        err = {}
        if prob_id == 2:
            # Boris deliberately slows the wave, so a pointwise difference would
            # just measure the intended phase shift.  Compare the amplitude of
            # the k=1 Fourier mode instead: that is what should be unchanged.
            def amplitude(c, prob_id=prob_id):
                bz = load(prob_id, "Bz", c)[0, :]
                return 2.0 * np.abs(np.fft.rfft(bz)[1]) / len(bz)

            amp_ref = amplitude(REF)
            for c in all_c:
                err[c] = abs(amplitude(c) - amp_ref) / amp_ref
        else:
            rho_ref = load(prob_id, "rho", REF)
            for c in all_c:
                rho = load(prob_id, "rho", c)
                err[c] = np.mean(np.abs(rho - rho_ref)) / np.mean(np.abs(rho_ref))
        summary[prob_id] = (speedup, err, all_c)

        # ------------------------------------------------------------------
        # per-problem figures
        # ------------------------------------------------------------------
        if prob_id in (1, 4):
            for field, label in (
                ("rho", "density"),
                ("P_B", "magnetic pressure"),
                ("v", "|v|"),
                ("ca", "Alfven speed"),
                ("cf", "fast speed"),
            ):
                strip(prob_id, field, all_c, prefix + field + ".png", label)

        if prob_id == 2:
            plt.figure()
            N = load(prob_id, "Bz", REF).shape[0]
            dx = 1.0 / N
            xlin = np.linspace(0.5 * dx, 1.0 - 0.5 * dx, N)
            plt.plot(xlin, 0.1 * np.ones(N), "k--", lw=0.8)
            plt.plot(xlin, -0.1 * np.ones(N), "k--", lw=0.8)
            for c in all_c:
                plt.plot(xlin, load(prob_id, "Bz", c)[0, :], label="c_limit = " + c)
            plt.ylim(-0.18, 0.18)
            plt.legend(loc="upper left", fontsize=7, ncol=2)
            plt.xlabel("x")
            plt.ylabel("Bz")
            plt.title("Alfven wave at t = 5 (amplitude should stay 0.1)")
            plt.tight_layout()
            plt.savefig(prefix + "Bz.png", dpi=120)
            plt.close()

        # Timestep history.  The last step of every run is truncated so that it
        # lands exactly on t_end, so it is not a CFL-limited step -- drop it or
        # it shows up as a spurious cliff at the right edge of the plot.
        plt.figure()
        for c in all_c:
            plt.plot(np.cumsum(dt[c])[:-1], dt[c][:-1], label=f"c_limit = {c}")
        plt.legend(loc="lower left", fontsize=8)
        plt.xlabel("t")
        plt.ylabel("dt")
        plt.title(TITLES[prob_id] + ": timestep")
        plt.tight_layout()
        plt.savefig(prefix + "dt.png", dpi=120)
        plt.close()

    # ----------------------------------------------------------------------
    # the summary plot: how much solution change did each speedup cost?
    # ----------------------------------------------------------------------
    plt.figure(figsize=(6, 4.5))
    for prob_id, (speedup, err, all_c) in summary.items():
        s = [speedup[c] for c in all_c]
        e = [err[c] for c in all_c]
        plt.plot(s, e, "o-", label=f"{prob_id}: {TITLES[prob_id]}")
        for c in all_c:
            if c != REF:
                plt.annotate(
                    c,
                    (speedup[c], err[c]),
                    fontsize=6,
                    xytext=(3, 3),
                    textcoords="offset points",
                )
    plt.yscale("log")
    plt.xlabel("speedup  (timesteps taken, relative to ideal MHD)")
    plt.ylabel("relative change in the solution")
    plt.title("What the Boris correction costs for what it buys")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("speedup_vs_error.png", dpi=120)
    plt.close()

    # ----------------------------------------------------------------------
    print(f"{'problem':<28} {'c_limit':>8} {'speedup':>9} {'change':>10}")
    for prob_id, (speedup, err, all_c) in summary.items():
        for c in all_c:
            print(f"{TITLES[prob_id]:<28} {c:>8} {speedup[c]:>8.2f}x {err[c]:>9.1%}")


if __name__ == "__main__":
    main()
