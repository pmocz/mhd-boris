# Analysis of Boris Simulation results

import numpy as np
import matplotlib.pyplot as plt


def main():
    for prob_id in [1, 2]:
        if prob_id == 1:
            c_limits = [1.6, 1.8, 2.0, 2.2, 2.4]
        elif prob_id == 2:
            c_limits = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]

        rho = {}
        P_B = {}
        Bz = {}
        v = {}
        ca = {}
        cf = {}
        dt = {}

        # Load the data (rho, P_B, dt)
        prefix = "p" + str(prob_id) + "_"
        for c_limit in c_limits:
            rho[c_limit] = np.load(
                "output/" + prefix + "data_rho_" + str(c_limit) + ".npy"
            )
            P_B[c_limit] = np.load(
                "output/" + prefix + "data_P_B_" + str(c_limit) + ".npy"
            )
            Bz[c_limit] = np.load(
                "output/" + prefix + "data_Bz_" + str(c_limit) + ".npy"
            )
            v[c_limit] = np.load("output/" + prefix + "data_v_" + str(c_limit) + ".npy")
            ca[c_limit] = np.load(
                "output/" + prefix + "data_ca_" + str(c_limit) + ".npy"
            )
            cf[c_limit] = np.load(
                "output/" + prefix + "data_cf_" + str(c_limit) + ".npy"
            )
            dt[c_limit] = np.load(
                "output/" + prefix + "data_dt_" + str(c_limit) + ".npy"
            )

        # stack the rho next to each other and plot it as an image
        rho_all = np.hstack([rho[c_limit] for c_limit in c_limits])
        P_B_all = np.hstack([P_B[c_limit] for c_limit in c_limits])
        v_all = np.hstack([v[c_limit] for c_limit in c_limits])
        ca_all = np.hstack([ca[c_limit] for c_limit in c_limits])
        cf_all = np.hstack([cf[c_limit] for c_limit in c_limits])

        if prob_id == 1:
            plt.figure(figsize=(12, 4))
            plt.imshow(rho_all, aspect=1, cmap="jet")
            plt.colorbar()
            plt.xlabel("c_limit = " + str(c_limits))
            plt.savefig(prefix + "rho.png")
            plt.show()

            plt.figure(figsize=(12, 4))
            plt.imshow(P_B_all, aspect=1, cmap="jet")
            plt.colorbar()
            plt.xlabel("c_limit = " + str(c_limits))
            plt.savefig(prefix + "P_B.png")
            plt.show()

            plt.figure(figsize=(12, 4))
            plt.imshow(v_all, aspect=1, cmap="jet")
            plt.colorbar()
            plt.xlabel("c_limit = " + str(c_limits))
            plt.savefig(prefix + "v.png")
            plt.show()

            plt.figure(figsize=(12, 4))
            plt.imshow(ca_all, aspect=1, cmap="jet")
            plt.colorbar()
            plt.xlabel("c_limit = " + str(c_limits))
            plt.savefig(prefix + "ca.png")
            plt.show()

            plt.figure(figsize=(12, 4))
            plt.imshow(cf_all, aspect=1, cmap="jet")
            plt.colorbar()
            plt.xlabel("c_limit = " + str(c_limits))
            plt.savefig(prefix + "cf.png")
            plt.show()

        if prob_id == 2:
            # plot Bz lineout for each c_limit
            plt.figure()
            N = Bz[c_limit].shape[0]
            dx = 1.0 / N
            xlin = np.linspace(0.5 * dx, 1.0 - 0.5 * dx, N)
            lim = 0.1 * np.ones(N)
            for c_limit in c_limits:
                plt.plot(xlin, lim, "k--")
                plt.plot(xlin, -lim, "k--")
                plt.plot(xlin, Bz[c_limit][0, :], label="c_limit = " + str(c_limit))
                plt.ylim(-0.18, 0.18)
            plt.legend(loc="upper left")
            plt.xlabel("x")
            plt.ylabel("Bz")
            plt.savefig(prefix + "Bz.png")
            plt.show()

        # plot the dt for each c_limit
        plt.figure()
        for c_limit in c_limits:
            plt.plot(
                np.cumsum(dt[c_limit]), dt[c_limit], label="c_limit = " + str(c_limit)
            )
        plt.legend(loc="lower left")
        plt.xlabel("t")
        plt.savefig(prefix + "dt.png")
        plt.show()


if __name__ == "__main__":
    main()
