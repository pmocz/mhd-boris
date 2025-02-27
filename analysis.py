# Analysis of Boris Simulation results

import numpy as np
import matplotlib.pyplot as plt


def main():
    for prob_id in [1]:
        cf_limits = [0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]

        rho = {}
        P_B = {}
        v = {}
        ca = {}
        cf = {}
        dt = {}

        # Load the data (rho, P_B, dt)
        prefix = "p" + str(prob_id) + "_"
        for cf_limit in cf_limits:
            rho[cf_limit] = np.load(
                "output/" + prefix + "data_rho_" + str(cf_limit) + ".npy"
            )
            P_B[cf_limit] = np.load(
                "output/" + prefix + "data_P_B_" + str(cf_limit) + ".npy"
            )
            v[cf_limit] = np.load(
                "output/" + prefix + "data_v_" + str(cf_limit) + ".npy"
            )
            ca[cf_limit] = np.load(
                "output/" + prefix + "data_ca_" + str(cf_limit) + ".npy"
            )
            cf[cf_limit] = np.load(
                "output/" + prefix + "data_cf_" + str(cf_limit) + ".npy"
            )
            dt[cf_limit] = np.load(
                "output/" + prefix + "data_dt_" + str(cf_limit) + ".npy"
            )

        # stack the rho next to each other and plot it as an image
        rho_all = np.hstack([rho[cf_limit] for cf_limit in cf_limits])
        P_B_all = np.hstack([P_B[cf_limit] for cf_limit in cf_limits])
        v_all = np.hstack([v[cf_limit] for cf_limit in cf_limits])
        ca_all = np.hstack([ca[cf_limit] for cf_limit in cf_limits])
        cf_all = np.hstack([cf[cf_limit] for cf_limit in cf_limits])

        plt.figure(figsize=(12, 4))
        plt.imshow(rho_all, aspect=1, cmap="jet")
        plt.colorbar()
        plt.xlabel("cf_max = " + str(cf_limits))
        plt.savefig(prefix + "rho.png")
        plt.show()

        plt.figure(figsize=(12, 4))
        plt.imshow(P_B_all, aspect=1, cmap="jet")
        plt.colorbar()
        plt.xlabel("cf_max = " + str(cf_limits))
        plt.savefig(prefix + "P_B.png")
        plt.show()

        plt.figure(figsize=(12, 4))
        plt.imshow(v_all, aspect=1, cmap="jet")
        plt.colorbar()
        plt.xlabel("cf_max = " + str(cf_limits))
        plt.savefig(prefix + "v.png")
        plt.show()

        plt.figure(figsize=(12, 4))
        plt.imshow(ca_all, aspect=1, cmap="jet")
        plt.colorbar()
        plt.xlabel("cf_max = " + str(cf_limits))
        plt.savefig(prefix + "ca.png")
        plt.show()

        plt.figure(figsize=(12, 4))
        plt.imshow(cf_all, aspect=1, cmap="jet")
        plt.colorbar()
        plt.xlabel("cf_max = " + str(cf_limits))
        plt.savefig(prefix + "cf.png")
        plt.show()

        # plot the dt for each cf_limit
        plt.figure()
        for cf_limit in cf_limits:
            plt.plot(
                np.cumsum(dt[cf_limit]), dt[cf_limit], label="cf_max = " + str(cf_limit)
            )
        plt.legend(loc="lower left")
        plt.xlabel("t")
        plt.savefig(prefix + "dt.png")
        plt.show()

        return


if __name__ == "__main__":
    main()
