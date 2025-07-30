"""Helper functions for the epoch of reheating."""


def is_instant_reheating(N_star, rho_reh_GeV, w_reh, DeltaN_reh, lnR_rad):
    """Check whether any given parameter combination amounts to instant reheating.

    Parameters
    ----------
    N_star : float
        Number of e-folds of inflation after horizon crossing of pivot scale `K_STAR`.
    rho_reh_GeV : float
        Energy density at the end of reheating in GeV.
    w_reh : float
        Equation of state parameter during reheating.
    DeltaN_reh : float
        Number of e-folds during reheating.
    lnR_rad : float
        Contribution to the calibration of `N_star` or `N_end` that comes from reheating
        but is agnostic to the details of reheating. See Martin & Ringeval (2010).
        https://arxiv.org/abs/1004.5525

    """
    if N_star is not None:
        return False
    elif (
        rho_reh_GeV is None and w_reh is None and DeltaN_reh is None and lnR_rad is None
        or w_reh is not None and w_reh == 1 / 3
        or DeltaN_reh is not None and DeltaN_reh == 0
        or lnR_rad is not None and lnR_rad == 0
    ):
        if (
            w_reh is not None and w_reh != 1 / 3
            or DeltaN_reh is not None and DeltaN_reh != 0
            or lnR_rad is not None and lnR_rad != 0
            or rho_reh_GeV is not None
        ):
            raise ValueError(
                f"When `w_reh=1/3` or `DeltaN_reh=0` or `lnR_rad=0`, then we are assuming instant "
                f"reheating, which requires all other parameters to also match the instant "
                f"reheating condition or to be set to `None`. However, got w_reh={w_reh}, "
                f"DeltaN_reh={DeltaN_reh}, lnR_rad={lnR_rad}, and rho_reh_GeV={rho_reh_GeV}."
            )
        return True
    else:
        return False
