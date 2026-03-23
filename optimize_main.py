from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, Bounds, NonlinearConstraint

from src.explorer import Explorer
from src.explorer.calculations_explorer import compute_cd_from_dataframe


def main():
    explorer = Explorer(
        project_name="spheres_case",
        output_dir="Explorer_outputs",
        total_points=30000,
        n_surface_each=180,
        seed=42,
    )

    # Initialize model once
    explorer.initialize_model(
        config_path="configs/spheres_fusion.yaml",
    )

    # Reusable paths
    output_dir = Path("Explorer_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    input_csv = output_dir / "opt_case.csv"
    pred_csv = output_dir / "opt_case_pred.csv"

    # Bounds: keep sphere well within domain and in top half
    bounds = Bounds(
        lb=[-0.5, 1e-6],   # x_min, y_min
        ub=[10.0, 5.0],   # x_max, y_max
    )

    # Non-overlap constraint with a small margin
    radius = explorer.radius
    margin = 0.05
    min_center_distance = 2.0 * radius + margin

    def nonoverlap_constraint(vars_xy):
        x_secondary, y_secondary = vars_xy
        center_distance = np.sqrt(x_secondary**2 + y_secondary**2)
        return center_distance - min_center_distance

    nonoverlap = NonlinearConstraint(
        fun=nonoverlap_constraint,
        lb=0.0,
        ub=np.inf,
    )

    # ------------------------------------------------------------------
    # Cache so optimizer + contour plot do not repeatedly recompute cases
    # ------------------------------------------------------------------
    eval_cache = {}

    def evaluate_cd(x_secondary, y_secondary):
        key = (round(float(x_secondary), 8), round(float(y_secondary), 8))
        if key in eval_cache:
            return eval_cache[key]

        # 1) Build input case
        explorer.create_single_case(
            x_secondary=float(x_secondary),
            y_secondary=float(y_secondary),
            csv_path=input_csv,
        )

        # 2) Predict
        pred_df, input_df = explorer.predict_from_csv(
            input_csv_path=input_csv,
            output_csv_path=pred_csv,
        )

        # 3) Bring over geometry columns needed for force integration
        pred_df["is_on_surface"] = input_df["is_on_surface"].values
        pred_df["Area[i] (m^2)"] = input_df["Area[i] (m^2)"].values
        pred_df["Area[j] (m^2)"] = input_df["Area[j] (m^2)"].values
        pred_df["Area[k] (m^2)"] = input_df["Area[k] (m^2)"].values

        # 4) Keep only secondary sphere surface rows
        n = explorer.n_surface_each
        secondary_df = pred_df.iloc[n:2 * n].copy()

        # 5) Compute drag proxy
        cd, force_vector, surface_mask = compute_cd_from_dataframe(secondary_df)

        eval_cache[key] = cd
        return cd

    def objective(vars_xy):
        x_secondary, y_secondary = vars_xy

        try:
            cd = evaluate_cd(x_secondary, y_secondary)
            print(f"x = {x_secondary:.6f}, y = {y_secondary:.6f}, Cd = {cd:.6f}")
            return cd
        except ValueError as e:
            print(f"x = {x_secondary:.6f}, y = {y_secondary:.6f}, INVALID: {e}")
        return 1e12

    # Initial guess
    x0 = np.array([2.75, 1e-6], dtype=float)

    result = minimize(
        fun=objective,
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        constraints=[nonoverlap],
        options={
            "maxiter": 30,
            "disp": True,
            "ftol": 1e-4,
        },
    )

    print("\nOptimization result")
    print("-------------------")
    print("Success:", result.success)
    print("Message:", result.message)
    print("Optimal x:", result.x[0])
    print("Optimal y:", result.x[1])
    print("Minimum Cd:", result.fun)

    # ------------------------------------------------------------------
    # Contour plot over bounded design space
    # ------------------------------------------------------------------
    def plot_objective_contours(result, nx=60, ny=60):
        x_min, y_min = bounds.lb
        x_max, y_max = bounds.ub

        x_vals = np.linspace(x_min, x_max, nx)
        y_vals = np.linspace(y_min, y_max, ny)
        X, Y = np.meshgrid(x_vals, y_vals)

        Z = np.full_like(X, np.nan, dtype=float)

        for i in range(ny):
            for j in range(nx):
                x_ij = X[i, j]
                y_ij = Y[i, j]

                # Only evaluate feasible points
                if nonoverlap_constraint([x_ij, y_ij]) >= 0.0:
                    Z[i, j] = evaluate_cd(x_ij, y_ij)

        valid = np.isfinite(Z)
        if not np.any(valid):
            print("No feasible points found for contour plot.")
            return

        zmin = np.nanmin(Z)
        zmax = np.nanmax(Z)

        plt.figure(figsize=(9, 6))

        contourf = plt.contourf(X, Y, Z, levels=30)
        plt.colorbar(contourf, label="Cd")

        contour = plt.contour(X, Y, Z, levels=15, linewidths=0.8)
        plt.clabel(contour, inline=True, fontsize=8)

        # Plot non-overlap boundary: x^2 + y^2 = min_center_distance^2
        theta = np.linspace(0.0, np.pi / 2.0, 400)
        x_circle = min_center_distance * np.cos(theta)
        y_circle = min_center_distance * np.sin(theta)

        # Only keep boundary portion inside bounds
        mask = (
            (x_circle >= x_min) & (x_circle <= x_max) &
            (y_circle >= y_min) & (y_circle <= y_max)
        )
        plt.plot(
            x_circle[mask],
            y_circle[mask],
            "r--",
            linewidth=2,
            label="Non-overlap boundary",
        )

        # Initial guess
        plt.plot(x0[0], x0[1], "wo", markersize=8, markeredgecolor="k", label="Initial guess")

        # Optimum
        plt.plot(result.x[0], result.x[1], "r*", markersize=14, label="Optimizer minimum")

        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.xlabel("x_secondary")
        plt.ylabel("y_secondary")
        plt.title("Cd contour in bounded design space")
        plt.legend()
        plt.tight_layout()

        contour_path = output_dir / "objective_contour.png"
        plt.savefig(contour_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved contour plot to: {contour_path}")

    plot_objective_contours(result)

    # ------------------------------------------------------------------
    # Re-run optimal case and plot field
    # ------------------------------------------------------------------
    x_opt, y_opt = result.x

    print("\nRebuilding optimal case for plotting...")

    optimal_input_csv = output_dir / "optimal_case.csv"
    optimal_pred_csv = output_dir / "optimal_case_pred.csv"

    # 1) Build optimal geometry
    explorer.create_single_case(
        x_secondary=float(x_opt),
        y_secondary=float(y_opt),
        csv_path=optimal_input_csv,
    )

    # 2) Predict
    pred_df, input_df = explorer.predict_from_csv(
        input_csv_path=optimal_input_csv,
        output_csv_path=optimal_pred_csv,
    )

    # 3) Merge geometry info
    pred_df["is_on_surface"] = input_df["is_on_surface"].values
    pred_df["Area[i] (m^2)"] = input_df["Area[i] (m^2)"].values
    pred_df["Area[j] (m^2)"] = input_df["Area[j] (m^2)"].values
    pred_df["Area[k] (m^2)"] = input_df["Area[k] (m^2)"].values

    # 4) Plot desired field
    explorer.plot_predicted_field(
        pred_df=pred_df,
        input_df=input_df,
        field_name="Absolute Pressure (Pa)",
        output_plot=output_dir / "optimal_pressure.png",
        levels=100,
    )

    print("Optimal case plotted.")


if __name__ == "__main__":
    main()