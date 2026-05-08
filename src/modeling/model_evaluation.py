import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from pathlib import Path
from loguru import logger
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error,
)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import PLOTS_DIR, PROCESSED_DIR, VIZ_CONFIG

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    logger.warning("shap not installed. Run: pip install shap")


class ModelEvaluator:
    def __init__(self):
        self.output_dir = PLOTS_DIR / "model_eval"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir = PROCESSED_DIR / "eval_results"
        self.results_dir.mkdir(parents=True, exist_ok=True)

    # Core Metrics
    def regression_metrics(
        self,
        y_true: np.ndarray | pd.Series,
        y_pred: np.ndarray | pd.Series,
        model_name: str = "Model",
        variable: str = "variable",
    ) -> dict:
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true, y_pred = y_true[mask], y_pred[mask]

        rmse  = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae   = float(mean_absolute_error(y_true, y_pred))
        r2    = float(r2_score(y_true, y_pred))
        bias  = float(np.mean(y_pred - y_true))

        try:
            mape = float(mean_absolute_percentage_error(y_true, y_pred)) * 100
        except Exception:
            mape = float("nan")

        # Skill score vs climatological mean (persistence)
        clim_rmse = float(np.sqrt(np.mean((y_true - y_true.mean()) ** 2)))
        skill_score = 1 - (rmse / (clim_rmse + 1e-9))

        # Index of Agreement (Willmott 1981)
        numerator   = np.sum((y_true - y_pred) ** 2)
        denominator = np.sum(
            (np.abs(y_pred - y_true.mean()) + np.abs(y_true - y_true.mean())) ** 2
        )
        ioa = 1 - numerator / (denominator + 1e-9)

        # Pearson correlation
        corr, corr_p = stats.pearsonr(y_true, y_pred)

        metrics = {
            "model":       model_name,
            "variable":    variable,
            "n":           len(y_true),
            "rmse":        round(rmse, 4),
            "mae":         round(mae, 4),
            "r2":          round(r2, 4),
            "bias":        round(bias, 4),
            "mape_pct":    round(mape, 2),
            "skill_score": round(skill_score, 4),
            "ioa":         round(ioa, 4),
            "pearson_r":   round(corr, 4),
            "pearson_p":   round(corr_p, 6),
        }

        logger.info(
            f"[{model_name}] {variable} | RMSE={rmse:.3f} | MAE={mae:.3f} | "
            f"R²={r2:.3f} | Skill={skill_score:.3f} | IOA={ioa:.3f}"
        )
        return metrics

    def climatological_skill_score(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        baseline_type: str = "climatology",
    ) -> float:
        if baseline_type == "climatology":
            baseline_pred = np.full_like(y_true, y_true.mean())
        elif baseline_type == "persistence":
            baseline_pred = np.roll(y_true, 1)
            baseline_pred[0] = y_true[0]
        else:
            raise ValueError("baseline_type must be 'climatology' or 'persistence'")

        model_mse    = mean_squared_error(y_true, y_pred)
        baseline_mse = mean_squared_error(y_true, baseline_pred)
        skill = 1 - model_mse / (baseline_mse + 1e-9)
        return round(skill, 4)

    # Cross-Validation
    def walk_forward_cv(
        self,
        model_class,
        df: pd.DataFrame,
        feature_cols: list[str],
        target_col: str,
        n_splits: int = 5,
        min_train_ratio: float = 0.5,
        model_kwargs: dict | None = None,
    ) -> pd.DataFrame:
        model_kwargs = model_kwargs or {}
        n = len(df)
        min_train = int(n * min_train_ratio)
        fold_size = (n - min_train) // n_splits

        results = []
        for fold in range(n_splits):
            train_end = min_train + fold * fold_size
            test_end  = min(train_end + fold_size, n)

            if test_end >= n:
                break

            train = df.iloc[:train_end]
            test  = df.iloc[train_end:test_end]

            X_train = train[feature_cols]
            y_train = train[target_col]
            X_test  = test[feature_cols]
            y_test  = test[target_col]

            model = model_class(**model_kwargs)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            metrics = self.regression_metrics(y_test, y_pred, f"Fold {fold+1}", target_col)
            metrics["train_start"] = str(train.index[0].date())
            metrics["train_end"]   = str(train.index[-1].date())
            metrics["test_start"]  = str(test.index[0].date())
            metrics["test_end"]    = str(test.index[-1].date())
            results.append(metrics)

        cv_df = pd.DataFrame(results)
        logger.info(
            f"Walk-forward CV ({n_splits} folds) | "
            f"RMSE={cv_df['rmse'].mean():.3f}±{cv_df['rmse'].std():.3f} | "
            f"R²={cv_df['r2'].mean():.3f}"
        )
        return cv_df

    # Model Comparison
    def compare_models(
        self,
        predictions: dict[str, np.ndarray],
        y_true: np.ndarray,
        variable: str = "temperature",
    ) -> pd.DataFrame:
        rows = []
        for name, y_pred in predictions.items():
            metrics = self.regression_metrics(y_true, y_pred, name, variable)
            rows.append(metrics)

        comparison = pd.DataFrame(rows).sort_values("rmse")
        path = self.results_dir / f"model_comparison_{variable}.csv"
        comparison.to_csv(path, index=False)
        logger.success(f"Model comparison saved: {path}")
        return comparison

    # Residual Diagnostics
    def residual_diagnostics(
        self,
        y_true: np.ndarray | pd.Series,
        y_pred: np.ndarray | pd.Series,
        model_name: str = "Model",
        output_name: str | None = None,
    ) -> plt.Figure:
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        residuals = y_pred - y_true

        fig = plt.figure(figsize=(14, 10), dpi=VIZ_CONFIG["figure_dpi"])
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

        # Predicted vs Actual
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.scatter(y_true, y_pred, alpha=0.5, s=15, color="#1565C0")
        lim = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        ax1.plot(lim, lim, "r--", lw=1.5, label="1:1 line")
        ax1.set_xlabel("Observed")
        ax1.set_ylabel("Predicted")
        ax1.set_title(f"{model_name} — Predicted vs Actual")
        ax1.legend()
        r2 = r2_score(y_true, y_pred)
        ax1.text(0.05, 0.95, f"R² = {r2:.3f}", transform=ax1.transAxes,
                 va="top", fontsize=10, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7))

        # Residuals vs Predicted
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.scatter(y_pred, residuals, alpha=0.4, s=15, color="#E53935")
        ax2.axhline(0, color="black", lw=1)
        ax2.axhline(residuals.std(), color="grey", lw=0.8, ls="--", label="+1σ")
        ax2.axhline(-residuals.std(), color="grey", lw=0.8, ls="--", label="-1σ")
        ax2.set_xlabel("Predicted")
        ax2.set_ylabel("Residual")
        ax2.set_title("Residuals vs Predicted")
        ax2.legend(fontsize=9)

        # Residual time series
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(residuals, color="#1B5E20", lw=0.8, alpha=0.7)
        ax3.axhline(0, color="black", lw=1)
        ax3.fill_between(range(len(residuals)), residuals, 0,
                          alpha=0.25, color="#1B5E20")
        ax3.set_xlabel("Sample index")
        ax3.set_ylabel("Residual")
        ax3.set_title("Residual Time Series")

        # Q-Q plot
        ax4 = fig.add_subplot(gs[1, 1])
        stats.probplot(residuals, dist="norm", plot=ax4)
        ax4.set_title("Residual Q-Q Plot (Normality)")
        ax4.get_lines()[0].set(markerfacecolor="#1565C0", markersize=4, alpha=0.6)
        ax4.get_lines()[1].set(color="red", lw=1.5)

        fig.suptitle(f"Residual Diagnostics — {model_name}", fontsize=14, fontweight="bold")

        if output_name:
            path = self.output_dir / f"{output_name}_diagnostics.png"
            fig.savefig(path, dpi=VIZ_CONFIG["figure_dpi"], bbox_inches="tight")
            logger.success(f"Diagnostics plot saved: {path}")

        return fig

    def feature_importance_plot(
        self,
        importance_df: pd.DataFrame,
        model_name: str = "Model",
        output_name: str | None = None,
    ) -> plt.Figure:
        df = importance_df.sort_values("importance", ascending=True).tail(15)

        fig, ax = plt.subplots(figsize=(9, 6), dpi=VIZ_CONFIG["figure_dpi"])
        bars = ax.barh(df["feature"], df["importance"],
                       color="#1565C0", alpha=0.8, edgecolor="#0D47A1", linewidth=0.5)

        ax.set_xlabel("Importance Score", fontsize=11)
        ax.set_title(f"Feature Importances — {model_name}", fontsize=13, fontweight="bold")
        ax.grid(True, axis="x", alpha=0.3)

        for bar, val in zip(bars, df["importance"].values):
            ax.text(
                bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8,
            )

        plt.tight_layout()
        if output_name:
            path = self.output_dir / f"{output_name}_feature_importance.png"
            fig.savefig(path, dpi=VIZ_CONFIG["figure_dpi"], bbox_inches="tight")
            logger.success(f"Feature importance plot saved: {path}")
        return fig

    def model_comparison_plot(
        self,
        comparison_df: pd.DataFrame,
        output_name: str | None = None,
    ) -> plt.Figure:
        metrics = ["rmse", "mae", "r2", "skill_score", "ioa"]
        n_models = len(comparison_df)

        fig, axes = plt.subplots(1, len(metrics), figsize=(16, 5), dpi=VIZ_CONFIG["figure_dpi"])
        colors = plt.cm.Set2(np.linspace(0, 1, n_models))

        for ax, metric in zip(axes, metrics):
            vals = comparison_df[metric].values
            names = comparison_df["model"].values
            bar_colors = colors

            # For RMSE and MAE, lower is better — invert colour
            if metric in ("rmse", "mae"):
                bar_colors = colors[::-1]

            bars = ax.bar(names, vals, color=bar_colors, edgecolor="black", linewidth=0.5)
            ax.set_title(metric.upper(), fontsize=10, fontweight="bold")
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                        f"{v:.3f}", ha="center", fontsize=8)

        fig.suptitle("Model Comparison — Albania Climate Prediction", fontsize=13, fontweight="bold")
        plt.tight_layout()

        if output_name:
            path = self.output_dir / f"{output_name}_comparison.png"
            fig.savefig(path, dpi=VIZ_CONFIG["figure_dpi"], bbox_inches="tight")
            logger.success(f"Comparison plot saved: {path}")
        return fig

    # SHAP Explainability
    def _get_shap_explainer(self, model, X: pd.DataFrame):
        if not HAS_SHAP:
            raise ImportError("shap not installed. Run: pip install shap")

        model_type = type(model).__name__
        tree_types = (
            "XGBRegressor", "XGBClassifier",
            "LGBMRegressor", "LGBMClassifier",
            "GradientBoostingRegressor", "GradientBoostingClassifier",
            "RandomForestRegressor", "RandomForestClassifier",
            "ExtraTreesRegressor", "ExtraTreesClassifier",
        )
        if model_type in tree_types:
            logger.info(f"Using SHAP TreeExplainer for {model_type}.")
            return shap.TreeExplainer(model)
        else:
            logger.info(f"Using SHAP PermutationExplainer for {model_type} (slower).")
            return shap.PermutationExplainer(model.predict, X)

    def shap_values(
        self,
        model,
        X: pd.DataFrame,
        sample_size: int | None = None,
    ) -> shap.Explanation:
        if not HAS_SHAP:
            raise ImportError("shap not installed. Run: pip install shap")

        X_explain = X.copy()
        if sample_size and len(X_explain) > sample_size:
            X_explain = X_explain.sample(n=sample_size, random_state=42)
            logger.info(f"SHAP: sampled {sample_size} rows from {len(X)} for explanation.")

        if type(model).__name__ in ("XGBRegressor", "XGBClassifier") and hasattr(model, "get_booster"):
            try:
                import xgboost as xgb

                booster = model.get_booster()
                dmatrix = xgb.DMatrix(X_explain, feature_names=list(X_explain.columns))
                contribs = booster.predict(dmatrix, pred_contribs=True)
                sv = shap.Explanation(
                    values=contribs[:, :-1],
                    base_values=contribs[:, -1],
                    data=X_explain.to_numpy(),
                    feature_names=list(X_explain.columns),
                )
                logger.success(f"XGBoost native SHAP values computed: shape={sv.values.shape}")
                return sv
            except Exception as e:
                logger.warning(f"XGBoost native SHAP failed; falling back to SHAP explainer: {e}")

        explainer = self._get_shap_explainer(model, X_explain)
        sv = explainer(X_explain)
        logger.success(f"SHAP values computed: shape={sv.values.shape}")
        return sv

    def shap_summary_plot(
        self,
        model,
        X: pd.DataFrame,
        model_name: str = "Model",
        variable: str = "temperature",
        plot_type: str = "dot",
        max_features: int = 15,
        sample_size: int | None = 500,
        output_name: str | None = None,
    ) -> plt.Figure:
        if not HAS_SHAP:
            raise ImportError("shap not installed. Run: pip install shap")

        sv = self.shap_values(model, X, sample_size=sample_size)

        fig, ax = plt.subplots(figsize=(10, max(6, max_features * 0.45)),
                               dpi=VIZ_CONFIG["figure_dpi"])
        plt.sca(ax)

        shap.summary_plot(
            sv,
            X.loc[sv.data.index] if hasattr(sv, "data") and sv.data is not None else X,
            plot_type=plot_type,
            max_display=max_features,
            show=False,
            plot_size=None,
        )

        fig.suptitle(
            f"SHAP {'Beeswarm' if plot_type == 'dot' else 'Bar'} Summary\n"
            f"{model_name} — {variable}",
            fontsize=13, fontweight="bold", y=1.01,
        )
        plt.tight_layout()

        if output_name:
            path = self.output_dir / f"{output_name}_shap_summary_{plot_type}.png"
            fig.savefig(path, dpi=VIZ_CONFIG["figure_dpi"], bbox_inches="tight")
            logger.success(f"SHAP summary plot saved: {path}")

        return fig

    def shap_dependence_plot(
        self,
        model,
        X: pd.DataFrame,
        feature: str,
        interaction_feature: str = "auto",
        model_name: str = "Model",
        variable: str = "temperature",
        sample_size: int | None = 500,
        output_name: str | None = None,
    ) -> plt.Figure:
        if not HAS_SHAP:
            raise ImportError("shap not installed. Run: pip install shap")

        sv = self.shap_values(model, X, sample_size=sample_size)

        # Map feature name to index
        feature_cols = list(X.columns)
        if feature not in feature_cols:
            raise ValueError(f"Feature '{feature}' not in model features.")

        fig, ax = plt.subplots(figsize=(9, 6), dpi=VIZ_CONFIG["figure_dpi"])
        plt.sca(ax)

        shap.dependence_plot(
            feature,
            sv.values,
            X.iloc[:len(sv.values)],
            interaction_index=interaction_feature,
            ax=ax,
            show=False,
        )

        ax.set_title(
            f"SHAP Dependence: {feature}\n{model_name} — {variable}",
            fontsize=12, fontweight="bold",
        )
        plt.tight_layout()

        if output_name:
            path = self.output_dir / f"{output_name}_shap_dep_{feature}.png"
            fig.savefig(path, dpi=VIZ_CONFIG["figure_dpi"], bbox_inches="tight")
            logger.success(f"SHAP dependence plot saved: {path}")

        return fig

    def shap_waterfall_plot(
        self,
        model,
        X: pd.DataFrame,
        sample_idx: int = 0,
        model_name: str = "Model",
        variable: str = "temperature",
        max_features: int = 12,
        output_name: str | None = None,
    ) -> plt.Figure:
        if not HAS_SHAP:
            raise ImportError("shap not installed. Run: pip install shap")

        sv = self.shap_values(model, X)
        single = sv[sample_idx]

        fig, ax = plt.subplots(figsize=(10, max(6, max_features * 0.5)),
                               dpi=VIZ_CONFIG["figure_dpi"])
        plt.sca(ax)

        shap.waterfall_plot(single, max_display=max_features, show=False)

        fig.suptitle(
            f"SHAP Waterfall — Sample {sample_idx}\n{model_name} — {variable}",
            fontsize=12, fontweight="bold",
        )
        plt.tight_layout()

        if output_name:
            path = self.output_dir / f"{output_name}_shap_waterfall_{sample_idx}.png"
            fig.savefig(path, dpi=VIZ_CONFIG["figure_dpi"], bbox_inches="tight")
            logger.success(f"SHAP waterfall plot saved: {path}")

        return fig

    def shap_force_plot_html(
        self,
        model,
        X: pd.DataFrame,
        sample_idx: int = 0,
        model_name: str = "Model",
        output_name: str | None = None,
    ) -> Path | None:
        if not HAS_SHAP:
            raise ImportError("shap not installed. Run: pip install shap")

        explainer = self._get_shap_explainer(model, X)
        # Use raw shap_values array for force_plot (requires base_values scalar)
        sv_raw = explainer.shap_values(X.iloc[[sample_idx]])
        base_val = explainer.expected_value

        # For multi-output models, take first output
        if isinstance(sv_raw, list):
            sv_raw = sv_raw[0]
        if isinstance(base_val, (list, np.ndarray)):
            base_val = base_val[0]

        force = shap.force_plot(
            base_val,
            sv_raw[0],
            X.iloc[sample_idx],
            show=False,
        )

        output_name = output_name or f"{model_name.lower().replace(' ', '_')}"
        path = self.output_dir / f"{output_name}_shap_force_{sample_idx}.html"
        shap.save_html(str(path), force)
        logger.success(f"SHAP force plot (HTML) saved: {path}")
        return path

    def shap_mean_importance(
        self,
        model,
        X: pd.DataFrame,
        sample_size: int | None = 500,
    ) -> pd.DataFrame:
        if not HAS_SHAP:
            raise ImportError("shap not installed. Run: pip install shap")

        sv = self.shap_values(model, X, sample_size=sample_size)
        mean_abs = np.abs(sv.values).mean(axis=0)

        importance_df = pd.DataFrame({
            "feature":        list(X.columns),
            "mean_abs_shap":  mean_abs,
        }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

        logger.info(
            f"SHAP mean importance — top 5: "
            + ", ".join(
                f"{r.feature}={r.mean_abs_shap:.4f}"
                for _, r in importance_df.head(5).iterrows()
            )
        )
        return importance_df

    def shap_values_to_csv(
        self,
        model,
        X: pd.DataFrame,
        sample_size: int | None = None,
        output_name: str = "shap_values",
    ) -> Path:
        if not HAS_SHAP:
            raise ImportError("shap not installed. Run: pip install shap")

        sv = self.shap_values(model, X, sample_size=sample_size)
        X_sub = X.iloc[:len(sv.values)] if sample_size else X

        shap_df = pd.DataFrame(
            sv.values,
            columns=X.columns,
            index=X_sub.index,
        )
        path = self.results_dir / f"{output_name}.csv"
        shap_df.to_csv(path)
        logger.success(f"SHAP values saved: {path} ({shap_df.shape})")
        return path

    def shap_full_report(
        self,
        model,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        model_name: str = "Model",
        variable: str = "temperature",
        city: str = "Albania",
        sample_size: int = 500,
    ) -> dict[str, Path]:
        if not HAS_SHAP:
            raise ImportError("shap not installed. Run: pip install shap")

        logger.info(f"Generating SHAP full report for {model_name} — {city} — {variable} …")
        slug = f"{city.lower()}_{variable}_{model_name.lower().replace(' ', '_')}"
        saved = {}

        # Beeswarm summary
        fig = self.shap_summary_plot(
            model, X_test, model_name, variable,
            plot_type="dot", sample_size=sample_size, output_name=slug,
        )
        saved["beeswarm"] = self.output_dir / f"{slug}_shap_summary_dot.png"
        plt.close(fig)

        # Bar summary
        fig = self.shap_summary_plot(
            model, X_test, model_name, variable,
            plot_type="bar", sample_size=sample_size, output_name=slug,
        )
        saved["bar"] = self.output_dir / f"{slug}_shap_summary_bar.png"
        plt.close(fig)

        # Waterfall (middle test sample for a representative example)
        mid_idx = len(X_test) // 2
        fig = self.shap_waterfall_plot(
            model, X_test, sample_idx=mid_idx,
            model_name=model_name, variable=variable, output_name=slug,
        )
        saved["waterfall"] = self.output_dir / f"{slug}_shap_waterfall_{mid_idx}.png"
        plt.close(fig)

        # Force plot HTML
        try:
            html_path = self.shap_force_plot_html(
                model, X_test, sample_idx=mid_idx,
                model_name=model_name, output_name=slug,
            )
            if html_path:
                saved["force_html"] = html_path
        except Exception as e:
            logger.warning(f"Force plot failed: {e}")

        # Dependence plot for top feature
        try:
            importance_df = self.shap_mean_importance(model, X_test, sample_size=sample_size)
            top_feature = importance_df.iloc[0]["feature"]
            fig = self.shap_dependence_plot(
                model, X_test, feature=top_feature,
                model_name=model_name, variable=variable,
                sample_size=sample_size, output_name=slug,
            )
            saved["dependence"] = self.output_dir / f"{slug}_shap_dep_{top_feature}.png"
            plt.close(fig)
        except Exception as e:
            logger.warning(f"Dependence plot failed: {e}")

        # ── 6. SHAP values CSV
        csv_path = self.shap_values_to_csv(
            model, X_test, sample_size=sample_size, output_name=f"{slug}_shap_values"
        )
        saved["csv"] = csv_path

        logger.success(
            f"SHAP report complete — {len(saved)} outputs saved to {self.output_dir}"
        )
        return saved

    def save_metrics(self, metrics: dict | list[dict], name: str = "metrics") -> Path:
        if isinstance(metrics, dict):
            metrics = [metrics]
        df = pd.DataFrame(metrics)
        path = self.results_dir / f"{name}.csv"
        df.to_csv(path, index=False)
        logger.success(f"Metrics saved: {path}")
        return path