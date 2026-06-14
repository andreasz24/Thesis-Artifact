from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────────────────────────────────
# src/config.py  ->  src/  ->  repo root
ROOT = Path(__file__).resolve().parent.parent

DATA_PATH   = ROOT / "data" / "corporateCreditRatingWithFinancialRatios.csv"
OUTPUTS_DIR = ROOT / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# Shared artifact produced by notebook 1 and consumed by notebooks 2-5
SPLITS_PATH = OUTPUTS_DIR / "splits.pkl"

# Per-model saved results (consistent format, used in the comparison notebook)
XGB_RESULTS_PATH      = OUTPUTS_DIR / "xgb_results.pkl"
TABPFN_RESULTS_PATH   = OUTPUTS_DIR / "tabpfn3_results.pkl"
AUTOGLUON_RESULTS_PATH = OUTPUTS_DIR / "autogluon_results.pkl"

# AutoGluon writes large model folders here (git-ignored)
AG_GRID_DIR = OUTPUTS_DIR / "ag_grid"
AG_BEST_DIR = OUTPUTS_DIR / "autogluon_best"

# ──────────────────────────────────────────────────────────────────────────
# DATA / LABELS
# ──────────────────────────────────────────────────────────────────────────
# The 16 financial-ratio features used as model inputs
FEATURE_COLS = [
    "Current Ratio", "Long-term Debt / Capital", "Debt/Equity Ratio",
    "Gross Margin", "Operating Margin", "EBIT Margin", "EBITDA Margin",
    "Pre-Tax Profit Margin", "Net Profit Margin", "Asset Turnover",
    "ROE - Return On Equity", "Return On Tangible Equity",
    "ROA - Return On Assets", "ROI - Return On Investment",
    "Operating Cash Flow Per Share", "Free Cash Flow Per Share",
]

# Broad rating classes in ascending order of creditworthiness.
# Encoded as consecutive integers: D=0, C=1, ... AAA=9.
RATING_ORDER  = ["D", "C", "CC", "CCC", "B", "BB", "BBB", "A", "AA", "AAA"]
RATING_LABELS = RATING_ORDER  # alias used for confusion-matrix tick labels

# Column holding the company name — used as the grouping variable for the
# leakage-safe train/test split.
GROUP_COL  = "Corporation"
RATING_COL = "Rating"
TARGET_COL = "Rating_Num"

# ──────────────────────────────────────────────────────────────────────────
# REPRODUCIBILITY / SPLIT
# ──────────────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE    = 0.2
WINSOR_LO    = 0.01   # 1st percentile
WINSOR_HI    = 0.99   # 99th percentile

# ──────────────────────────────────────────────────────────────────────────
# EXPERIMENT GRIDS
# ──────────────────────────────────────────────────────────────────────────
# TabPFN-3 n_estimators sweep
N_ESTIMATORS_RANGE = [1, 2, 4, 8, 16]

# AutoGluon 2x2 grid
FOLDS      = [3, 8]    # num_bag_folds
STACKS     = [1, 2]    # num_stack_levels
TIME_LIMIT = 300       # seconds per configuration
