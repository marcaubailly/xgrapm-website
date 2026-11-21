# xGRAPM — Football Matchup & Lineup Impact Simulator

End-to-end machine learning application implementing **Expected Goals Regularized Adjusted Plus-Minus (xGRAPM)** with an **On-Ball Micro Prior** and **Off-Ball Macro Residual** pipeline across 2,513 Big 5 European league players.

## Architecture
1. **Stage 1 (Micro Prior):** Estimates on-ball player value ($\beta_{\text{Micro}}$) from event-level actions.
2. **Stage 2 (Macro Residual):** Fits a stint-level Ridge regression on non-penalty xG residuals ($\delta_{\text{Macro}}$) to capture off-ball positioning, pressing, and lineup synergy.
3. **Inference API (`xgrapm-api`):** Containerized FastAPI service deployed on Google Cloud Run (`https://xgrapm-api-753801721214.us-central1.run.app`).
4. **Interactive Dashboard (`xgrapm-website`):** Streamlit frontend deployed on Streamlit Community Cloud.
