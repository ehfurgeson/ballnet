import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    import os
    import nflreadpy as nfl
    import polars as pl
    from supabase import create_client, Client

    # ==============================================================================
    # CONFIGURATION & SETUP
    # ==============================================================================

    # Supabase Credentials (Ensure these are set in your environment)
    # SUPABASE_URL = os.environ.get("SUPABASE_URL")
    # SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
    # supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

    SEASON = 2025 # Define the current season you are pulling
    CURRENT_WEEK = 3 # You can compute this dynamically based on the current date

    # ==============================================================================
    # 1. THRESHOLD & MATH HELPERS
    # ==============================================================================

    def get_qb_ramp_and_hold_threshold(current_week: int) -> int:
        """
        Ramp-and-Hold Logic for QB Pass Attempts.
        Ramps up by 15 attempts per week, capping at 40 (the stabilization point).
        """
        return min(current_week * 15, 40)

    def calculate_fractional_percentile(df: pl.DataFrame, metric_col: str, asc: bool = True) -> pl.Expr:
        """
        Calculates the Fractional Percentile Rank to elegantly handle ties.
        Formula: PR = ((c + 0.5f) / N) * 100
    
        Mathematical shortcut using Polars:
        The Polars `rank(method="average")` perfectly equates to `c + (f + 1)/2`.
        Therefore, (Average Rank - 0.5) is mathematically identical to (c + 0.5f).
        """
        N = df.height
    
        # If asc=False (e.g., Sacks Taken where lower is better), we reverse the rank
        return (
            (pl.col(metric_col).rank(method="average", descending=not asc) - 0.5) / N * 100
        ).round(1).alias(f"{metric_col}_percentile")

    # ==============================================================================
    # 2. DATA GATHERING
    # ==============================================================================

    def fetch_and_merge_data(season: int) -> pl.DataFrame:
        print(f"Fetching data for season {season}...")
    
        # Load basic player stats (Returns a Polars DataFrame)
        basic_stats = nfl.load_player_stats(seasons=[season])
    
        # Load Next Gen Stats passing metrics (Returns a Polars DataFrame)
        ngs_passing = nfl.load_nextgen_stats(stat_type="passing", seasons=[season])
    
        # Filter basic stats to just the overall season aggregations per player
        basic_season = (
            basic_stats
            .group_by(["player_id", "player_display_name", "position", "recent_team"])
            .agg([
                pl.col("attempts").sum(),
                pl.col("passing_yards").sum(),
                pl.col("passing_tds").sum(),
                pl.col("interceptions").sum(),
                pl.col("sack_fumbles_lost").sum(), # Proxy for sacks/fumbles depending on columns available
                (pl.col("completions").sum() / pl.col("attempts").sum()).alias("completion_percentage")
            ])
        )
    
        # Filter NGS passing stats to the season level (week 0 usually denotes season totals in NGS)
        ngs_season = (
            ngs_passing.filter(pl.col("week") == 0)
            .select([
                "player_gsis_id", 
                "avg_time_to_throw", 
                "cpoe", 
                "expected_completion_percentage",
                "avg_air_distance"
            ])
        )
    
        # Merge basic stats with NGS metrics on the player ID
        merged_df = basic_season.join(
            ngs_season, 
            left_on="player_id", 
            right_on="player_gsis_id", 
            how="left"
        )
    
        return merged_df

    # ==============================================================================
    # 3. PERCENTILE PROCESSING
    # ==============================================================================

    def process_qb_percentiles(df: pl.DataFrame, current_week: int) -> pl.DataFrame:
        print("Processing QB Percentiles...")
    
        # 1. Filter down to Quarterbacks
        qbs = df.filter(pl.col("position") == "QB")
    
        # 2. Apply Ramp-and-Hold Threshold
        min_attempts = get_qb_ramp_and_hold_threshold(current_week)
        qualified_qbs = qbs.filter(pl.col("attempts") >= min_attempts)
    
        print(f"Qualified QBs based on {min_attempts} attempts: {qualified_qbs.height}")
    
        # 3. Calculate 1-100 Percentiles for all desired metrics
        percentile_df = qualified_qbs.with_columns([
            calculate_fractional_percentile(qualified_qbs, "passing_yards"),
            calculate_fractional_percentile(qualified_qbs, "passing_tds"),
            calculate_fractional_percentile(qualified_qbs, "interceptions", asc=False), # Fewer is better
            calculate_fractional_percentile(qualified_qbs, "completion_percentage"),
            calculate_fractional_percentile(qualified_qbs, "cpoe"),
            calculate_fractional_percentile(qualified_qbs, "avg_time_to_throw", asc=False) # Faster TTT is usually "red"
        ])
    
        return percentile_df

    # ==============================================================================
    # 4. DATABASE SYNC
    # ==============================================================================

    # def push_to_supabase(df: pl.DataFrame, table_name: str):
    #     print(f"Pushing {df.height} records to Supabase table: '{table_name}'")
    
    #     # Convert Polars DataFrame to a list of Python dictionaries for the JSON payload
    #     # Drop any nulls if necessary, or let Supabase handle them
    #     records = df.to_dicts()
    
    #     # Upsert the data (requires 'player_id' to be set as the Primary Key in Supabase)
    #     response = supabase.table(table_name).upsert(records).execute()
    #     print("Push successful!")

    # ==============================================================================
    # MAIN EXECUTION
    # ==============================================================================

    if __name__ == "__main__":
        # Step 1: Pull and merge
        raw_data = fetch_and_merge_data(SEASON)
    
        # Step 2: Process position-specific percentiles
        qb_savant_data = process_qb_percentiles(raw_data, CURRENT_WEEK)
    
        # Show data
        print(qb_savant_data.head(10))

        # Step 3: Push to backend
        # push_to_supabase(qb_savant_data, "qb_savant_metrics")
    return


if __name__ == "__main__":
    app.run()
