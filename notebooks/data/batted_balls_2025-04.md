# `batted_balls_2025-04.parquet`

Pitch-level Statcast data for the month of April 2025, restricted to batted-ball events (rows where `launch_speed` is non-null).

- **Source:** Baseball Savant via `pybaseball.statcast(start_dt="2025-04-01", end_dt="2025-04-30")`
- **Rows:** 37,453
- **Columns:** 21
- **File size:** ~580 KB
- **Sort order:** `game_pk` ascending, then `batter` ascending
- **Pre-filter applied:** `launch_speed IS NOT NULL` (excludes all swings-and-misses, called pitches, and balls in play without measured exit velocity)

## Companion file

For game-level covariates (venue, weather, attendance, etc.), see `games_2025-04.md` / `games_2025-04.parquet`. The two files join cleanly on `game_pk`.

## Fields

| Column | Type | Description | Units / Encoding |
|---|---|---|---|
| `game_pk` | `Int64` | MLBAM game identifier. **Join key** to `games_2025-04.parquet`. | Integer ID |
| `game_date` | `Date` | Date of the game the pitch occurred in. | ISO date (`YYYY-MM-DD`) |
| `batter` | `Int64` | MLBAM ID of the batter. Duplicated as `mlb_id` by the fetch script. | Integer ID, joinable to `player_name` via MLBAM |
| `player_name` | `String` | Batter's full name as reported by Statcast. | `"Last, First"` |
| `pitcher` | `Int64` | MLBAM ID of the pitcher who threw the pitch. | Integer ID |
| `home_team` | `String` | Home team abbreviation (e.g. `"NYM"`). | Three-letter team code |
| `away_team` | `String` | Away team abbreviation. | Three-letter team code |
| `pitch_name` | `String` | Human-readable pitch classification from Statcast's automated classifier. | e.g. `"4-Seam Fastball"`, `"Slider"`, `"Changeup"`, `"Curveball"`; may be `"null"` or unlabeled for unclassified pitches |
| `description` | `String` | Result of the pitch itself (ball/strike/in-play outcome). | e.g. `"ball"`, `"called_strike"`, `"swinging_strike"`, `"hit_into_play"` |
| `events` | `String` | Terminal event of the at-bat. Only populated on the final pitch of a plate appearance. | e.g. `"single"`, `"double"`, `"triple"`, `"home_run"`, `"field_out"`, `"strikeout"`, `"walk"`; null on non-terminal pitches of a PA |
| `launch_speed` | `Float64` | Exit velocity off the bat. | mph |
| `launch_angle` | `Int64` | Vertical launch angle of the ball off the bat. | degrees (negative = downward) |
| `hc_x` | `Float64` | Horizontal landing coordinate of the hit on Statcast's field plane. | feet, in Statcast's field-coordinate system (origin near home plate, +x toward third base side from catcher's view) |
| `hc_y` | `Float64` | Depth (distance-from-home) coordinate of the hit. | feet, in Statcast's field-coordinate system (+y away from home plate) |
| `hit_distance_sc` | `Int64` | Projected total distance the batted ball would have traveled. | feet |
| `estimated_ba_using_speedangle` | `Float64` | Expected batting average (`xBA`) for the batted ball given its launch speed and launch angle. | probability in [0, 1] |
| `estimated_slg_using_speedangle` | `Float64` | Expected slugging average (`xSLG`) for the batted ball given its launch speed and launch angle. | probability in [0, 1] for singles; up to ~4.0 weighted by total bases |
| `woba_value` | `Float64` | Linear weight (wOBA contribution) credited to the play. | per-play wOBA value |
| `babip_value` | `Int64` | BABIP flag (1 = ball in play on a BIP-eligible event, 0 = otherwise, null = N/A). | 0 / 1 / null |
| `iso_value` | `Int64` | ISO (isolated power) flag (1 = extra-base hit, 0 = otherwise, null = N/A). | 0 / 1 / null |
| `mlb_id` | `Int64` | Duplicate of `batter`, materialized via `.alias("mlb_id")` in the fetch script. | Integer ID |

## Notes

- All rows are batted balls by construction, so the expected value fields (`estimated_ba_using_speedangle`, `estimated_slg_using_speedangle`, `woba_value`) are populated for the play-outcome events and null for the non-terminal pitches of the same at-bat.
- `hc_x` / `hc_y` use Statcast's raw field coordinates (in feet), **not** the 0-250 spray-angle convention that some plotting libraries expect. To convert to spray angle in degrees (CCW from straight-away center field): `atan2(hc_x - 125.42, 198.27 - hc_y) * 180 / pi`. The constant offset reflects the camera/field calibration.
- `batter` and `mlb_id` carry identical values; the duplication is by design of the fetch script and harmless downstream, but only one is needed for joining to player identity.
- `babip_value` and `iso_value` arrive as `Int64` (0/1 flags), not `Float64`. If you need to do arithmetic on them, cast with `pl.col("babip_value").cast(pl.Float64)`.
- Statcast occasionally retroactively updates historical events. Row counts for any given date may drift slightly across re-fetches.
- **Schema history:** an earlier version of this file lacked `game_pk`, `home_team`, and `away_team`. Those three columns were added to support joining to `games_2025-04.parquet`; everything else is unchanged.
