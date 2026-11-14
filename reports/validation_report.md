# Data Validation Report

Run at: `2026-11-14T19:58:20`

Severity **error** = record quarantined to `data/errors/<table>_errors.csv`. Severity **warning** = record kept, issue fixed in the preparation stage and listed in `data/errors/<table>_warnings.csv`.

## Summary

| Table | Input rows | Valid rows | Quarantined | With warnings | Pass rate |
|---|---:|---:|---:|---:|---:|
| users | 1,010 | 1,000 | 10 | 147 | 99.01% |
| items | 606 | 600 | 6 | 47 | 99.01% |
| interactions | 59,891 | 56,681 | 3,210 | 0 | 94.64% |

## users

| Rule | Severity | Description | Failed rows |
|---|---|---|---:|
| `missing_user_id` | error | user_id is null/blank | 0 |
| `invalid_user_id_format` | error | user_id must match U##### | 0 |
| `duplicate_user_id` | error | user_id already seen (duplicate record) | 10 |
| `age_not_numeric` | warning | age is not a number | 3 |
| `age_out_of_range` | warning | age outside [10, 100] | 7 |
| `missing_age` | warning | age is null | 21 |
| `missing_country` | warning | country is null | 20 |
| `invalid_signup_date` | warning | signup_date not parseable as YYYY-MM-DD | 5 |
| `nonstandard_gender` | warning | gender not in {M, F, Other} | 101 |

## items

| Rule | Severity | Description | Failed rows |
|---|---|---|---:|
| `missing_item_id` | error | item_id is null/blank | 0 |
| `invalid_item_id_format` | error | item_id must match I##### | 0 |
| `duplicate_item_id` | error | item_id already seen (duplicate record) | 6 |
| `missing_title` | error | title is null | 0 |
| `missing_genres` | warning | genres is null | 12 |
| `release_year_out_of_range` | warning | release_year outside [1900, current year] | 6 |
| `nonstandard_genre_case` | warning | genres not Title-Case | 29 |

## interactions

| Rule | Severity | Description | Failed rows |
|---|---|---|---:|
| `missing_user_id` | error | user_id is null | 148 |
| `missing_item_id` | error | item_id is null | 0 |
| `missing_rating` | error | rating is null | 305 |
| `missing_timestamp` | error | timestamp is null | 75 |
| `rating_wrong_type` | error | rating is not numeric | 293 |
| `rating_out_of_range` | error | rating outside [1, 5] | 595 |
| `timestamp_unparseable` | error | timestamp not in format %Y-%m-%d %H:%M:%S | 219 |
| `timestamp_in_future` | error | timestamp later than pipeline run time | 302 |
| `timestamp_too_old` | error | timestamp earlier than 2015-01-01 | 149 |
| `orphan_user_id` | error | user_id not present in validated users (referential integrity) | 298 |
| `orphan_item_id` | error | item_id not present in validated items (referential integrity) | 296 |
| `exact_duplicate` | error | row is an exact duplicate of an earlier row | 587 |
