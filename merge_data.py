import io
import pandas as pd

LOCATIONS = ["Cuttack", "Ludhiana", "Maruteru", "Palampur", "Raipur", "Rajendranagar"]


def load_nasa_file(location):
    with open("nasa_daily/" + location + ".csv") as f:
        text = f.read().split("-END HEADER-")[1]
    df = pd.read_csv(io.StringIO(text))
    df = df.replace(-999, float("nan"))
    df["Location"] = location
    df["Standard Week"] = ((df["DOY"] - 1) // 7 + 1).clip(upper=52)
    return df


def make_weekly(daily):
    weekly = daily.groupby(["Location", "YEAR", "Standard Week"]).agg(
        NASA_MaxT=("T2M_MAX", "mean"),
        NASA_MinT=("T2M_MIN", "mean"),
        NASA_RH=("RH2M", "mean"),
        NASA_RF_mm=("PRECTOTCORR", "sum"),
        NASA_WS_kmph=("WS2M", "mean"),
    ).reset_index()
    weekly["NASA_WS_kmph"] = weekly["NASA_WS_kmph"] * 3.6
    weekly = weekly.rename(columns={"YEAR": "Observation Year"})
    return weekly.round(2)


def main():
    all_files = [load_nasa_file(loc) for loc in LOCATIONS]
    daily = pd.concat(all_files)
    weekly = make_weekly(daily)

    rice = pd.read_csv("RICE.csv")
    merged = rice.merge(weekly, on=["Location", "Observation Year", "Standard Week"], how="left")

    weekly.to_csv("nasa_weekly_all_locations.csv", index=False)
    merged.to_csv("RICE_merged.csv", index=False)

    print("Rows in RICE.csv:", len(rice))
    print("Rows in merged file:", len(merged))
    print("Rows with no NASA data:", merged["NASA_MaxT"].isna().sum())


main()
