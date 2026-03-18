"""
Usage:
    python3 get_goes_data_LSJ.py --start-date YYYY-MM-DD [--end-date YYYY-MM-DD]
Example:
    python3 get_goes_data_LSJ.py --start-date 2021-05-01 --end-date 2021-05-31
"""

import os
import requests
import pandas as pd
from shapely.geometry import Point, Polygon
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
from datetime import datetime
import argparse

# ---------------- FILE PATHS ----------------
data_in_path = '/Users/cronjobs/src/GOES/'
out_dir = '/Volumes/T7/data/epa-habs/GOES'
lat_csv = r"/Users/cronjobs/src/GOES/LATITUDE.csv"
lon_csv = r"/Users/cronjobs/src/GOES/LONGITUDE.csv"
kml_file = r"/Users/cronjobs/src/GOES/LSJmasking.kml"
base_url = "https://academic.uprm.edu/hdc/GOES-PRWEB_RESULTS/solar_radiation/"
output_folder_daily = os.path.join(out_dir,'daily_PRWEB_RESULTS_solrad')
# output_folder_averaged = os.path.join(out_dir,'averaged')ß
os.makedirs(output_folder_daily, exist_ok=True)
# os.makedirs(output_folder_averaged, exist_ok=True)


# ---------------- USER INPUT ----------------
parser = argparse.ArgumentParser(description="Download and process solar radiation CSVs for SAN_JOSE_LAKE.")
parser.add_argument("--start-date", "-s", required=True, help="Start date (YYYY-MM-DD)")
parser.add_argument("--end-date", "-e", help="End date (YYYY-MM-DD). If omitted, processes from start date onward.")
args = parser.parse_args()

start_date_str = args.start_date
start_date = datetime.strptime(start_date_str, "%Y-%m-%d")

end_date = None
if args.end_date:
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d")

# ---------------- PARSING FUNCTIONS ----------------
def parse_csv_cell_row(cell_row):
    return [float(x) for x in str(cell_row).split(',') if x.strip()]

def load_polygons_from_kml(kml_file_path):
    ns = {'kml': 'http://www.opengis.net/kml/2.2'}
    tree = ET.parse(kml_file_path)
    root = tree.getroot()
    polygons = []
    for placemark in root.findall('.//kml:Placemark', ns):
        coords_text = placemark.find('.//kml:coordinates', ns)
        if coords_text is None:
            continue
        coords = []
        for line in coords_text.text.strip().split():
            lon, lat, *_ = map(float, line.split(','))
            coords.append((lon, lat))
        polygons.append(Polygon(coords))
    print(f"✅ Total polygons loaded: {len(polygons)}")
    return polygons

# ---------------- LOAD LAT/LON ----------------
lat_values = []
with open(lat_csv) as f:
    for line in f:
        lat_values.extend(parse_csv_cell_row(line))

lon_values = []
with open(lon_csv) as f:
    for line in f:
        lon_values.extend(parse_csv_cell_row(line))

grid_size = len(lat_values)
print(f"✅ Grid size: {grid_size} points")

# ---------------- LOAD POLYGONS ----------------
polygons = load_polygons_from_kml(kml_file)

# ---------------- GET CSV FILES FROM WEBSITE ----------------
resp = requests.get(base_url)
soup = BeautifulSoup(resp.text, 'html.parser')
csv_links = [a['href'] for a in soup.find_all('a') if a['href'].endswith('.csv')]

# ---------------- PROCESS EACH DAY ----------------
for csv_file in csv_links:
    try:
        # Extract date from filename
        file_date_str = csv_file.replace('solar_radiation', '').replace('.csv', '')
        file_date = datetime.strptime(file_date_str, "%Y%m%d")
        
        if file_date < start_date:
            continue
        if end_date and file_date > end_date:
            continue

        print(f"\n📥 Processing {csv_file} ({file_date.date()})...")

        # Download magnitude CSV
        csv_url = base_url + csv_file
        r = requests.get(csv_url)
        r.raise_for_status()
        mag_lines = r.text.strip().splitlines()

        # Flatten magnitude CSV
        mag_values = []
        for line in mag_lines:
            # convert from MJ/m^2/day to Watt/m^2
            vals = parse_csv_cell_row(line)
            converted = [v * 1e6 / (3600 * 24) for v in vals]
            mag_values.extend(converted)

        if len(mag_values) != grid_size:
            print(f"⚠️ Total magnitude values ({len(mag_values)}) do not match lat/lon grid ({grid_size}). Skipping this file.")
            continue

        # Combine points
        points = list(zip(lat_values, lon_values, mag_values))
        print(f"✅ Total points loaded: {len(points)}")

        # compute mean of magnitude values
        # mag_values_mean.append(sum(mag_values) / len(mag_values))

        # Filter points inside polygons
        points_inside = []
        for lat, lon, mag in points:
            pt = Point(lon, lat)
            if any(pt.within(poly) for poly in polygons):
                points_inside.append((lat, lon, mag))

        print(f"✅ Points inside polygon: {len(points_inside)}")

        # Save daily result
        output_csv = os.path.join(output_folder_daily, f"{file_date.date()}.csv")
        df_inside = pd.DataFrame(points_inside, columns=['Latitude', 'Longitude', 'Magnitude'])
        df_inside.to_csv(output_csv, index=False)
        print(f"💾 Saved to {output_csv}")

    except Exception as e:
        print(f"❌ Error processing {csv_file}: {e}")
