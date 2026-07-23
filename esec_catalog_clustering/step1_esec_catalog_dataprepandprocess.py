import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
import io
import time
import re
import glob
import numpy as np

from obspy import read
from obspy.clients.fdsn import Client
from obspy import UTCDateTime


#---------------------------- GLOBAL VARIABLES------------------------------
# Set Global Variables
#---------------------------------------------------------------------------
DB_DIR = "/data/sswar_files/PRJ_GIS_QA/esce_catalog_test/final_test_results_for_production/metadata/DB_Files"
DB_FILES_PATH = "/data/sswar_files/PRJ_GIS_QA/esce_catalog_test/final_test_results_for_production/metadata/DB_Files/*.csv" ##Set DB_FILES_PATH to None When no DB Files Available

SPUD_URL = "https://ds.iris.edu/spud/esec"
CATALOG_PATH = "/data/sswar_files/PRJ_GIS_QA/esce_catalog_test/final_test_results_for_production/Master_ESEC_Catalog_vel_all.csv"
SAC_OUT_DIR = "/data/sswar_files/PRJ_GIS_QA/esce_catalog_test/final_test_results_for_production/esec_resp_removed_vel"
SAC_PROC_DIR = "/data/sswar_files/PRJ_GIS_QA/esce_catalog_test/final_test_results_for_production/esec_processed_vel"

EVENT_COLS = ["Eventid", "Name", "Starttime", "Endtime", "Latitude", "Longitude", 
                "Type", "AreaTotal", "AreaSource", "AreaSourceLow", "AreaSourceHigh", 
                "Volume", "VolumeLow", "VolumeHigh", "Mass", "MassLow", "MassHigh", 
                "H", "HLow", "HHigh"]


##---------------------------- Helper Functions ----------------------------
def _parse_db_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    
    sections = {}
    current_section = None
    current_lines = []
    
    lines = content.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i]
        # Look for section headers like:
        # #
        # # events
        # #
        if line.strip() == '#' and i + 2 < len(lines) and lines[i+2].strip() == '#':
            section_name = lines[i+1].strip('# ').strip()
            if current_section and current_lines:
                sections[current_section] = current_lines
            current_section = section_name
            current_lines = []
            i += 3
            continue
        
        if current_section:
            if line.startswith('# ') and '|' in line:
                current_lines.append(line.strip('# '))
            elif line.strip() and not line.startswith('#'):
                current_lines.append(line.strip())
        i += 1
        
    if current_section and current_lines:
        sections[current_section] = current_lines
        
    dataframes = {}
    for sec, sec_lines in sections.items():
        if not sec_lines:
            continue
        df = pd.read_csv(io.StringIO('\n'.join(sec_lines)), sep='|', na_values=['None', 'none', 'NA', ''])
        dataframes[sec] = df
        
    return dataframes

def _download_db_files():
    print("Fetching SPUD ESEC catalog page...")
    resp = requests.get(SPUD_URL)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, 'html.parser')
    
    tbody = soup.find('tbody', id='dataTable_data')
    if not tbody:
        raise ValueError("Could not find table body with id 'dataTable_data'")
    
    rows = tbody.find_all('tr')
    data_rks = [row.get('data-rk') for row in rows if row.get('data-rk')]
    print(f"Found {len(data_rks)} events in the catalog.")
    
    os.makedirs(DB_DIR, exist_ok=True)
    downloaded_files = []
    
    for idx, rk in enumerate(data_rks):
        print(f"Processing event {idx+1}/{len(data_rks)} (data-rk: {rk})")
        detail_url = f"{SPUD_URL}/{rk}"
        d_resp = requests.get(detail_url)
        d_soup = BeautifulSoup(d_resp.text, 'html.parser')
        
        # Look for attachments table
        # We search for links that contain 'window.open' and '/spudservice/data/'
        links = d_soup.find_all('a', onclick=re.compile(r'/spudservice/data/\d+'))
        db_key = None
        filename = None
        for link in links:
            text = link.text.strip()
            if text.endswith('_DB.csv'):
                match = re.search(r'/spudservice/data/(\d+)', link['onclick'])
                if match:
                    db_key = match.group(1)
                    filename = text
                    break
        
        if db_key and filename:
            file_path = os.path.join(DB_DIR, filename)
            downloaded_files.append(file_path)
            if not os.path.exists(file_path):
                print(f"  Downloading {filename}...")
                dl_url = f"https://ds.iris.edu/spudservice/data/{db_key}"
                db_resp = requests.get(dl_url)
                with open(file_path, 'w') as f:
                    f.write(db_resp.text)
            else:
                print(f"  {filename} already exists.")
        else:
            print(f"  Warning: No DB.csv found for event {rk}")
            
    return downloaded_files

def _prepare_master_catalog(db_files):
    master_records = []
    
    for db_file in db_files:
        dfs = _parse_db_file(db_file)
        if 'events' not in dfs:
            print(f"Skipping {db_file}: no events section.")
            continue
            
        events_df = dfs['events']
        # Extract requested columns, filling missing ones with NA
        event_dict = {}
        for col in EVENT_COLS:
            if col in events_df.columns:
                val = events_df.iloc[0][col]
                event_dict[col] = val if pd.notnull(val) else pd.NA
            else:
                event_dict[col] = pd.NA
                
        has_stations = False
        if 'stations' in dfs and 'sta_nearby' in dfs:
            stations_df = dfs['stations']
            sta_nearby_df = dfs['sta_nearby']
            
            # Merge
            merged = pd.merge(sta_nearby_df, stations_df, on='Sid', how='inner')
            
            # Filter channels
            if 'Channel' in merged.columns:
                merged = merged[merged['Channel'].isin(['BHZ', 'HHZ', 'EHZ'])]
                
            # Filter distance
            if 'StasourceRadiusKm' in merged.columns:
                merged['StasourceRadiusKm'] = pd.to_numeric(merged['StasourceRadiusKm'], errors='coerce')
                merged = merged[merged['StasourceRadiusKm'] <= 200]
                
            # Filter detections
            det_mask = pd.Series(False, index=merged.index)
            for col in ['DetectHF', 'DetectLP', 'DetectVHF']: ## this list is good enough, I have verified
                if col in merged.columns:
                    merged[col] = pd.to_numeric(merged[col], errors='coerce')
                    det_mask = det_mask | (merged[col] == 1)
            
            merged = merged[det_mask]
            
            if not merged.empty:
                merged = merged.sort_values(by='StasourceRadiusKm').head(10)
                for _, row in merged.iterrows():
                    record = event_dict.copy()
                    record['StationNetwork'] = row.get('Network', pd.NA)
                    record['StationName'] = row.get('Name_y', row.get('Name', pd.NA)) # Name can be in both, Name_y from stations
                    if pd.isna(record['StationName']):
                        record['StationName'] = row.get('Name', pd.NA) # Fallback
                    record['StationChannel'] = row.get('Channel', pd.NA)
                    record['StationLocationCode'] = row.get('LocationCode', pd.NA)
                    record['StationLatitude'] = row.get('Latitude', pd.NA)
                    record['StationLongitude'] = row.get('Longitude', pd.NA)
                    record['StationElevation'] = row.get('ElevationMasl', pd.NA)
                    record['RecordSource'] = row.get('Source', pd.NA)

                    record['StationDistance'] = row.get('StasourceRadiusKm', pd.NA)
                    record['DetectHF'] = row.get('DetectHF', pd.NA)
                    record['DetectLP'] = row.get('DetectLP', pd.NA)
                    record['DetectVHF'] = row.get('DetectVHF', pd.NA)
                    
                    master_records.append(record)
                has_stations = True
                
        if not has_stations:
            record = event_dict.copy()
            record['StationNetwork'] = pd.NA
            record['StationName'] = pd.NA
            record['StationChannel'] = pd.NA
            record['StationLocationCode'] = pd.NA
            record['StationLatitude'] = pd.NA
            record['StationLongitude'] = pd.NA
            record['StationElevation'] = pd.NA
            record['RecordSource'] = pd.NA
            record['StationDistance'] = pd.NA
            record['DetectHF'] = pd.NA
            record['DetectLP'] = pd.NA
            record['DetectVHF'] = pd.NA
            master_records.append(record)
            
    master_df = pd.DataFrame(master_records)
    master_df.to_csv(CATALOG_PATH, index=False)
    print(f"Master catalog saved to {CATALOG_PATH} with {len(master_df)} rows.")

def _download_waveforms(output_response="VEL"):
    print(f"Loading master catalog from {CATALOG_PATH}")
    if not os.path.exists(CATALOG_PATH):
        print("Master catalog not found. Please ensure Step 1 has completed successfully.")
        return
        
    df = pd.read_csv(CATALOG_PATH)
    
    # Initialize tracking columns if not present
    if 'if_downloaded' not in df.columns:
        df['if_downloaded'] = False
    if 'StationChannel_Available' not in df.columns:
        df['StationChannel_Available'] = df['StationChannel']
    if 'SignalSamplingRate' not in df.columns:
        df['SignalSamplingRate'] = None
        
    os.makedirs(SAC_OUT_DIR, exist_ok=True)
    
    total_rows = len(df)
    print(f"Total rows in catalog to process: {total_rows}")
    
    for idx, row in df.iterrows():
        if pd.isna(row['StationNetwork']) or pd.isna(row['StationName']):
            df.at[idx, 'if_downloaded'] = False
            continue
            
        event_id = str(row['Eventid']).replace('.0', '')
        event_type = str(row['Type']).replace(' ', '_').replace('/', '_')
        net = str(row['StationNetwork'])
        sta = str(row['StationName'])
        chan = str(row['StationChannel'])
        dist = str(np.round(row['StationDistance'],3))
        location_code = str(row['StationLocationCode'])
        
        filename = f"{event_id}_{event_type}_{net}_{sta}_{chan}_{location_code}_{dist}.sac"
        filepath = os.path.join(SAC_OUT_DIR, filename)
        
        if os.path.exists(filepath):
            df.at[idx, 'if_downloaded'] = True
            print(f"[{idx+1}/{total_rows}] Already downloaded: {filename}")
            continue
            
        # 1. Parse Times Once (Outside the client loop)
        try:
            start_str = str(row['Starttime']).replace('_', '-')
            if '_' in start_str and len(start_str.split()) == 2:
                date_part, time_part = start_str.split()
                date_part = date_part.replace('_', '-')
                if len(time_part) == 6:
                    time_part = f"{time_part[0:2]}:{time_part[2:4]}:{time_part[4:6]}"
                start_str = f"{date_part}T{time_part}"
                
            end_str = str(row['Endtime']).replace('_', '-')
            if '_' in end_str and len(end_str.split()) == 2:
                date_part, time_part = end_str.split()
                date_part = date_part.replace('_', '-')
                if len(time_part) == 6:
                    time_part = f"{time_part[0:2]}:{time_part[2:4]}:{time_part[4:6]}"
                end_str = f"{date_part}T{time_part}"
                
            start_time = UTCDateTime(pd.to_datetime(str(row['Starttime']), errors='coerce'))
            end_time = UTCDateTime(pd.to_datetime(str(row['Endtime']), errors='coerce'))
            
        except Exception:
            try:
                start_time = UTCDateTime(str(row['Starttime']).replace('_', 'T'))
                end_time = UTCDateTime(str(row['Endtime']).replace('_', 'T'))
            except Exception as e:
                print(f"[{idx+1}/{total_rows}] Error parsing dates for {filename}. Start: {row['Starttime']}, End: {row['Endtime']}. Error: {e}")
                df.at[idx, 'if_downloaded'] = False
                df.at[idx, 'StationChannel_Available'] = None
                continue
            
        t1 = start_time - 10
        t2 = end_time + 10
        
        record_source = str(row['RecordSource'])
        if record_source==record_source:
            clients_to_try = list(record_source.split(','))
        else:
            clients_to_try = ['IRIS']

        print(f"[{idx+1}/{total_rows}] Downloading {net}.{sta}.*.*{chan} from {t1} to {t2}...")
        print(f"  -> Data sources to attempt: {clients_to_try}")
        
        download_success = False
        for client_name in clients_to_try:
            try:
                client = Client(client_name)
            except ValueError:
                print(f"  -> Unknown client '{client_name}', skipping...")
                continue
                
            try:
                st = client.get_waveforms(net, sta, '*', '*Z', t1, t2)
                st.merge(fill_value='latest')

                if len(st.select(channel="EHZ")) > 0:
                    st = st.select(channel="EHZ")
                    df.at[idx, 'StationChannel_Available'] = "EHZ"
                elif len(st.select(channel="HHZ")) > 0:
                    st = st.select(channel="HHZ")
                    df.at[idx, 'StationChannel_Available'] = "HHZ"
                elif len(st.select(channel="BHZ")) > 0:
                    st = st.select(channel="BHZ")
                    df.at[idx, 'StationChannel_Available'] = "BHZ" 
                else:
                    raise ValueError(f"No suitable Z channel found via {client_name}.")         

                ## Keep only One Trace by making sure additonal traces for location codes gets removed
                if len(st) > 1:
                    st = st[:1]

                # Response Removal
                inv = client.get_stations(network=net, station=sta, location='*', 
                                          channel=st[0].stats.channel, starttime=t1, 
                                          endtime=t2, level="response")
                
                sample_rate = st[0].stats.sampling_rate
                st.detrend("demean")
                st.detrend("linear")
                st.remove_response(inventory=inv, output=output_response, 
                                   pre_filt=[0.005, 0.01, sample_rate/3, sample_rate/2], 
                                   zero_mean=True, taper=True, taper_fraction=0.05)
                
                # Save and break the loop on success
                st.write(filepath, format="SAC")
                df.at[idx, 'if_downloaded'] = True
                df.at[idx, 'SignalSamplingRate'] = sample_rate
                print(f"  -> SUCCESS: Downloaded via {client_name} and saved as {filename}")
                
                download_success = True
                break  # Exit the client loop, move to the next event
                
            except Exception as e:
                print(f"  -> Failed via {client_name}: {str(e).splitlines()[0]}")
                continue 
                
        # 4. Handle Complete Failure
        if not download_success:
            print(f"  -> Exhausted all data sources. Could not download event.")
            df.at[idx, 'StationChannel_Available'] = None
            df.at[idx, 'if_downloaded'] = False
            
    df.to_csv(CATALOG_PATH, index=False)
    print("Waveforms downloaded and Catalog updated with download status!!")

def _estimate_snr(trace_data):
    max_val = np.max(np.abs(trace_data))
        
    trace_norm = trace_data / max_val
    abs_data = np.abs(trace_norm)    
    noise_floor = np.median(abs_data)
    
    if noise_floor == 0:
        return 0 
        
    signal_amp = np.percentile(abs_data, 95)
    snr_estimate = signal_amp / noise_floor
    return snr_estimate

def _process_waveforms(fmin=1.0, fmax=5.0):
    print(f"Loading master catalog from {CATALOG_PATH}")
    if not os.path.exists(CATALOG_PATH):
        print("Master catalog not found. Please ensure previous steps have completed successfully.")
        return
        
    df = pd.read_csv(CATALOG_PATH)
    if 'if_downloaded' not in df.columns:
        print("if_downloaded column not found in catalog. Run Step 2 first.")
        return
    
    if 'if_processed' not in df.columns:
        df['if_processed'] = False
    
    if 'SNR' not in df.columns:
        df['SNR'] = 0.0
        
    df_down = df[df['if_downloaded'] == True]
    total_rows = len(df_down)
    print(f"Total downloaded files to process: {total_rows}")
    
    os.makedirs(SAC_PROC_DIR, exist_ok=True)
    
    count = 0
    for idx, row in df_down.iterrows():
        count += 1
        event_id = str(row['Eventid']).replace('.0', '')
        event_type = str(row['Type']).replace(' ', '_').replace('/', '_')
        net = str(row['StationNetwork'])
        sta = str(row['StationName'])
        chan = str(row['StationChannel'])
        dist = str(np.round(row['StationDistance'],3))
        location_code = str(row.get('StationLocationCode'))
        
        filename = f"{event_id}_{event_type}_{net}_{sta}_{chan}_{location_code}_{dist}.sac"
        in_filepath = os.path.join(SAC_OUT_DIR, filename)
        out_filepath = os.path.join(SAC_PROC_DIR, filename)
        
        if os.path.exists(out_filepath):
            print(f"[{count}/{total_rows}] Already processed: {filename}")
            df.at[idx, 'if_processed'] = True
            continue
            
        if os.path.exists(in_filepath):
            try:
                st = read(in_filepath)
                st.filter('bandpass', freqmin=fmin, freqmax=fmax, corners=4, zerophase=True)
                st.taper(max_percentage=.05)
                
                st.write(out_filepath, format="SAC")
                print(f"[{count}/{total_rows}] Processed and saved: {filename}")

                df.at[idx, 'SNR'] = _estimate_snr(st[0].data)
                df.at[idx, 'if_processed'] = True
            
            except Exception as e:
                print(f"[{count}/{total_rows}] Error processing {filename}: {e}")
                df.at[idx, 'if_processed'] = False
                df.at[idx, 'SNR'] = 0.0
        else:
            print(f"[{count}/{total_rows}] File not found: {in_filepath}")
            df.at[idx, 'if_processed'] = False
            df.at[idx, 'SNR'] = 0.0
    df.to_csv(CATALOG_PATH, index=False)
    print("Waveforms processed and Catalog updated with processing status!!")

def main(input_db_files, output_response, fmin, fmax):

    if input_db_files:
        db_files = glob.glob(input_db_files)
    else:
        os.makedirs(DB_DIR, exist_ok=True)
        db_files = _download_db_files()

    ## Step B: Prepare the ESEC Catalog (Master File)
    _prepare_master_catalog(db_files)
    
    ## Step C: Download the Waveforms and Remove Responses
    os.makedirs(SAC_OUT_DIR, exist_ok=True)
    _download_waveforms(output_response=output_response)

    ## Step D: Process the Waveforms and Calculate SNR
    os.makedirs(SAC_PROC_DIR, exist_ok=True)
    _process_waveforms(fmin=fmin, fmax=fmax)


##---------------------------- Main ----------------------------
if __name__ == "__main__":
    main(input_db_files=DB_FILES_PATH, output_response="VEL", fmin=1.0, fmax=5.0)
    print("Data Prep and Processing Completed")


##---------------------------- How to Run ----------------------------
## nohup /home/software/miniconda3/envs/spec_master_dev/bin/python -u /data/sswar_files/PRJ_GIS_QA/esce_catalog_test/final_test_results_for_production/step1_esec_catalog_dataprepandprocess.py > /data/sswar_files/PRJ_GIS_QA/esce_catalog_test/final_test_results_for_production/terminal_outputs/step1_esec_catalog_dataprepandprocess_prod_test.out 2>&1 &
## 740255