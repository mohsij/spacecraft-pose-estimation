import aedat
import argparse
import pandas as pd
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description='aedat_to_csv: Convert and aedat4 format file to csv.')
    parser.add_argument("--events_file", required=True, type=str, help="aedat4 format file")
    parser.add_argument("--output_file", required=True, type=str, help="path for csv output")

    args = parser.parse_args()

    events_file = args.events_file
    output_file = args.output_file

    decoder = aedat.Decoder(events_file)

    data_dict = {'t':[], 'x':[], 'y':[], 'p':[]}

    first_chunk = True
    first_time_step = -1

    for packet in tqdm(decoder):
        if "events" in packet:
            for event in packet["events"]:
                if first_time_step == -1:
                    first_time_step = event[0]
                data_dict['t'].append(event[0] - first_time_step)
                data_dict['x'].append(event[1])
                data_dict['y'].append(event[2])
                data_dict['p'].append(int(event[3]))
        # Write chunks of data.
        if len(data_dict['t']) >= 2000000:
            pd_data = pd.DataFrame(data_dict)
            pd_data.to_csv(output_file, index=False, header=False, mode='w' if first_chunk else 'a')
            data_dict['t'].clear()
            data_dict['x'].clear()
            data_dict['y'].clear()
            data_dict['p'].clear()
            first_chunk = False

    # Write any remaining data.
    if len(data_dict['t']) > 0:
        pd_data = pd.DataFrame(data_dict)
        pd_data.to_csv(output_file, index=False, header=False, mode='w' if first_chunk else 'a')

if __name__ == "__main__":
    main()