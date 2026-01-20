import argparse
import json
import subprocess
from enum import Enum
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd

from IPython import embed

from datasets import load_tsv_datasets, create_bursts, create_bursts_unknown
from models import (
    KDCandidatePredictor,
    KDPointsTransformer,
    CandidateScoreTransformer,
    CandidateSelector,
)

burstshark_path = "./burstshark-x86_64-linux"
def create_bursts(pcap_path, filter):
    
    program_args = [
        burstshark_path,
        "-r",
        pcap_path,
        "-Y",
        filter,
        "--min-bytes",
        "50000"
    ] 

    program_args.append("-E")
    program_args.append("--wlan")

    completed = subprocess.run(program_args, capture_output=True, text=True)

    if completed.returncode != 0:
        raise Exception(f'Burstshark error: "{completed.stderr.strip()}"')

    lines = completed.stdout.strip().split("\n")
    result_times = []
    result_size = []
    result_service = []

    for line in lines:
        cols = line.split()
        burst_start_ts, burst_end_ts, burst_size = cols[6], cols[7], cols[10]

        result_times.append(burst_end_ts)
        result_size.append(burst_size)
        result_service.append("amazon")

        # result.append(f"{burst_end_ts} {burst_size} amazon")

    return result_times, result_size, result_service

def load_dataset(recompute) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("Loading dataset...", end="", flush=True)
    if recompute:
        df_videos, df_representations = load_tsv_datasets()
    else:
        df_videos = pd.read_parquet("./precomputed/tsv/videos.parquet")
        df_representations = pd.read_parquet("./precomputed/tsv/representations.parquet")
    print("\rLoading dataset... DONE", flush=True)

    return df_videos, df_representations

def load_kd(
    df_representations: pd.DataFrame, recompute, all
) -> tuple[KDPointsTransformer, dict[str, KDCandidatePredictor]]:
    
    # Do all the Fallout videos and another series too and see what happens
    target_videos = [
        'amzn1.dv.gti.220a83cb-b9b4-4cad-b8a0-e768bb603fa7',
        'amzn1.dv.gti.067864b0-18a9-407f-a784-3ca9adfc85c7',
        'amzn1.dv.gti.46823027-bc86-4148-b6bd-d053b302077c',
        'amzn1.dv.gti.ef369b14-5eb5-4558-b5ce-f9a1ed0d5816',
        'amzn1.dv.gti.e156cbf3-a6a5-4efd-808a-0c34b4393127',
        'amzn1.dv.gti.74820634-9d96-49d1-bd4d-706dce4b9af2',
        'amzn1.dv.gti.28f66f4f-d342-4446-bfa0-2a12248a3974',
        'amzn1.dv.gti.8276269a-402e-4ece-a2b0-4eb5e2504a05',
         # Tom Clancy Season 1
        'amzn1.dv.gti.56b2af4a-833b-5d9e-32be-368db8e46fab',
        'amzn1.dv.gti.68b2af50-8785-37b2-e65a-1182e11651ca',
        'amzn1.dv.gti.80b2af51-18c5-8ee8-ef00-c28f0feb87e4',
        'amzn1.dv.gti.32b2af53-56c9-8b42-7aab-e29a39246d93',
        'amzn1.dv.gti.e2b2af59-ff4f-f318-c35c-df2543d31cde',
        'amzn1.dv.gti.52b2af5b-d498-7184-636a-ae87e5945574',
        'amzn1.dv.gti.7eb2af5d-97df-1061-e1ba-15fe2b502b9e',
        'amzn1.dv.gti.f8b2af60-4c42-e2ca-cf9f-0ba2eb446491',
        'amzn1.dv.gti.16b6f808-662a-5a55-7a3e-3fc89f790ff5',
        'amzn1.dv.gti.d2b6f7c1-23d5-534d-73c9-32729b10d0dc',
        'amzn1.dv.gti.58b6f7c1-f204-370f-4134-8907df24a7fe',
        'amzn1.dv.gti.7cb6f7c1-23f8-32e9-3be0-019dc2f4010f',
        'amzn1.dv.gti.b2b6f7c1-92fb-e9e0-03d4-d96511bf1456',
        'amzn1.dv.gti.eab6f7c1-22fa-a912-0f0b-8e6f2785cd65',
        'amzn1.dv.gti.26b6f7c1-2218-79d3-60fb-2e2e9cf3c7c9',
        'amzn1.dv.gti.00b6f7c2-6421-f898-adaa-8536ca832a69',
        'amzn1.dv.gti.d52a4bd0-cbe7-45b1-b469-52d32d7144e2',
        'amzn1.dv.gti.71b74ef4-4e7f-4de0-9eff-248ee3e87bdc',
        'amzn1.dv.gti.d86951ae-6640-4d37-b7b5-1ea6bcbdd0b7',
        'amzn1.dv.gti.24037dcc-07a3-4b58-9d5f-c0cea73079f3',
        'amzn1.dv.gti.6ecd1e49-835e-47ca-8f99-1b90626990ca',
        'amzn1.dv.gti.320b3658-da1a-4c66-aadf-e0f4aa4d825c',
        'amzn1.dv.gti.fccbbff2-a565-451c-a399-7cdea92ae935',
        'amzn1.dv.gti.7b74cdd8-e3ad-451f-8d3b-4574b8ec2a6c',
        'amzn1.dv.gti.784e76b6-e15f-498a-8d3d-d50e17c17a38',
        'amzn1.dv.gti.556b61b9-9902-4bc6-b542-68f9646c9e08',
        'amzn1.dv.gti.64539ad6-3878-4ad2-931b-e57a8da3855f',
        'amzn1.dv.gti.0e69e513-2c5e-4542-a404-b2a1a352f59a',
        'amzn1.dv.gti.9ed02eac-ae27-4b64-a603-2f2216c4a0da',
        'amzn1.dv.gti.23411853-f8ab-4e90-8765-d88e6ee2bfe2'
    ]

    print("target_videos length", len(target_videos))

    # Only build kd-tree with target videos
    if not all:
        df_representations = df_representations[df_representations["video"].isin(target_videos)]

    X = df_representations["segment_sizes"].to_numpy()
    y = df_representations.index.to_numpy(dtype="uint32")

    kdp = KDPointsTransformer().fit(X)
    kdt = {}
    if recompute:
        for service, ind in df_representations.groupby("service").indices.items():
            print(f"Building k-d tree for {service}...", end="", flush=True)
            kdt[service] = KDCandidatePredictor().fit(kdp.transform(X[ind]), y[ind])
            print(f"\rBuilding k-d tree for {service}... DONE", flush=True) 

    return kdp, kdt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="+", type=Path)
    parser.add_argument("-a", "--all", action="store_true", help="Generate entire kd-tree")
    parser.add_argument("-b", "--burst", action="store_true", help="Using a burst file instead of a pcap")
    args = parser.parse_args()

    df_videos, df_representations = load_dataset(recompute = False) # Use the segments we already have
    kdp, kdt = load_kd(df_representations, recompute = True, all = args.all) # Make a new kd-tree

    cst = CandidateScoreTransformer()
    cs = CandidateSelector(2.2)
    
    print("\n")

    for path in args.path:

        if args.burst:
            burst_data = np.loadtxt(
                path,
                dtype=[
                    ("times", "float64"),
                    ("bursts", "int32"),
                    ("services", "U6"),
                ],
            )

            times, bursts, services = (
                burst_data["times"],
                burst_data["bursts"],
                burst_data["services"]
            )
        else:
            times, bursts, services = create_bursts(path, "wlan.ra == 80:a9:97:41:2b:09 && wlan.sa")

            times = np.array(times, dtype=np.float64)
            bursts = np.array(bursts, dtype=np.int32)
            services = np.array(services, dtype="U6")

        unique_services, indices = np.unique(services, return_index=True)
        service_kd_pred = []
        for service in unique_services[np.argsort(indices)]:
            kd_pred = kdt[service].predict(
                kdp.transform([bursts[services == service]])
            )[0]
            service_kd_pred.append(kd_pred)

        i_and_ell = np.concatenate([kd_pred[0] for kd_pred in service_kd_pred], axis=0)
        distances = np.concatenate([kd_pred[1] for kd_pred in service_kd_pred], axis=0)

        out = cs.predict(cst.transform([(i_and_ell, distances)]))[0]
        
        identified =  np.unique(out[:, 0])
        identified = identified[identified != -1]
        video_names = df_representations.iloc[identified]["video"]
        video_titles = df_videos.loc[video_names]["title"]

        print("Currently analyzing:", path)
        print("Identified the following: ", list(video_titles))

        first_identification = int(np.argmax(out[:, 0] != -1))
        first_identification += 8 # because of window size (k=8)
        time_delta = times[first_identification] - times[0]
        if len(video_titles):
            print("First identification happend", time_delta, "seconds into sniffing")
        else:
            print("No identifications!")

        print("\n")

    embed()

if __name__ == "__main__":
    main()
