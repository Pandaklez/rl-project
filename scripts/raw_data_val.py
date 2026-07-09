import numpy as np
import h5py
import argparse
import json
import os

def compare_index_files(json1,json2):
    print("comparing index files...")
    index1 = json.load(open(json1,"r"))
    index2 = json.load(open(json2,"r"))

    k1, k2 = index1.keys(),index2.keys()

    if k1 != k2:
        print(f"different keys: {k1} vs {k2}")

    for split in k1:
        clips1, clips2 = index1[split],index2[split]

        l1,l2 = len(clips1),len(clips2)
        if l1 != l2:
            print(f"Different length for split {split}, {l1} vs {l2}")

        for i,(clip1, clip2) in enumerate(zip(clips1, clips2)):
            if clip1 != clip2: 
                print(f"Different clip names in split {split} at index {i}, {clip1} vs {clip2}")
    print("Split index files done")
    print()

def compare_index_to_h5(h5, indexes):
    print("comparing index to h5...")
    n_missing = 0
    n_misplaced = 0
    for split in ["train", "val", "test"]:
        h5_clips = list(h5[split].keys())
        index_clips = indexes[split]
        for i,index_clip in enumerate(index_clips):
            if index_clip not in h5_clips:
                print(f"In split {split}: Index clip {index_clip} (idx {i}) not in H5")
                n_missing += 1

            for other_split in ["train", "val", "test"]:
                other_h5_clips = list(h5[other_split].keys())
                if other_split != split and index_clip in other_h5_clips:
                    print(f"In split {split}: Index clip {index_clip} (idx {i}) found in other split {other_split}")
                    n_misplaced += 1
        # for i,(h5_clip, index_clip) in enumerate(zip(h5_clips, index_clips)):
        #     if h5_clip != index_clip: 
        #         print(f"Different clip names in split {split} at index {i}, {h5_clip} vs {index_clip}")
    print(f"Split index vs h5 done, n_missing: {n_missing}, n_misplaced: {n_misplaced}")
    print()

def val_norm_stats(data, split, camera, ds): 
    print()
    print("stats check")
    print(f"{split} - {camera} - {ds}: ")
    n_missing = 0
    raw_data = []
    for clip in data[split].keys():
        try:
            row = data[split][clip][camera]
        except:
            # print(f"Could not read split {split}, clip {clip}, camera {camera}")
            n_missing += 1
            continue
        # if camera: 
        #     try:
        #         row = data[split][clip][camera]
        #     except:
        #         # print(f"Could not read split {split}, clip {clip}, camera {camera}")
        #         n_missing += 1
        #         continue
        # else:
        #     row = data[split][clip]
        raw_data.append(row[ds])

    if ds == "betas":
        np_data = np.stack(raw_data,axis = 0)
        print(f"n_missing: {n_missing}")
        print(f'\t shape: {np_data.shape}')
        print(f'\t mu: {np_data.mean(axis=0).mean()}')
        print(f'\t sigma: {np_data.std(axis = 0).mean()}')

    else:
        np_data = np.concat(raw_data,axis = 0)
        print(f"n_missing: {n_missing}")
        print(f'\t shape: {np_data.shape}')
        print(f'\t mu: {np_data.mean(axis=0).mean().mean()}')
        print(f'\t sigma: {np_data.std(axis = 0).mean().mean()}')
    return n_missing

                

def rmse(a, b):
    T = min(len(a), len(b))
    return np.sqrt(np.mean((a[:T] - b[:T]) ** 2))


def val_rmse(h5):
    datasets = ["poses", "trans", "betas"]

    for split in ["train", "val", "test"]:
        print(f"\n===== {split.upper()} =====")

        clips = [
            clip
            for clip in h5[split].keys()
            if "pg1" in h5[split][clip] and "pg2" in h5[split][clip]
        ]

        for ds in datasets:
            pg1_self = []
            pg1_other = []
            pg2_self = []
            pg2_other = []

            for i, clip in enumerate(clips):
                other_clip = clips[i - 1]  

                gt = np.asarray(h5[split][clip]["gt"][ds])
                gt_other = np.asarray(h5[split][other_clip]["gt"][ds])

                pred_pg1 = np.asarray(h5[split][clip]["pg1"][ds])
                pred_pg2 = np.asarray(h5[split][clip]["pg2"][ds])

                e1_self = rmse(pred_pg1, gt)
                e1_other = rmse(pred_pg1, gt_other)

                e2_self = rmse(pred_pg2, gt)
                e2_other = rmse(pred_pg2, gt_other)

                pg1_self.append(e1_self)
                pg1_other.append(e1_other)
                pg2_self.append(e2_self)
                pg2_other.append(e2_other)

            pg1_acc = np.mean(np.array(pg1_self) < np.array(pg1_other))
            pg2_acc = np.mean(np.array(pg2_self) < np.array(pg2_other))

            print(
                f"{ds:6s} | "
                f"PG1: {pg1_acc:.1%} "
                f"(RMSE {np.mean(pg1_self):.4f} vs {np.mean(pg1_other):.4f}) | "
                f"PG2: {pg2_acc:.1%} "
                f"(RMSE {np.mean(pg2_self):.4f} vs {np.mean(pg2_other):.4f})"
            )




        

    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_index",  default="data/split_index.json",
                        help="JSON with {'train': [...], 'val': [...], 'test': [...]}")
    parser.add_argument("--old_split_index",  default="data/_backup_split_index.json",
                        help="JSON with {'train': [...], 'val': [...], 'test': [...]}")
    parser.add_argument("--processed_h5",    default="data/processed_movi.h5",
                        help="Output path for the normalized, merged HDF5")

    args = parser.parse_args()
    split_path = os.path.join(os.getcwd(),args.split_index)
    old_split_path = os.path.join(os.getcwd(),args.old_split_index)
    data_path = os.path.join(os.getcwd(),args.processed_h5)

    data = h5py.File(data_path,"r")
    split_index = json.load(open(split_path,"r"))

    compare_index_files(split_path,old_split_path)
    compare_index_to_h5(data,split_index)

    tot_missing = 0
    
    for split in ["train","val", "test"]:
        for camera in ("gt","pg1","pg2"):
            for ds in ["poses","trans","betas"]:
                tot_missing += val_norm_stats(data, split, camera, ds)

    print(f"Post stats check, n_missing: {tot_missing}\n")

    val_rmse(data)



if __name__ == '__main__':
    main()