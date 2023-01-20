import os
import fiftyone.zoo as foz
import fiftyone as fo

def getDataset(split, dataset_name="litepose-coco", fiftyonepath=os.path.join(os.environ.get("HOME"),"fiftyone")):

    def _cleanNone(ds):
        remIds = [p["id"] for p in (iter(ds)) if p["keypoints"] == None]
        ds.delete_samples(remIds)

    # download the dataset if it doesn't exist
    dataset = foz.load_zoo_dataset(
        "coco-2017",
        split=split,
        dataset_name=dataset_name
    )

    labels_file = os.path.join(fiftyonepath, "coco-2017/raw/person_keypoints_val2017.json")
    dataset_file = os.path.join(fiftyonepath, "coco-2017/validation")

    ds = fo.Dataset.from_dir(
        dataset_type = fo.types.COCODetectionDataset,
        label_types = ["detections", "keypoints"],
        dataset_dir = dataset_file,
        labels_path = labels_file
    )

    # avoid the Dataset load each time
    ds.persistent = True

    print("Dataset cleaning...")
    _cleanNone(ds)
    print("Done!")

    return ds