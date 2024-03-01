import os
import time
import argparse
import json

JSONS_PATH = "../common/jsons/"
BLENDFILE_PATH = "../common/empty_scene.blend"

def get_id_info(path, dataset_type):
    if dataset_type == "modelnet":
        category = path.split("/")[-3]
        obj_id = path.split("/")[-1][:-4]
        return category, obj_id

    if dataset_type == "shapenet":
        category = path.split("/")[-4]
        obj_id = path.split("/")[-3]
        return category, obj_id

    if dataset_type == "toys":
        category = path.split("/")[-3]
        obj_id = path.split("/")[-2]
        return category, obj_id
    
    if dataset_type == "toys200":
        obj_id = path.split('/')[-1]
        category = ""
        return category, obj_id

def main():
    
    # set paths
    blender_script_path = os.path.abspath("sample_pose.py")
    blendfile_path = os.path.abspath(BLENDFILE_PATH)
    
    # load arguments
    parser = argparse.ArgumentParser(description="Range of Objects")
    parser.add_argument("--start", type=int, help="start point")
    parser.add_argument("--end", type=int, help="end point")
    parser.add_argument("--input_path", type=str, help="dataset input path")
    parser.add_argument("--output_path", type=str, help="render output path")
    parser.add_argument("--blender_path", type=str, help="path to blender")
    parser.add_argument(
        "--dataset_type", type=str, help="either <modelnet>, <shapenet> or <toys>"
    )
    parser.add_argument("--orientation", type=str)
    parser.add_argument("--n_samples", type=int, default=50)

    args = parser.parse_args()

    print("Start index: {}".format(args.start))
    print("End index: {}".format(args.end))
    print("Input path: {}".format(args.input_path))
    print("Output path: {}".format(args.output_path))
    print("Dataset type: {}".format(args.dataset_type))

    data_json_path = os.path.join(JSONS_PATH, "{}_dict.json".format(args.dataset_type))

    with open(data_json_path, "r") as f:
        data_dict = json.load(f)
    
    # collect paths
    paths = []
    for category, object_paths in data_dict.items():
        object_paths = [
            os.path.join(args.input_path, category, x) for x in object_paths
        ]
        paths.extend(object_paths)
    
    i = 0
    global_time = time.time()
    
    # render each object
    for idx, p in enumerate(paths[args.start : args.end]):
        start_time = time.time()
        fpath = os.path.join(args.input_path, p)
        render_output_path = os.path.join(
            args.output_path, *get_id_info(fpath, args.dataset_type)
        )

        cmd = (
            f"{args.blender_path} -noaudio --background {blendfile_path} --python {blender_script_path} -- "
            f"--dataset_type {args.dataset_type} "
            f"--output_path {render_output_path} "
            f"--input_path {fpath} "
            f"--orientation {args.orientation} "
            f"--n_samples {args.n_samples} "
        )   #+ "1>/tmp/out.txt"

        os.system(cmd)
        print(cmd)

        print(
            "--- {:.2f} seconds for obj {} [{}/{}] ---".format(
                time.time() - start_time, fpath, args.start + i, args.end
            )
        )

        if i % 50 == 0:
            print(
                "Total time since start is {:.2f} minutes".format(
                    (time.time() - global_time) / 60
                )
            )
        i += 1

    total_time = time.time() - global_time
    print("Total time {:.2f}".format(total_time / 60))


if __name__ == "__main__":
    main()
