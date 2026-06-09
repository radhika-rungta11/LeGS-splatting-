import os
import json


scene_names = {
    "mipnerf360": ["bicycle", "flowers", "garden", "stump", "treehill", "room", "counter", "kitchen", "bonsai"],
    "tanks_and_temples": ["truck", "train"],
    "deep_blending": ["drjohnson", "playroom"]
}
output_path = "output/rl"

for dataset in scene_names.keys():
    print(dataset)
    SSIM, PSNR, LPIP, NGS = 0., 0., 0., 0.
    peak_memory = 0.
    training_time = 0.
    for scene in scene_names[dataset]:
        result_path = os.path.join(output_path, scene, "results.json")
        training_statistics_path = os.path.join(output_path, scene, "training_statistics.json")

        if not os.path.exists(result_path):
            continue

        with open(result_path, "r") as f:
            results = json.load(f)["ours_30000"]
            SSIM += results["SSIM"]
            PSNR += results["PSNR"]
            LPIP += results["LPIPS"]

        with open(training_statistics_path, "r") as f:
            training_statistics = json.load(f)
            NGS += int(training_statistics["GS_number"])
            training_time += float(training_statistics["total"]['time_s'])
            peak_memory = max(peak_memory, float(training_statistics["peak_gpu_memory_mb"]))

    SSIM /= len(scene_names[dataset])
    PSNR /= len(scene_names[dataset])
    LPIP /= len(scene_names[dataset])
    NGS /= len(scene_names[dataset])
    training_time /= len(scene_names[dataset])
    
    NGS /= 1e6
    training_time /= 60.

    print(f"SSIM: {SSIM:.4f}, PSNR: {PSNR:.4f}, LPIP: {LPIP:.4f}, NGS: {NGS:.4f}, Training time: {training_time:.4f} m, Peak Memory: {peak_memory:.4f} MB")
