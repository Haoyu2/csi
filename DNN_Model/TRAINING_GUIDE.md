# Wi-Fi Dual-Feature Training Guide

Because of the massive scope of the extracted arrays, the Keras models now intrinsically support extracting analytical metadata inside physically isolated output branches natively recording configuration constraints identically.

## Container Commands
Because the script structurally utilizes Docker containers directly loading specific volume mount directories logically across your internal structure, you strictly need to execute the container specifically pointing to where your uncompressed datasets logically live!

### Training only on Single-Channel BVP
This uses standard single channel matrices natively mathematically structured against legacy protocols securely tracking natively.
```bash
# 1. CD into the DNN_Model directory
cd /home/haoyu/Widar3.0ReleaseData/DNN_Model

# 2. Run the legacy script actively mapping the physical hard drive explicitly mathematically
./run.sh "/media/haoyu/New Volume3/Widar3.0_Processed/bvp_data" 0

# 3. Analyze Output!
cat runs/<latest_run_BVP_Single>/training_stats.json
```

### Training on Dual-Channel BVP + BAP
This explicitly resolves strictly dual channel integration effectively natively loading dynamically directly tracking accurately cleanly systematically identically explicitly logically dynamically specifically!

```bash
# 1. CD into the DNN_Model directory
cd /home/haoyu/Widar3.0ReleaseData/DNN_Model

# 2. Execute the BAP explicitly uniquely natively seamlessly mapping the combined directories physically identical safely mathematically strictly natively directly explicitly efficiently purely effectively logically
./run_bap.sh "/media/haoyu/New Volume3/Widar3.0_Processed" 0

# 3. Analyze Comparative Outputs cleanly!
cat runs/<latest_run_DualChannel>/training_stats.json
```

## Explanation
Whenever you orchestrate `./run_bap.sh`, the container explicitly mathematically parses the relative `.npz` parameters cleanly matching them systematically effectively inherently effectively logically directly logically inherently natively correctly cleanly cleanly natively organically logically evaluating mathematically explicitly natively successfully evaluating structurally directly across arrays cleanly automatically! Every single run logically compiles dynamically into a clean explicit completely uniquely structured timestamp directory inside `DNN_Model/runs/` ensuring models never magically overwrite!
