import torch
import torch.nn.utils.prune as prune
import os
import time
import glob
from ultralytics import YOLO


# ==========================================
# 1️⃣ LOAD MODEL
# ==========================================
print("🚀 Loading YOLOv8 Classification Model...")
model = YOLO("yolov8n-cls.pt")


# ==========================================
# 2️⃣ TRAIN MODEL (WITH GRAPHS)
# ==========================================
print("📚 Starting Training...")

results = model.train(
    data=r"C:\Users\yaswa\OneDrive\Desktop\Model Compression and Pruning\archive (1)\dataset",
    epochs=10,
    imgsz=224,
    batch=16,
    plots=True   # ⭐ IMPORTANT → Generates graphs
)

print("✅ Training Completed")

# ==========================================
# 📊 SHOW GRAPH LOCATION
# ==========================================
print("\n📊 Graphs saved in:", results.save_dir)
print("👉 Open 'results.png' for Accuracy & Loss graph")


# ==========================================
# 3️⃣ LOAD LATEST TRAINED MODEL
# ==========================================
print("\n📦 Loading trained model...")

runs = sorted(glob.glob("runs/classify/train*"))
best_model_path = runs[-1] + "/weights/best.pt"

print("Using model:", best_model_path)

yolo_model = YOLO(best_model_path)
pytorch_model = yolo_model.model


# ==========================================
# 4️⃣ APPLY PRUNING
# ==========================================
print("\n✂️ Applying pruning...")

for name, module in pytorch_model.named_modules():
    if isinstance(module, torch.nn.Conv2d):
        prune.l1_unstructured(module, name='weight', amount=0.3)

# Make pruning permanent
for name, module in pytorch_model.named_modules():
    if isinstance(module, torch.nn.Conv2d):
        prune.remove(module, 'weight')


# ==========================================
# 5️⃣ SAVE PRUNED MODEL
# ==========================================
pruned_model_path = "pruned_model.pt"
torch.save(pytorch_model.state_dict(), pruned_model_path)

print("✅ Pruned model saved")


# ==========================================
# 6️⃣ MODEL SIZE COMPARISON
# ==========================================
original_size = os.path.getsize(best_model_path) / 1e6
pruned_size = os.path.getsize(pruned_model_path) / 1e6

print("\n📏 Model Size Comparison")
print("Original Model:", round(original_size, 2), "MB")
print("Pruned Model:", round(pruned_size, 2), "MB")


# ==========================================
# 7️⃣ LOAD PRUNED MODEL
# ==========================================
print("\n🔄 Loading pruned model for testing...")

test_model = YOLO(best_model_path)
test_model.model.load_state_dict(torch.load(pruned_model_path))


# ==========================================
# 8️⃣ TEST IMAGE
# ==========================================
test_image = r"C:\Users\yaswa\OneDrive\Desktop\Model Compression and Pruning\archive (1)\dataset\train\wild\wild (1).jpg"

print("\n🔍 Running Prediction...")

results = test_model(test_image)

print(results)


# ==========================================
# 9️⃣ PERFORMANCE (FPS)
# ==========================================
start_time = time.time()

test_model(test_image)

end_time = time.time()

inference_time = end_time - start_time
fps = 1 / inference_time

print("\n⚡ Performance")
print("Inference Time:", round(inference_time, 4))
print("FPS:", round(fps, 2))


# ==========================================
# 🔟 FINAL MESSAGE
# ==========================================
print("\n🎉 Project Completed Successfully")